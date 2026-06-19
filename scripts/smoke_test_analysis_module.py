"""Single end-to-end smoke test for the /analysis module.

Checks (in order): required directories, third-party dependencies, the model
registry (PCA/PLS-DA/SVM/Random Forest/Decision Tree/PLS Regression/SVR/k-means/HCA),
preprocessing on a synthetic spectrum, training on a synthetic dataset for every
task type, model save/load/predict/delete, and export (JSON/CSV/XLSX/ZIP) through
the real FastAPI routes (in-process, no server needs to be running).

Usage:
    python scripts/smoke_test_analysis_module.py

Exit code is 0 when status is "ok" or "warning", 1 when "error" (so it is CI-friendly).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: List[Dict[str, Any]] = []


def run_check(name: str, fn: Callable[[], str]) -> None:
    start = time.perf_counter()
    try:
        message = fn() or "ok"
        status = "ok"
    except AssertionError as exc:
        message = f"assertion failed: {exc}"
        status = "error"
    except Exception as exc:  # noqa: BLE001 - smoke test must keep going and report everything
        message = f"{type(exc).__name__}: {exc}"
        status = "error"
    CHECKS.append({"name": name, "status": status, "message": message, "time_sec": round(time.perf_counter() - start, 4)})


def check_directories() -> str:
    required = ["uploads", "saved_models", "saved_runs", "user_data"]
    created = []
    for name in required:
        path = ROOT / name
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(name)
    smoke_exports = ROOT / "user_data" / "_smoke_test" / "exports"
    smoke_exports.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "user_data" / "_smoke_test", ignore_errors=True)
    if (ROOT / "datasets").exists():
        pass  # legacy/optional directory, presence is informational only
    return f"required directories present (created missing: {created or 'none'})"


def check_dependencies() -> str:
    versions = {}
    for module_name in ["numpy", "pandas", "scipy", "sklearn", "fastapi", "plotly", "joblib"]:
        module = __import__(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return json.dumps(versions)


def check_model_registry() -> str:
    from services.analysis.models import create_model, supported_model_types

    expected = {"pca", "plsda", "svm", "random_forest", "decision_tree", "pls", "svr", "kmeans", "hca"}
    supported = set(supported_model_types())
    missing = expected - supported
    assert not missing, f"missing from supported_model_types(): {missing}"
    for model_type in expected:
        model = create_model(model_type=model_type)
        assert model.model_type == model_type, f"{model_type}: model_type mismatch ({model.model_type})"
    try:
        create_model(model_type="mcr_als")
        raise AssertionError("MCR-ALS should not be creatable (variant A: removed, not a stub)")
    except ValueError as exc:
        assert "not implemented" in str(exc).lower(), f"unexpected MCR-ALS error message: {exc}"
    return f"{len(expected)} model types instantiate correctly; MCR-ALS correctly rejected"


def _synthetic_spectrum():
    import numpy as np

    axis = np.linspace(400.0, 1800.0, 120)
    baseline = 0.0005 * axis
    peak = 5.0 * np.exp(-((axis - 1000.0) ** 2) / (2 * 25.0 ** 2))
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.05, size=axis.shape)
    return axis, baseline + peak + noise


def check_preprocessing() -> str:
    import numpy as np
    from data_processing import baseline_als, smooth_signal, normalize_snv, filter_frequency_range

    axis, amplitude = _synthetic_spectrum()

    baseline = baseline_als(amplitude, lam=1e5, p=0.01)
    assert baseline.shape == amplitude.shape and np.all(np.isfinite(baseline)), "baseline_als produced invalid output"

    smoothed = smooth_signal(amplitude, window_length=11, polyorder=3)
    assert smoothed.shape == amplitude.shape and np.all(np.isfinite(smoothed)), "smooth_signal produced invalid output"

    normalized = normalize_snv(amplitude)
    assert abs(float(np.mean(normalized))) < 1e-6, "normalize_snv did not center the signal"
    assert abs(float(np.std(normalized)) - 1.0) < 1e-6, "normalize_snv did not scale to unit variance"

    cropped_axis, cropped_amp = filter_frequency_range(axis, amplitude, 600.0, 1200.0)
    assert cropped_axis.size < axis.size and cropped_axis.size > 0, "filter_frequency_range did not crop the range"
    assert float(np.min(cropped_axis)) >= 600.0 and float(np.max(cropped_axis)) <= 1200.0, "crop produced out-of-range axis values"

    return "baseline correction, smoothing, SNV normalization and crop all produced valid output"


def _synthetic_dataset(n_samples: int = 24, n_features: int = 16, seed: int = 0):
    import numpy as np
    from services.analysis.spectrum_loader import SpectrumDataset

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    half = n_samples // 2
    x[half:] += 1.8
    axis = np.linspace(400.0, 1800.0, n_features)
    dataset = SpectrumDataset(
        x=x,
        spectral_axis=axis,
        sample_names=[f"s{i}" for i in range(n_samples)],
        source_format="smoke_test",
        filename="smoke_test.csv",
        file_format="csv",
        metadata={},
    )
    class_labels = np.array(["A"] * half + ["B"] * (n_samples - half))
    numeric_target = rng.normal(loc=0.0, scale=1.0, size=n_samples) + (x[:, 0] * 0.5)
    return dataset, class_labels, numeric_target


def check_training_all_task_types() -> str:
    from services.analysis.analysis_service import AnalysisService

    dataset, class_labels, numeric_target = _synthetic_dataset()
    summaries = []
    with tempfile.TemporaryDirectory() as tmp:
        service = AnalysisService(model_root=Path(tmp))

        out = service.train_and_save(
            model_type="plsda",
            dataset=dataset,
            raw_targets="\n".join(class_labels.tolist()),
            n_components=None,
            model_name="smoke_plsda",
            do_validation=True,
            test_size=0.3,
            random_state=42,
            hyperparameter_search=True,
            refit_on_full_data=True,
        )
        assert out["outputs"]["type"] == "classification"
        assert out["training_policy"]["refit_on_full_data"] is True
        assert out["validation"]["kfold"]["status"] == "ok"
        assert out["validation"]["bootstrap"]["status"] == "ok"
        assert out["validation"]["permutation_test"]["status"] == "ok"
        assert out["hyperparameter_search"]["status"] == "ok"
        summaries.append("classification(plsda)+grid_search=ok")

        out = service.train_and_save(
            model_type="pls",
            dataset=dataset,
            raw_targets="\n".join(map(str, numeric_target.tolist())),
            n_components=None,
            model_name="smoke_pls",
            do_validation=True,
            test_size=0.3,
            random_state=42,
        )
        assert out["outputs"]["type"] == "regression"
        assert out["validation"]["kfold"]["status"] == "ok"
        summaries.append("regression(pls)=ok")

        out = service.train_and_save(
            model_type="kmeans",
            dataset=dataset,
            raw_targets="\n".join(class_labels.tolist()),
            n_components=None,
            model_name="smoke_kmeans",
            do_validation=True,
            test_size=0.3,
            random_state=42,
        )
        assert out["outputs"]["type"] == "clustering"
        summaries.append("clustering(kmeans)=ok")

        out = service.train_and_save(
            model_type="pca",
            dataset=dataset,
            raw_targets=None,
            n_components=None,
            model_name="smoke_pca",
            do_validation=True,
            test_size=0.3,
            random_state=42,
        )
        assert out["outputs"]["type"] == "pca"
        summaries.append("pca=ok")

    return "; ".join(summaries)


def check_model_save_load_predict_delete() -> str:
    from services.analysis.analysis_service import AnalysisService

    dataset, class_labels, _ = _synthetic_dataset()
    tmp = tempfile.mkdtemp()
    try:
        service = AnalysisService(model_root=Path(tmp))
        out = service.train_and_save(
            model_type="svm",
            dataset=dataset,
            raw_targets="\n".join(class_labels.tolist()),
            n_components=None,
            model_name="smoke_svm",
            do_validation=True,
            test_size=0.3,
            random_state=42,
        )
        model_id = out["saved_model"]["model_id"]
        model, metadata = service.model_manager.load(model_type="svm", model_id=model_id)
        assert metadata["model_id"] == model_id
        result = service.infer(model_type="svm", model_id=model_id, dataset=dataset)
        assert result["status"] == "success"
        removed = service.model_manager.delete(model_type="svm", model_id=model_id)
        assert removed["model_removed"] and removed["meta_removed"]
        return f"trained, saved, loaded, predicted and deleted model {model_id}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_hca_infer_rejected() -> str:
    from services.analysis.analysis_service import AnalysisService

    dataset, class_labels, _ = _synthetic_dataset()
    tmp = tempfile.mkdtemp()
    try:
        service = AnalysisService(model_root=Path(tmp))
        out = service.train_and_save(
            model_type="hca",
            dataset=dataset,
            raw_targets="\n".join(class_labels.tolist()),
            n_components=None,
            model_name="smoke_hca",
            do_validation=True,
            test_size=0.3,
            random_state=42,
        )
        assert out["model_metadata"]["can_infer_new_samples"] is False
        model_id = out["saved_model"]["model_id"]
        try:
            service.infer(model_type="hca", model_id=model_id, dataset=dataset)
            raise AssertionError("HCA inference should raise ValueError, but it succeeded")
        except ValueError as exc:
            assert "cannot be used for direct prediction" in str(exc)
        return "HCA correctly documents can_infer_new_samples=false and rejects /infer"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_export_via_routes() -> str:
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    username = f"smoke_{os.getpid()}_{int(time.time() * 1000)}"
    password = "smoke-test-password"
    assert client.post("/register", data={"username": username, "password": password, "confirm_password": password}, follow_redirects=False).status_code == 303
    assert client.post("/login", data={"username": username, "password": password}, follow_redirects=False).status_code == 303

    content = "sample_id,target,400,420,440,460,480,500,520,540\n"
    for idx in range(16):
        group = "A" if idx < 8 else "B"
        offset = 0.0 if group == "A" else 1.5
        values = [1 + offset, 2 + offset, 3 + offset, 2 + offset, 1.5 + offset, 1.2 + offset, 1.0 + offset, 0.8 + offset]
        content += f"s{idx},{group}," + ",".join(map(str, values)) + "\n"

    import_response = client.post("/analysis/dataset/import-standard", files={"file": ("smoke.csv", content.encode("utf-8"), "text/csv")})
    assert import_response.status_code == 200, import_response.text
    dataset_id = import_response.json()["dataset_id"]

    train_response = client.post(
        "/analysis/model/train",
        json={
            "dataset_id": dataset_id,
            "model_type": "compare_classification",
            "model_name": "smoke_compare",
            "validation": {"enabled": True, "random_state": 42, "test_size": 0.3},
        },
    )
    assert train_response.status_code == 200, train_response.text
    train_data = train_response.json()
    run_id = train_data["run_id"]

    # ВКР-mode defaults: validation.hyperparameter_search was NOT specified above, so the
    # compare_classification default (hyperparameter_search=True) must have kicked in for every
    # search-capable method, with unique model_id and a documented best_method_selection.
    method_rows = train_data["run"]["results"]["method_results"]
    assert method_rows, "compare_classification produced no method_results"
    model_ids = [row["model_id"] for row in method_rows]
    assert len(model_ids) == len(set(model_ids)), f"duplicate model_id across comparison methods: {model_ids}"
    for row in method_rows:
        assert row["validation"]["status"] == "ok", f"{row['method']}: validation did not run by default in ВКР mode"
        assert row["hyperparameter_search"]["enabled"] is True, f"{row['method']}: hyperparameter_search not enabled by default in ВКР mode"
        assert row["best_params"] is not None, f"{row['method']}: best_params is null despite hyperparameter_search running"
        assert row["metrics"]["confusion_matrix"] == row["test_outputs"]["confusion_matrix"], f"{row['method']}: metrics.confusion_matrix != test_outputs.confusion_matrix"
        assert row["outputs"]["legacy_outputs_field"] is True and row["outputs"]["outputs_source"] == "full_dataset_diagnostic"
    selection = train_data["run"]["best_method_selection"]
    assert selection["selected_method"] and selection["tie_breakers"], "best_method_selection is incomplete"
    assert train_data["run"]["vkr_tables"]["classification_comparison"], "vkr_tables.classification_comparison is empty"
    assert train_data["pipeline_performance"] and train_data["comparison_performance"], "pipeline_performance/comparison_performance missing from response"

    checks = []
    csv_response = client.get(f"/analysis/runs/{run_id}/csv")
    assert csv_response.status_code == 200 and csv_response.content, "run CSV export failed"
    checks.append("run_csv")

    zip_response = client.get(f"/analysis/runs/{run_id}/export")
    assert zip_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        assert "run_metadata.json" in archive.namelist()
    checks.append("run_zip")

    report_response = client.get(f"/analysis/reports/from-run/{run_id}?format=json")
    assert report_response.status_code == 200
    checks.append("report_json")

    thesis_json = client.get(f"/analysis/runs/{run_id}/thesis-tables?format=json")
    assert thesis_json.status_code == 200
    tables = thesis_json.json()["tables"]
    assert tables["classification"], "thesis classification table is empty"
    checks.append("thesis_tables_json")

    thesis_csv = client.get(f"/analysis/runs/{run_id}/thesis-tables?format=csv")
    assert thesis_csv.status_code == 200 and thesis_csv.content
    checks.append("thesis_tables_csv_zip")

    thesis_xlsx = client.get(f"/analysis/runs/{run_id}/thesis-tables?format=xlsx")
    assert thesis_xlsx.status_code == 200 and thesis_xlsx.content
    checks.append("thesis_tables_xlsx")

    dataset_csv = client.get(f"/analysis/dataset/{dataset_id}/export?format=csv")
    assert dataset_csv.status_code == 200
    checks.append("dataset_csv")

    dataset_xlsx = client.get(f"/analysis/dataset/{dataset_id}/export?format=xlsx")
    assert dataset_xlsx.status_code == 200
    checks.append("dataset_xlsx")

    _cleanup_smoke_user(username)
    return "exported: " + ", ".join(checks)


def _cleanup_smoke_user(username: str) -> None:
    import sqlite3

    shutil.rmtree(ROOT / "user_data" / f"user_{username}", ignore_errors=True)
    db_path = ROOT / "users.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            cur.execute("DELETE FROM saved_presets WHERE user_id = ?", (row[0],))
            cur.execute("DELETE FROM users WHERE id = ?", (row[0],))
            conn.commit()
    finally:
        conn.close()


def main() -> int:
    run_check("directories", check_directories)
    run_check("dependencies", check_dependencies)
    run_check("model_registry", check_model_registry)
    run_check("preprocessing", check_preprocessing)
    run_check("training_all_task_types", check_training_all_task_types)
    run_check("model_save_load_predict_delete", check_model_save_load_predict_delete)
    run_check("hca_infer_rejected", check_hca_infer_rejected)
    run_check("export_via_routes", check_export_via_routes)

    passed = sum(1 for c in CHECKS if c["status"] == "ok")
    warnings = sum(1 for c in CHECKS if c["status"] == "warning")
    failed = sum(1 for c in CHECKS if c["status"] == "error")
    overall = "error" if failed else ("warning" if warnings else "ok")

    report = {
        "status": overall,
        "checks": CHECKS,
        "summary": {"passed": passed, "warnings": warnings, "failed": failed},
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
