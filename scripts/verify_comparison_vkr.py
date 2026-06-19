"""One-shot verification for the ВКР-readiness rework of compare_classification/compare_regression.

Builds two synthetic datasets shaped like the real ones the user referenced
(Ramam_ex1-like: 100x2000, classes health/heart disease; Voda_ex2-like: 135x1812,
numeric target with 9 levels x 15 replicates in [11, 99]) and runs them through the
real /analysis/model/train route (in-process, no server needed) with NO explicit
validation/hyperparameter_search override - to prove the new ВКР-mode defaults
(validation enabled, GridSearchCV enabled) actually kick in without the caller
having to ask for them.

Prints a checklist against the "must NOT contain" list from the rework spec and
writes the two full run JSON payloads to disk for manual review.

Usage:
    python scripts/verify_comparison_vkr.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _classification_csv(n_per_class: int = 50, n_features: int = 2000, seed: int = 42) -> str:
    rng = random.Random(seed)
    header = "sample_id,target," + ",".join(str(400 + i) for i in range(n_features)) + "\n"
    rows = [header]
    classes = ["health", "heart disease"]
    base_a = [rng.uniform(0.2, 1.0) for _ in range(n_features)]
    base_b = [v + rng.uniform(-0.3, 0.6) for v in base_a]
    idx = 0
    for cls, base in zip(classes, [base_a, base_b]):
        for _ in range(n_per_class):
            values = [round(v + rng.uniform(-0.05, 0.05), 6) for v in base]
            rows.append(f"s{idx},{cls}," + ",".join(map(str, values)) + "\n")
            idx += 1
    return "".join(rows)


def _regression_csv(levels=9, replicates=15, n_features: int = 1812, seed: int = 7) -> str:
    rng = random.Random(seed)
    header = "sample_id,target," + ",".join(str(400 + i) for i in range(n_features)) + "\n"
    rows = [header]
    concentrations = [11 + i * (99 - 11) / (levels - 1) for i in range(levels)]
    base = [rng.uniform(0.2, 1.0) for _ in range(n_features)]
    slope = [rng.uniform(0.001, 0.01) for _ in range(n_features)]
    idx = 0
    for conc in concentrations:
        for _ in range(replicates):
            values = [round(b + s * conc + rng.uniform(-0.02, 0.02), 6) for b, s in zip(base, slope)]
            rows.append(f"s{idx},{round(conc, 3)}," + ",".join(map(str, values)) + "\n")
            idx += 1
    return "".join(rows)


def _must_not(condition: bool, message: str, failures: list) -> None:
    if condition:
        failures.append(message)


def _check_comparison_run(run_json: dict, run_type: str) -> list:
    failures: list = []
    run = run_json["run"]
    rows = run["results"]["method_results"]
    model_ids = [row.get("model_id") for row in rows]

    _must_not(not rows, "нет ни одной строки method_results", failures)
    _must_not(len(model_ids) != len(set(model_ids)), f"дублирующиеся model_id: {model_ids}", failures)

    search_capable = {"plsda", "svm", "random_forest", "decision_tree"} if run_type == "compare_classification" else {"pls", "svr"}
    for row in rows:
        method = row.get("method")
        validation = row.get("validation") or {}
        training_policy = row.get("training_policy") or {}
        hp = row.get("hyperparameter_search") or {}
        metrics = row.get("metrics") or {}

        _must_not(validation.get("status") == "skipped", f"[{method}] validation.status == 'skipped' (ВКР-режим должен включать валидацию по умолчанию)", failures)
        _must_not(training_policy.get("metrics_source") == "not_available", f"[{method}] training_policy.metrics_source == 'not_available'", failures)
        if method in search_capable:
            _must_not(hp.get("enabled") is False, f"[{method}] hyperparameter_search.enabled == False в ВКР-режиме", failures)
            _must_not(row.get("best_params") is None, f"[{method}] best_params is null, хотя GridSearchCV должен был отработать", failures)
        _must_not(metrics.get("train_samples") is None, f"[{method}] metrics.train_samples is null", failures)
        _must_not(metrics.get("test_samples") is None, f"[{method}] metrics.test_samples is null", failures)

        if run_type == "compare_classification":
            test_cm = (row.get("test_outputs") or {}).get("confusion_matrix")
            _must_not(metrics.get("confusion_matrix") != test_cm, f"[{method}] metrics.confusion_matrix != test_outputs.confusion_matrix", failures)
            warnings_text = " ".join(metrics.get("warnings") or [])
            _must_not(
                "unavailable" in warnings_text.lower() and metrics.get("confusion_matrix") is not None,
                f"[{method}] противоречивое предупреждение про confusion_matrix при наличии метрики",
                failures,
            )
            kfold = validation.get("kfold") or {}
            _must_not(kfold.get("status") != "ok", f"[{method}] kfold.status != 'ok'", failures)
            bootstrap = validation.get("bootstrap") or validation.get("bootstrap_ci") or {}
            _must_not(bootstrap.get("status") != "ok", f"[{method}] bootstrap.status != 'ok'", failures)
            if method == "plsda":
                permutation = validation.get("permutation_test") or {}
                _must_not(permutation.get("status") != "ok", "[plsda] permutation_test обязателен и должен быть status=ok", failures)
        else:
            kfold = validation.get("kfold") or {}
            _must_not(kfold.get("status") != "ok", f"[{method}] kfold.status != 'ok'", failures)
            if method == "pls":
                permutation = validation.get("permutation_test") or {}
                _must_not(permutation.get("status") != "ok", "[pls] permutation_test обязателен и должен быть status=ok", failures)

    _must_not(not run.get("best_method_selection", {}).get("selected_method"), "best_method_selection.selected_method отсутствует", failures)
    _must_not(not run.get("best_method_selection", {}).get("tie_breakers"), "best_method_selection.tie_breakers отсутствует", failures)
    vkr = run.get("vkr_tables") or {}
    table_key = "classification_comparison" if run_type == "compare_classification" else "regression_comparison"
    _must_not(not vkr.get(table_key), f"vkr_tables.{table_key} пуст", failures)
    _must_not(not vkr.get("performance_summary"), "vkr_tables.performance_summary пуст", failures)
    _must_not(not run.get("pipeline_performance"), "pipeline_performance отсутствует", failures)
    _must_not(not run.get("comparison_performance"), "comparison_performance отсутствует", failures)
    return failures


def main() -> int:
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    username = f"vkrcheck_{os.getpid()}_{int(time.time() * 1000)}"
    password = "vkr-check-password"
    assert client.post("/register", data={"username": username, "password": password, "confirm_password": password}, follow_redirects=False).status_code == 303
    assert client.post("/login", data={"username": username, "password": password}, follow_redirects=False).status_code == 303

    overall_ok = True
    deliverables = []

    print("=== compare_classification (Ramam_ex1-like, 100x2000, health/heart disease) ===")
    clf_csv = _classification_csv()
    imp = client.post("/analysis/dataset/import-standard", files={"file": ("ramam_like.csv", clf_csv.encode("utf-8"), "text/csv")})
    assert imp.status_code == 200, imp.text
    dataset_id = imp.json()["dataset_id"]
    t0 = time.perf_counter()
    train = client.post(
        "/analysis/model/train",
        json={"dataset_id": dataset_id, "model_type": "compare_classification", "model_name": "vkr_check"},
    )
    print(f"  trained in {time.perf_counter() - t0:.1f}s, status={train.status_code}")
    assert train.status_code == 200, train.text
    run_id = train.json()["run_id"]
    run_json = client.get(f"/analysis/runs/{run_id}").json()
    out_path = ROOT / "verification_compare_classification.json"
    out_path.write_text(json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
    deliverables.append(str(out_path))
    failures = _check_comparison_run(run_json, "compare_classification")
    if failures:
        overall_ok = False
        print(f"  FAIL ({len(failures)} problems):")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  OK: ни одной проблемы из чек-листа не найдено")
    print(f"  best_method = {run_json['run'].get('best_method')}")
    print(f"  best_method_selection.reason = {run_json['run'].get('best_method_selection', {}).get('reason')}")

    print("\n=== compare_regression (Voda_ex2-like, 135x1812, conc (Y) 11..99, 9x15) ===")
    reg_csv = _regression_csv()
    imp = client.post("/analysis/dataset/import-standard", files={"file": ("voda_like.csv", reg_csv.encode("utf-8"), "text/csv")})
    assert imp.status_code == 200, imp.text
    dataset_id = imp.json()["dataset_id"]
    t0 = time.perf_counter()
    train = client.post(
        "/analysis/model/train",
        json={"dataset_id": dataset_id, "model_type": "compare_regression", "model_name": "vkr_check"},
    )
    print(f"  trained in {time.perf_counter() - t0:.1f}s, status={train.status_code}")
    assert train.status_code == 200, train.text
    run_id = train.json()["run_id"]
    run_json = client.get(f"/analysis/runs/{run_id}").json()
    out_path = ROOT / "verification_compare_regression.json"
    out_path.write_text(json.dumps(run_json, ensure_ascii=False, indent=2), encoding="utf-8")
    deliverables.append(str(out_path))
    failures = _check_comparison_run(run_json, "compare_regression")
    if failures:
        overall_ok = False
        print(f"  FAIL ({len(failures)} problems):")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  OK: ни одной проблемы из чек-листа не найдено")
    print(f"  best_method = {run_json['run'].get('best_method')}")
    print(f"  best_method_selection.reason = {run_json['run'].get('best_method_selection', {}).get('reason')}")

    import sqlite3

    shutil.rmtree(ROOT / "user_data" / f"user_{username}", ignore_errors=True)
    db_path = ROOT / "users.db"
    if db_path.exists():
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

    print("\nDeliverables written:")
    for path in deliverables:
        print(f"  - {path}")
    print("\nOVERALL:", "OK" if overall_ok else "FAILURES FOUND")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
