from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional


STAGE_LABELS: Dict[str, str] = {
    "preparing_data": "Подготовка данных",
    "train_test_split": "Разделение train/test",
    "grid_search": "Подбор гиперпараметров (GridSearchCV)",
    "fit": "Обучение модели",
    "validation": "Валидация (holdout)",
    "kfold": "k-fold проверка",
    "bootstrap": "Bootstrap-интервалы",
    "permutation": "Permutation test",
    "saving": "Сохранение результатов",
    "done": "Готово",
    "error": "Ошибка",
    "interrupted": "Прервано",
}

# If a run's status hasn't been updated for this long and it never reached a
# terminal state, the server most likely restarted/crashed mid-computation -
# report it as interrupted rather than "running" forever.
STALE_AFTER_SECONDS = 90.0


class AnalysisRunManager:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    CSV_FIELDS = [
        "run_id",
        "created_at",
        "run_type",
        "dataset_id",
        "dataset_name",
        "dataset_version",
        "used_processed_data",
        "target_name",
        "target_type",
        "method",
        "model_type",
        "model_id",
        "status",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "r2",
        "mae",
        "rmse",
        "silhouette_score",
        "confusion_matrix",
        "confusion_matrix_total",
        "confusion_matrix_accuracy",
        "train_samples",
        "test_samples",
        "validation_mode",
        "class_distribution",
        "classes",
        "preprocessing_config",
        "preprocessing_preset",
        "spectrometer_parameters",
        "best_params",
        "hyperparameter_search_enabled",
        "training_time_seconds",
        "fit_time_sec",
        "validation_time_sec",
        "kfold_time_sec",
        "bootstrap_time_sec",
        "permutation_time_sec",
        "predict_time_one_sample_ms",
        "peak_memory_mb",
        "warnings",
    ]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        return value

    @staticmethod
    def _normalize_row_warnings(warnings: Any) -> List[str]:
        if warnings is None:
            return []
        if isinstance(warnings, str):
            return [warnings]
        if isinstance(warnings, list):
            return [str(w) for w in warnings if w is not None]
        return [str(warnings)]

    @staticmethod
    def _safe_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
            raise ValueError("Некорректный run_id")
        return value

    def _csv_row(self, run: Dict[str, Any], method_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = run.get("run") if isinstance(run.get("run"), dict) else run
        metrics: Dict[str, Any] = {}
        row: Dict[str, Any] = {field: None for field in self.CSV_FIELDS}
        row["run_id"] = base.get("run_id")
        row["created_at"] = base.get("created_at")
        row["run_type"] = base.get("run_type")
        row["dataset_id"] = base.get("dataset_id")
        row["dataset_name"] = base.get("dataset_name")
        # Determine dataset_version with inference logic
        dataset_version = base.get("dataset_version")
        if dataset_version is None:
            # Try to infer from summary.version
            summary = base.get("summary")
            if isinstance(summary, dict):
                dataset_version = summary.get("version")
        if dataset_version is None:
            # Infer from used_processed_data
            used_processed = base.get("used_processed_data")
            if used_processed is True:
                dataset_version = "processed"
            elif used_processed is False:
                dataset_version = "raw"
        row["dataset_version"] = dataset_version
        row["used_processed_data"] = base.get("used_processed_data")
        row["target_name"] = base.get("target_name")
        row["target_type"] = base.get("target_type")
        row["preprocessing_config"] = base.get("preprocessing_config") or base.get("preprocessing") or {}
        row["preprocessing_preset"] = base.get("preprocessing_preset") or (base.get("preprocessing_summary") or {}).get("preset") or ""
        row["spectrometer_parameters"] = base.get("spectrometer_parameters") or {}
        row["class_distribution"] = base.get("class_distribution") or {}

        if method_row is None:
            metrics = (run.get("results") or {}).get("metrics") or {}
            saved_model = (run.get("results") or {}).get("saved_model") or {}
            performance = (run.get("results") or {}).get("performance") or {}
            best_params = (run.get("results") or {}).get("best_params") or saved_model.get("best_params")
            hp_search = (run.get("results") or {}).get("hyperparameter_search") or {}
            row["method"] = base.get("best_method") or saved_model.get("model_type") or base.get("method") or ""
            row["model_type"] = saved_model.get("model_type") or base.get("best_method") or base.get("method") or ""
            row["model_id"] = saved_model.get("model_id") or ""
            row["status"] = base.get("status") or ""
            row["warnings"] = self._normalize_row_warnings(base.get("warnings"))
        else:
            metrics = method_row.get("metrics") if isinstance(method_row.get("metrics"), dict) else {}
            performance = method_row.get("performance") or {}
            best_params = method_row.get("best_params")
            hp_search = method_row.get("hyperparameter_search") or {}
            row["method"] = method_row.get("method") or method_row.get("model_type") or ""
            row["model_type"] = method_row.get("model_type") or method_row.get("method") or ""
            row["model_id"] = method_row.get("model_id") or ""
            row["status"] = method_row.get("status") or base.get("status") or ""
            row["warnings"] = self._normalize_row_warnings(method_row.get("warnings") or base.get("warnings"))
            if row["warnings"] == [] and base.get("warnings"):
                row["warnings"] = self._normalize_row_warnings(base.get("warnings"))

        if row["method"] is None:
            row["method"] = ""
        if row["model_type"] is None:
            row["model_type"] = ""

        row["best_params"] = self._json_safe(best_params) if best_params else None
        row["hyperparameter_search_enabled"] = bool(hp_search.get("enabled")) if isinstance(hp_search, dict) else False
        row["training_time_seconds"] = performance.get("total_time_sec")
        row["fit_time_sec"] = performance.get("fit_time_sec")
        row["validation_time_sec"] = performance.get("validation_time_sec")
        row["kfold_time_sec"] = performance.get("kfold_time_sec")
        row["bootstrap_time_sec"] = performance.get("bootstrap_time_sec")
        row["permutation_time_sec"] = performance.get("permutation_time_sec")
        row["predict_time_one_sample_ms"] = performance.get("predict_time_one_sample_ms")
        row["peak_memory_mb"] = performance.get("peak_memory_mb")

        row["accuracy"] = metrics.get("accuracy")
        row["precision_macro"] = metrics.get("precision_macro")
        row["recall_macro"] = metrics.get("recall_macro")
        row["f1_macro"] = metrics.get("f1_macro")
        row["r2"] = metrics.get("r2")
        row["mae"] = metrics.get("mae")
        row["rmse"] = metrics.get("rmse")
        row["silhouette_score"] = metrics.get("silhouette_score")
        row["confusion_matrix"] = metrics.get("confusion_matrix") if metrics.get("confusion_matrix") is not None else method_row.get("confusion_matrix") if method_row else None
        row["confusion_matrix_total"] = metrics.get("confusion_matrix_total")
        row["confusion_matrix_accuracy"] = metrics.get("confusion_matrix_accuracy")
        row["train_samples"] = metrics.get("train_samples")
        row["test_samples"] = metrics.get("test_samples")
        row["validation_mode"] = metrics.get("validation_mode")
        row["classes"] = metrics.get("classes")

        if row["confusion_matrix_total"] is None and isinstance(row["confusion_matrix"], (list, tuple)):
            try:
                matrix = row["confusion_matrix"]
                row["confusion_matrix_total"] = int(sum(sum(r) for r in matrix))
            except Exception:
                row["confusion_matrix_total"] = None
        if row["confusion_matrix_accuracy"] is None and isinstance(row["confusion_matrix"], (list, tuple)):
            try:
                matrix = row["confusion_matrix"]
                total = sum(sum(r) for r in matrix)
                if total:
                    row["confusion_matrix_accuracy"] = float(sum(matrix[i][i] for i in range(len(matrix))) / total)
                else:
                    row["confusion_matrix_accuracy"] = None
            except Exception:
                row["confusion_matrix_accuracy"] = None

        if row["used_processed_data"] is None:
            row["warnings"].append("used_processed_data отсутствует")
        if row["confusion_matrix_total"] is not None and row["test_samples"] is not None and int(row["confusion_matrix_total"]) != int(row["test_samples"]):
            row["warnings"].append("confusion_matrix_total не равен test_samples")
        if row["confusion_matrix_accuracy"] is not None and row["accuracy"] is not None:
            try:
                if abs(float(row["confusion_matrix_accuracy"]) - float(row["accuracy"])) > 1e-6:
                    row["warnings"].append("confusion_matrix_accuracy отличается от accuracy")
            except Exception:
                pass

        row["warnings"] = list(dict.fromkeys([str(w) for w in row["warnings"] if w]))
        return {field: self._json_safe(row.get(field)) for field in self.CSV_FIELDS}

    def _csv_rows_for_run(self, run: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        method_rows = (run.get("results") or {}).get("method_results") or (run.get("results") or {}).get("comparison") or []
        if run.get("run_type") == "comparison" and method_rows:
            for method_row in method_rows:
                rows.append(self._csv_row(run, method_row=method_row))
            return rows
        if method_rows and len(method_rows) == 1 and run.get("run_type") != "comparison":
            rows.append(self._csv_row(run, method_row=method_rows[0]))
            return rows
        rows.append(self._csv_row(run, method_row=None))
        return rows

    def _write_comparison_csv(self, path: Path, run: Dict[str, Any]) -> None:
        csv_bytes = self._csv_bytes_for_run(run)
        path.write_bytes(csv_bytes)

    def _csv_bytes_for_run(self, run: Dict[str, Any]) -> bytes:
        rows = self._csv_rows_for_run(run)
        if not rows:
            return b""
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8-sig")

    def csv_bytes(self, run_id: str) -> bytes:
        run = self.get(run_id)
        return self._csv_bytes_for_run(run)

    def csv_rows(self, run_id: str) -> List[Dict[str, Any]]:
        """Public accessor for the same flat rows used by csv_bytes(), for callers (e.g. the xlsx
        export route) that need them as Python dicts instead of a CSV byte string."""
        return self._csv_rows_for_run(self.get(run_id))

    def new_run_id(self, prefix: str = "run") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{stamp}"
        idx = 1
        run_id = f"{base}_{idx:03d}"
        while (self.root_dir / run_id).exists():
            idx += 1
            run_id = f"{base}_{idx:03d}"
        return run_id

    def _run_dir(self, run_id: str) -> Path:
        return self.root_dir / self._safe_id(run_id)

    def _status_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "status.json"

    def start_run(self, run_id: str, run_type: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Creates the run directory and writes the initial status.json, BEFORE any heavy
        computation starts. This file is separate from run_metadata.json (written only once,
        at the very end, by save()) so an interrupted background job can never corrupt or
        touch the final artifacts of this or any other run."""
        run_id = self._safe_id(run_id)
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "run_type": run_type,
            "status": "running",
            "stage": "preparing_data",
            "stage_label": STAGE_LABELS["preparing_data"],
            "progress": {},
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        if extra:
            payload.update(extra)
        self._status_path(run_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def update_run_status(self, run_id: str, **kwargs: Any) -> Dict[str, Any]:
        run_id = self._safe_id(run_id)
        path = self._status_path(run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id}
        except json.JSONDecodeError:
            payload = {"run_id": run_id}
        stage = kwargs.get("stage")
        if stage is not None and "stage_label" not in kwargs:
            kwargs["stage_label"] = STAGE_LABELS.get(stage, stage)
        payload.update(kwargs)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._run_dir(run_id).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        run_id = self._safe_id(run_id)
        path = self._status_path(run_id)
        meta_exists = (self._run_dir(run_id) / "run_metadata.json").exists()
        if not path.exists():
            if meta_exists:
                return {"run_id": run_id, "status": "success", "stage": "done", "stage_label": STAGE_LABELS["done"], "error": None}
            raise FileNotFoundError("Эксперимент не найден")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "running":
            if meta_exists:
                # save() finished but the in-progress status update raced behind it - treat as success.
                payload["status"] = "success"
                payload["stage"] = "done"
                payload["stage_label"] = STAGE_LABELS["done"]
            else:
                try:
                    updated_at = datetime.fromisoformat(payload.get("updated_at"))
                except (TypeError, ValueError):
                    updated_at = None
                if updated_at is not None:
                    age = (datetime.now(timezone.utc) - updated_at).total_seconds()
                    if age > STALE_AFTER_SECONDS:
                        payload["status"] = "error"
                        payload["stage"] = "interrupted"
                        payload["stage_label"] = STAGE_LABELS["interrupted"]
                        payload["error"] = "Расчёт прерван: сервер был перезапущен или процесс остановлен во время вычислений."
        return payload

    def save(self, run: Dict[str, Any]) -> Dict[str, Any]:
        run_id = self._safe_id(str(run.get("run_id") or self.new_run_id()))
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **run,
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "results.json").write_text(
            json.dumps(payload.get("results", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        plots = payload.get("plots")
        if plots is None:
            plots = payload.get("results", {}).get("plots") or []
        (run_dir / "plots.json").write_text(
            json.dumps({"plots": plots}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_comparison_csv(run_dir / "comparison_table.csv", payload)
        return payload

    @staticmethod
    def _list_method_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = payload.get("results") or {}
        rows = results.get("method_results") or results.get("comparison") or []
        if rows:
            return rows
        return [{"method": payload.get("best_method"), "metrics": results.get("metrics") or {}, "validation": results.get("validation") or {}, "hyperparameter_search": results.get("hyperparameter_search") or {}}]

    @classmethod
    def _list_summary_extra(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Old run_metadata.json files predate this method and simply won't have these keys -
        callers (the history table) must treat their absence as 'нет данных', not an error."""
        rows = cls._list_method_rows(payload)
        validation_ok = any((row.get("validation") or {}).get("status") == "ok" for row in rows)
        any_validation_attempted = any((row.get("validation") or {}) for row in rows)
        hp_enabled = any((row.get("hyperparameter_search") or {}).get("enabled") for row in rows)
        best_method = payload.get("best_method")
        best_row = next((row for row in rows if row.get("method") == best_method or row.get("model_type") == best_method), rows[0] if rows else {})
        metrics = best_row.get("metrics") or {}
        primary_metric_name = next((key for key in ("f1_macro", "accuracy", "r2", "rmse", "silhouette_score") if metrics.get(key) is not None), None)
        return {
            "validation_status": "ok" if validation_ok else ("skipped" if any_validation_attempted else None),
            "hyperparameter_search_enabled": hp_enabled,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": metrics.get(primary_metric_name) if primary_metric_name else None,
        }

    def list(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for meta_path in self.root_dir.glob("*/run_metadata.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            items.append(
                {
                    "run_id": payload.get("run_id"),
                    "run_type": payload.get("run_type"),
                    "created_at": payload.get("created_at"),
                    "dataset_id": payload.get("dataset_id"),
                    "dataset_name": payload.get("dataset_name"),
                    "target_name": payload.get("target_name"),
                    "target_type": payload.get("target_type"),
                    "methods": payload.get("methods"),
                    "best_method": payload.get("best_method"),
                    "status": payload.get("status"),
                    "summary": payload.get("summary"),
                    **self._list_summary_extra(payload),
                }
            )
        items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return items

    def get(self, run_id: str) -> Dict[str, Any]:
        meta_path = self._run_dir(run_id) / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("Эксперимент не найден")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        plots_path = self._run_dir(run_id) / "plots.json"
        if plots_path.exists():
            try:
                payload["plots"] = json.loads(plots_path.read_text(encoding="utf-8")).get("plots", [])
            except json.JSONDecodeError:
                payload["plots"] = []
        else:
            payload.setdefault("plots", [])
        return payload

    def delete(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError("Эксперимент не найден")
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        run_dir.rmdir()
        return {"run_id": run_id, "deleted": True}

    def export_zip(self, run_id: str) -> bytes:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError("Эксперимент не найден")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in run_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir).as_posix())
        buffer.seek(0)
        return buffer.read()
