from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from services.analysis.model_manager import ModelManager
from services.analysis.models import create_model
from services.analysis.dataset_importer import SpectralDataset as ImportedSpectralDataset
from services.analysis.dataset_importer import preprocess_matrix
from services.analysis.spectrum_loader import SpectrumDataset, parse_spectrum_file, parse_target_values
from services.analysis.visualization import class_count_plot, pca_plot, pls_regression_plot, residual_plot
from services.analysis.hyperparameter_search import run_hyperparameter_search

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def _timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class _MemoryProbe:
    """Measures peak memory for one operation; prefers psutil RSS, falls back to tracemalloc."""

    def __init__(self) -> None:
        self._process = psutil.Process() if psutil is not None else None
        self._start_rss = self._process.memory_info().rss if self._process is not None else None
        self._tracemalloc_started_here = False
        if self._process is None and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started_here = True

    def peak_mb(self) -> float:
        if self._process is not None:
            current_rss = self._process.memory_info().rss
            delta = max(0, current_rss - (self._start_rss or 0))
            return round(delta / (1024 * 1024), 3)
        _current, peak = tracemalloc.get_traced_memory()
        return round(peak / (1024 * 1024), 3)

    def close(self) -> None:
        if self._tracemalloc_started_here:
            tracemalloc.stop()


class AnalysisService:
    def __init__(self, model_root: Path):
        self.model_manager = ModelManager(model_root)

    @staticmethod
    def parse_file(raw: bytes, filename: str) -> SpectrumDataset:
        return parse_spectrum_file(raw, filename)

    @staticmethod
    def _parse_supervised_targets(model_type: str, raw_targets: Optional[str], sample_count: int) -> Optional[np.ndarray]:
        normalized = model_type.lower().replace("-", "_")
        if normalized in {
            "pca",
            "pls",
            "pls_regression",
            "plsda",
            "pls_da",
            "svm",
            "svc",
            "random_forest",
            "rf",
            "decision_tree",
            "dt",
            "svr",
            "kmeans",
            "k_means",
            "hca",
            "hierarchical",
        } and raw_targets:
            return parse_target_values(raw_targets, sample_count)
        return None

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        normalized = model_type.strip().lower().replace("-", "_")
        aliases = {
            "pls_regression": "pls",
            "pls_da": "plsda",
            "svc": "svm",
            "rf": "random_forest",
            "dt": "decision_tree",
            "k_means": "kmeans",
            "hierarchical": "hca",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _task_type(model_type: str) -> str:
        normalized = AnalysisService._normalize_model_type(model_type)
        if normalized == "pca":
            return "visualization"
        if normalized in {"plsda", "svm", "random_forest", "decision_tree"}:
            return "classification"
        if normalized in {"pls", "svr"}:
            return "regression"
        if normalized in {"kmeans", "hca"}:
            return "clustering"
        return "analysis"

    @staticmethod
    def _target_kind(y: Optional[np.ndarray]) -> str:
        if y is None:
            return "none"
        try:
            numeric = np.asarray(y, dtype=float)
            unique = len(np.unique(numeric))
            return "numeric" if unique > max(10, int(len(numeric) * 0.1)) else "categorical"
        except ValueError:
            return "categorical"

    def train_and_save(
        self,
        model_type: str,
        dataset: SpectrumDataset,
        raw_targets: Optional[str],
        n_components: Optional[int],
        model_name: Optional[str],
        do_validation: bool = True,
        test_size: float = 0.3,
        random_state: int = 42,
        model_params: Optional[Dict[str, Any]] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
        hyperparameter_search: bool = False,
        refit_on_full_data: bool = True,
        model_id_override: Optional[str] = None,
        kfold_enabled: bool = True,
        bootstrap_enabled: bool = True,
        permutation_enabled: Optional[bool] = None,
        n_splits: int = 5,
        n_bootstrap: int = 1000,
        n_permutations: int = 200,
        quick_mode: bool = False,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        def _stage(name: str) -> None:
            if on_stage is not None:
                on_stage(name)

        memory_probe = _MemoryProbe()
        total_start = time.perf_counter()
        try:
            _stage("preparing_data")
            x = dataset.x
            y = self._parse_supervised_targets(model_type, raw_targets, dataset.sample_count)
            self._validate_method_inputs(model_type=model_type, x=x, y=y)

            normalized_model_type = self._normalize_model_type(model_type)
            params = self._default_model_params(
                model_type=normalized_model_type,
                x=x,
                y=y,
                n_components=n_components,
                random_state=random_state,
                provided=model_params,
            )
            validation, eval_model, tuned_params, validation_time_sec = self._run_validation(
                model_type=normalized_model_type,
                x=x,
                y=y,
                n_components=params.get("n_components"),
                model_params=params,
                do_validation=do_validation,
                test_size=test_size,
                random_state=random_state,
                hyperparameter_search=hyperparameter_search,
                kfold_enabled=kfold_enabled,
                bootstrap_enabled=bootstrap_enabled,
                permutation_enabled=permutation_enabled,
                n_splits=n_splits,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                quick_mode=quick_mode,
                on_stage=_stage,
            )

            _stage("fit")
            fit_start = time.perf_counter()
            if refit_on_full_data or eval_model is None:
                model = create_model(model_type=model_type, n_components=tuned_params.get("n_components"), model_params=tuned_params)
                model.fit(x, y)
                training_policy = {
                    "validation_model": "trained_on_train_split" if eval_model is not None else "not_available",
                    "saved_model": "refit_on_full_dataset",
                    "refit_on_full_data": True,
                    "metrics_source": "holdout_test_split" if eval_model is not None else "not_available",
                }
            else:
                model = eval_model
                training_policy = {
                    "validation_model": "trained_on_train_split",
                    "saved_model": "trained_on_train_split",
                    "refit_on_full_data": False,
                    "metrics_source": "holdout_test_split",
                }
            fit_time_sec = time.perf_counter() - fit_start

            result = model.training_result(x, y)
            if y is not None:
                result["target"] = [str(value) for value in np.asarray(y).tolist()]
            if dataset.sample_names:
                result["sample_ids"] = dataset.sample_names
            task_type = self._task_type(model.model_type)
            metrics = self._extract_metrics(model.model_type, result, validation)
            outputs_bundle = self._build_outputs(task_type, result, metrics, validation)
            outputs = outputs_bundle["outputs"]
            test_outputs = outputs_bundle["test_outputs"]
            full_dataset_outputs = outputs_bundle["full_dataset_outputs"]
            warnings = []
            if isinstance(validation, dict):
                warnings.extend(validation.get("warnings", []) or [])
            warnings.extend(metrics.get("warnings", []) or [])
            warnings.extend(result.get("warnings", []) or [])

            predict_start = time.perf_counter()
            try:
                model.predict(x[:1])
            except Exception:
                pass
            predict_time_one_sample_ms = (time.perf_counter() - predict_start) * 1000.0

            timestamp_name = _timestamp_name()
            display_name = (model_name or "").strip() or f"{model.model_type}_{timestamp_name}"
            axis = np.asarray(dataset.spectral_axis if dataset.spectral_axis is not None else np.arange(dataset.feature_count), dtype=float)
            saved_params = model.get_params()
            saved_classes = saved_params.get("classes") or result.get("classes", [])

            hyperparameter_search_report = validation.get("hyperparameter_search") if isinstance(validation, dict) else None
            if not isinstance(hyperparameter_search_report, dict):
                hyperparameter_search_report = {"enabled": bool(hyperparameter_search), "status": "skipped", "reason": "Validation disabled or unavailable", "best_params_source": "manual_or_default"}

            performance = {
                "import_time_sec": dataset.metadata.get("import_time_sec") if isinstance(dataset.metadata, dict) else None,
                "preprocessing_time_sec": dataset.metadata.get("preprocessing_time_sec") if isinstance(dataset.metadata, dict) else None,
                "hyperparameter_search_time_sec": hyperparameter_search_report.get("search_time_sec"),
                "fit_time_sec": round(fit_time_sec, 6),
                "validation_time_sec": round(validation_time_sec, 6) if validation_time_sec is not None else None,
                "kfold_time_sec": (validation.get("kfold") or {}).get("time_sec") if isinstance(validation, dict) else None,
                "bootstrap_time_sec": (validation.get("bootstrap") or {}).get("time_sec") if isinstance(validation, dict) else None,
                "permutation_time_sec": (validation.get("permutation_test") or {}).get("time_sec") if isinstance(validation, dict) else None,
                "predict_time_one_sample_ms": round(predict_time_one_sample_ms, 4),
                "export_time_sec": None,
                "total_time_sec": round(time.perf_counter() - total_start, 6),
                "peak_memory_mb": memory_probe.peak_mb(),
            }

            metadata = {
                "model_name": display_name,
                "display_name": display_name,
                "task_type": task_type,
                "input_shape": [int(dataset.sample_count), int(dataset.feature_count)],
                "feature_count": dataset.feature_count,
                "expected_feature_count": dataset.feature_count,
                "sample_count": dataset.sample_count,
                "source_format": dataset.source_format,
                "file_format": dataset.file_format,
                "preprocessing": {},
                "classes": saved_classes,
                "class_labels": saved_classes,
                "class_mapping": {str(idx): str(label) for idx, label in enumerate(saved_classes)},
                "target_column": None,
                "spectral_axis": axis.astype(float).tolist(),
                "axis": axis.astype(float).tolist(),
                "axis_min": float(np.min(axis)) if axis.size else None,
                "axis_max": float(np.max(axis)) if axis.size else None,
                "axis_range": [float(np.min(axis)), float(np.max(axis))] if axis.size else [None, None],
                "feature_order": [f"{float(value):.12g}" for value in axis],
                "params": saved_params,
                "model_params": tuned_params,
                "best_params": hyperparameter_search_report.get("best_params") if hyperparameter_search_report.get("status") == "ok" else None,
                "target_type": self._target_kind(y),
                "validation_config": {
                    "enabled": bool(do_validation),
                    "test_size": float(max(0.1, min(0.5, test_size))),
                    "random_state": int(random_state),
                },
                "validation": validation,
                "metrics": metrics,
                "outputs": outputs,
                "test_outputs": test_outputs,
                "full_dataset_outputs": full_dataset_outputs,
                "training_policy": training_policy,
                "training_policy_note": "metrics.confusion_matrix and metrics.* reflect test_outputs (holdout) when validation.status == 'ok'; full_dataset_outputs/outputs are diagnostic-only.",
                "hyperparameter_search": hyperparameter_search_report,
                "reproducibility": {
                    "random_state": int(random_state),
                    "user_overridden_random_state": int(random_state) != 42,
                    "numpy_random_seed": int(random_state),
                },
                "performance": performance,
            }
            if normalized_model_type == "hca":
                metadata["can_infer_new_samples"] = False
                metadata.setdefault("limitations", [])
                metadata["limitations"].append(
                    "HCA does not support direct inference for new spectra because AgglomerativeClustering has no predict method."
                )
            if metadata_extra:
                metadata.update(metadata_extra)
            metadata["feature_count"] = int(metadata.get("feature_count") or dataset.feature_count)
            metadata["expected_feature_count"] = int(metadata.get("expected_feature_count") or metadata["feature_count"])
            if not metadata.get("spectral_axis"):
                metadata["spectral_axis"] = axis.astype(float).tolist()
            if not metadata.get("axis"):
                metadata["axis"] = axis.astype(float).tolist()
            metadata["target_column"] = metadata.get("target_name") or metadata.get("target_column")
            metadata["preprocessing_applied"] = bool(metadata.get("used_processed_data"))

            _stage("saving")
            saved = self.model_manager.save(model_type=model.model_type, model_obj=model, metadata=metadata, model_id=model_id_override)
            plot_source = result
            if task_type == "classification":
                validation_classes = validation.get("classes") if isinstance(validation, dict) else []
                plot_source = {
                    "confusion_matrix": metrics.get("confusion_matrix"),
                    "classes": metrics.get("classes") or validation_classes,
                    "y_pred": metrics.get("y_pred") or [],
                    "y_true": metrics.get("y_true") or [],
                    "sample_ids": [],
                }
            plots = self._build_plots(model.model_type, plot_source, phase="training", dataset=dataset)
            plot = plots[0] if plots else {}
            prediction_rows = self._training_prediction_rows(result, dataset.sample_names)

            return {
                "model_type": model.model_type,
                "model_name": metadata["model_name"],
                "task_type": task_type,
                "saved_model": saved,
                "result": result,
                "summary": {
                    "n_samples": int(dataset.sample_count),
                    "n_features": int(dataset.feature_count),
                    "target_type": metadata["target_type"],
                    "task_type": task_type,
                },
                "metrics": metrics,
                "outputs": outputs,
                "test_outputs": test_outputs,
                "full_dataset_outputs": full_dataset_outputs,
                "legacy_predictions_field": True,
                "predictions": prediction_rows or result.get("y_pred") or result.get("cluster_labels") or result.get("scores") or result.get("y_pred"),
                "validation": validation,
                "training_policy": training_policy,
                "hyperparameter_search": hyperparameter_search_report,
                "reproducibility": metadata["reproducibility"],
                "performance": performance,
                "best_params": metadata.get("best_params"),
                "plot": plot,
                "plots": plots,
                "model_metadata": saved,
                "warnings": warnings,
            }
        finally:
            memory_probe.close()

    def _default_model_params(
        self,
        model_type: str,
        x: np.ndarray,
        y: Optional[np.ndarray],
        n_components: Optional[int],
        random_state: int,
        provided: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params = dict(provided or {})
        n_samples, n_features = x.shape
        class_count = len(np.unique(np.asarray(y, dtype=str))) if y is not None else 0
        imbalance = False
        if y is not None and class_count >= 2:
            _, counts = np.unique(np.asarray(y, dtype=str), return_counts=True)
            imbalance = bool(np.min(counts) / np.max(counts) < 0.3)

        if model_type == "pca":
            params.setdefault("n_components", min(2, n_samples, n_features))
        elif model_type == "plsda":
            params.setdefault("n_components", min(2, max(1, class_count), max(1, n_samples - 1), n_features))
        elif model_type == "pls":
            params.setdefault("n_components", min(5, max(1, n_samples - 1), n_features))
        elif model_type == "svm":
            params.setdefault("kernel", "rbf")
            params.setdefault("C", 1.0)
            params.setdefault("gamma", "scale")
            params.setdefault("class_weight", "balanced" if imbalance else None)
        elif model_type == "random_forest":
            params.setdefault("n_estimators", 200)
            params.setdefault("min_samples_split", 2)
            params.setdefault("class_weight", "balanced" if imbalance else None)
        elif model_type == "decision_tree":
            params.setdefault("max_depth", None)
            params.setdefault("min_samples_split", 2)
            params.setdefault("class_weight", "balanced" if imbalance else None)
        elif model_type in {"kmeans", "hca"}:
            params.setdefault("n_clusters", class_count if class_count >= 2 else 2)
            if model_type == "hca":
                params.setdefault("linkage", "ward")
        elif model_type == "svr":
            params.setdefault("kernel", "rbf")
            params.setdefault("C", 1.0)
            params.setdefault("epsilon", 0.1)
            params.setdefault("gamma", "scale")
        params.setdefault("random_state", int(random_state))
        if n_components is not None:
            params["n_components"] = int(n_components)
        return params

    def _validate_method_inputs(self, model_type: str, x: np.ndarray, y: Optional[np.ndarray]) -> None:
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
            raise ValueError("X должен быть матрицей samples x spectral_features минимум с двумя признаками.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Нельзя обучить модель: X содержит NaN или бесконечные значения.")
        normalized = self._normalize_model_type(model_type)
        if normalized in {"plsda", "svm", "random_forest", "decision_tree"}:
            if y is None:
                raise ValueError("Для выбранного метода требуется target, но он не найден.")
            labels = np.asarray(y, dtype=str)
            if len(np.unique(labels)) < 2:
                raise ValueError("Для классификации target должен содержать минимум два класса.")
        if normalized in {"pls", "svr"}:
            if y is None:
                raise ValueError("Для регрессии требуется target, но он не найден.")
            try:
                np.asarray(y, dtype=float)
            except ValueError as exc:
                raise ValueError("Для регрессии target должен быть числовым.") from exc
        if normalized not in {"pca", "plsda", "pls", "svm", "random_forest", "decision_tree", "kmeans", "hca", "svr"}:
            raise ValueError(f"Метод {model_type} не поддерживается.")

    def _extract_metrics(self, model_type: str, result: Dict[str, Any], validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = self._normalize_model_type(model_type)
        if isinstance(validation, dict) and validation.get("status") == "ok" and isinstance(validation.get("metrics"), dict):
            metrics = dict(validation["metrics"])
            if normalized in {"plsda", "svm", "random_forest", "decision_tree"}:
                cm = validation.get("confusion_matrix")
                cm_accuracy = self._confusion_matrix_accuracy(cm)
                metrics.update(
                    {
                        "confusion_matrix": cm,
                        "classes": validation.get("classes"),
                        "y_true": validation.get("y_true"),
                        "y_pred": validation.get("y_pred"),
                        "train_samples": validation.get("train_samples"),
                        "test_samples": validation.get("test_samples"),
                        "validation_mode": validation.get("mode"),
                        "confusion_matrix_total": self._confusion_matrix_total(cm),
                        "confusion_matrix_accuracy": cm_accuracy,
                    }
                )
                warnings = list(validation.get("warnings") or [])
                if cm is None:
                    warnings.append("Validation confusion_matrix is unavailable; no full-dataset confusion_matrix fallback was used.")
                elif metrics.get("accuracy") is not None and cm_accuracy is not None and abs(float(metrics["accuracy"]) - cm_accuracy) > 1e-6:
                    warnings.append("Validation confusion_matrix_accuracy differs from accuracy.")
                if warnings:
                    metrics["warnings"] = warnings
            else:
                metrics.update(
                    {
                        "train_samples": validation.get("train_samples"),
                        "test_samples": validation.get("test_samples"),
                        "validation_mode": validation.get("mode"),
                    }
                )
            return metrics
        if normalized == "pca":
            ratios = [float(v) for v in result.get("explained_variance_ratio", [])]
            return {"explained_variance_ratio": ratios, "explained_variance_total": float(sum(ratios))}
        if normalized in {"plsda", "svm", "random_forest", "decision_tree"}:
            metrics = {key: result.get(key) for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"] if key in result}
            metrics.update(
                {
                    "confusion_matrix": None,
                    "classes": result.get("classes"),
                    "y_true": None,
                    "y_pred": None,
                    "train_samples": None,
                    "test_samples": None,
                    "validation_mode": validation.get("status") if isinstance(validation, dict) else "unavailable",
                    "confusion_matrix_total": None,
                    "confusion_matrix_accuracy": None,
                    "warnings": [
                        "Validation did not run, so metrics.confusion_matrix is intentionally null (no honest holdout estimate exists). "
                        "A full-dataset diagnostic confusion matrix is still available in full_dataset_outputs.confusion_matrix / "
                        "outputs.confusion_matrix, but it is NOT a substitute for a held-out test metric."
                    ],
                }
            )
            return metrics
        if normalized in {"pls", "svr"}:
            return {key: result.get(key) for key in ["r2", "mae", "rmse"] if key in result}
        if normalized in {"kmeans", "hca"}:
            return {
                key: result.get(key)
                for key in [
                    "n_clusters",
                    "inertia",
                    "silhouette_score",
                    "cluster_distribution",
                    "adjusted_rand_score",
                    "normalized_mutual_info_score",
                ]
                if key in result
            }
        return {}

    @staticmethod
    def _full_dataset_outputs(task_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnostic predictions for ALL input samples (training data, possibly seen by the model).

        These are NOT held-out test metrics: for supervised methods the model that produced them may
        have been refit on the very same samples it is predicting here (see training_policy). Use
        test_outputs for an honest, non-overfit estimate of quality.
        """
        note = (
            "Predictions for all input samples (diagnostic view). The model may have been trained on "
            "some or all of these same samples (see training_policy) - NOT a held-out test estimate."
        )
        if task_type == "classification":
            return {
                "type": "classification",
                "note": note,
                "y_true": result.get("y_true") or [],
                "y_pred": result.get("y_pred") or [],
                "class_labels": result.get("classes") or [],
                "confusion_matrix": result.get("confusion_matrix") or [],
            }
        if task_type == "regression":
            y_true = result.get("y_true") or []
            y_pred = result.get("y_pred") or []
            residuals = []
            if y_true and y_pred and len(y_true) == len(y_pred):
                try:
                    residuals = [float(t) - float(p) for t, p in zip(y_true, y_pred)]
                except (TypeError, ValueError):
                    residuals = []
            return {"type": "regression", "note": note, "y_true": y_true, "y_pred": y_pred, "residuals": residuals}
        if task_type == "clustering":
            return {
                "type": "clustering",
                "note": "Clustering has no train/test split; cluster assignment covers the full dataset by definition.",
                "cluster_labels": result.get("cluster_labels") or [],
                "cluster_distribution": result.get("cluster_distribution") or {},
            }
        if task_type == "visualization":
            return {
                "type": "pca",
                "note": "PCA is unsupervised and has no train/test split; scores cover the full dataset.",
                "scores": result.get("scores") or [],
                "explained_variance_ratio": result.get("explained_variance_ratio") or [],
            }
        return {"type": task_type, "note": note}

    @staticmethod
    def _test_outputs(task_type: str, validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predictions for ONLY the held-out test split produced by holdout validation, when available."""
        if not isinstance(validation, dict) or validation.get("status") != "ok":
            reason = validation.get("reason") if isinstance(validation, dict) else "Validation did not run"
            return {"type": task_type, "status": "unavailable", "reason": reason or "Validation did not run", "note": "No holdout test split was evaluated for this run."}
        if task_type == "classification":
            return {
                "type": "classification",
                "status": "ok",
                "note": "Predictions for the held-out test split only (honest, non-overfit estimate).",
                "y_true": validation.get("y_true") or [],
                "y_pred": validation.get("y_pred") or [],
                "class_labels": validation.get("classes") or [],
                "confusion_matrix": validation.get("confusion_matrix") or [],
                "train_samples": validation.get("train_samples"),
                "test_samples": validation.get("test_samples"),
            }
        if task_type == "regression":
            y_true = validation.get("y_true") or []
            y_pred = validation.get("y_pred") or []
            residuals = []
            if y_true and y_pred and len(y_true) == len(y_pred):
                try:
                    residuals = [float(t) - float(p) for t, p in zip(y_true, y_pred)]
                except (TypeError, ValueError):
                    residuals = []
            return {
                "type": "regression",
                "status": "ok",
                "note": "Predictions for the held-out test split only (honest, non-overfit estimate).",
                "y_true": y_true,
                "y_pred": y_pred,
                "residuals": residuals,
                "train_samples": validation.get("train_samples"),
                "test_samples": validation.get("test_samples"),
            }
        return {"type": task_type, "status": "unavailable", "reason": f"{task_type} has no holdout test split"}

    @classmethod
    def _build_outputs(
        cls,
        task_type: str,
        result: Dict[str, Any],
        metrics: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds the test_outputs/full_dataset_outputs split plus a legacy 'outputs' field.

        legacy 'outputs' always mirrors full_dataset_outputs (its historical meaning, before the
        test/full split existed) and is tagged legacy_outputs_field/outputs_source so old and new
        consumers can both rely on it without ambiguity about which population it covers.
        """
        full_dataset_outputs = cls._full_dataset_outputs(task_type, result)
        test_outputs = cls._test_outputs(task_type, validation)
        legacy_outputs = dict(full_dataset_outputs)
        legacy_outputs["legacy_outputs_field"] = True
        legacy_outputs["outputs_source"] = "full_dataset_diagnostic"
        return {"outputs": legacy_outputs, "test_outputs": test_outputs, "full_dataset_outputs": full_dataset_outputs}

    @staticmethod
    def _confusion_matrix_total(matrix: Any) -> Optional[int]:
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=float)
        if arr.size == 0:
            return 0
        return int(np.sum(arr))

    @staticmethod
    def _confusion_matrix_accuracy(matrix: Any) -> Optional[float]:
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            return None
        total = float(np.sum(arr))
        if total <= 0:
            return None
        return float(np.trace(arr) / total)

    def _run_validation(
        self,
        model_type: str,
        x: np.ndarray,
        y: Optional[np.ndarray],
        n_components: Optional[int],
        model_params: Optional[Dict[str, Any]],
        do_validation: bool,
        test_size: float,
        random_state: int,
        hyperparameter_search: bool = False,
        kfold_enabled: bool = True,
        bootstrap_enabled: bool = True,
        permutation_enabled: Optional[bool] = None,
        n_splits: int = 5,
        n_bootstrap: int = 1000,
        n_permutations: int = 200,
        quick_mode: bool = False,
        on_stage: Optional[Callable[[str], None]] = None,
    ):
        """Returns (validation_dict, eval_model_or_None, tuned_params, validation_time_sec_or_None).

        kfold_enabled/bootstrap_enabled toggle those checks on/off (user-controllable from the UI).
        permutation_enabled is tri-state: None means "auto" (only runs for plsda/pls, as before),
        True/False is an explicit user override - but the test itself only HAS an implementation for
        plsda/pls, so True for any other model_type still reports status=skipped with that reason.

        eval_model is the model fitted ONLY on the train split (used for holdout metrics).
        tuned_params is model_params with hyperparameter-search best_params merged in, when search ran;
        otherwise it is model_params unchanged. The caller decides whether the FINAL saved model is
        eval_model itself (refit_on_full_data=False) or a fresh fit on the full dataset with tuned_params
        (refit_on_full_data=True).
        """
        def _stage(name: str) -> None:
            if on_stage is not None:
                on_stage(name)
        base_params = dict(model_params or {})
        no_search_report = {"enabled": bool(hyperparameter_search), "status": "skipped", "reason": "Validation disabled or unavailable", "best_params_source": "manual_or_default"}

        if not do_validation:
            return {"status": "skipped", "reason": "Validation disabled by user", "hyperparameter_search": no_search_report}, None, base_params, None

        if model_type in {"pca", "kmeans", "hca"}:
            return {
                "status": "skipped",
                "reason": f"{model_type.upper()} does not use supervised holdout metrics",
                "limitations": [
                    "Выбранный метод не является supervised-моделью для прогноза класса/концентрации.",
                    "Для надежной количественной валидации используйте PLS (регрессия) или PLS-DA (классификация).",
                ],
                "hyperparameter_search": no_search_report,
            }, None, base_params, None

        if y is None:
            return {
                "status": "skipped",
                "reason": "Target values are required for supervised validation",
                "limitations": [
                    "Для supervised-валидации необходимы истинные target-значения.",
                    "Без target можно выполнить только разведочный анализ.",
                ],
                "hyperparameter_search": no_search_report,
            }, None, base_params, None

        if len(x) < 6:
            return {
                "status": "skipped",
                "reason": "Need at least 6 spectra for stable train/test validation",
                "limitations": [
                    "Слишком мало спектров для устойчивой оценки качества модели.",
                    "Рекомендуется расширить выборку минимум до 20-30 образцов.",
                ],
                "hyperparameter_search": no_search_report,
            }, None, base_params, None

        validation_start = time.perf_counter()
        safe_test_size = float(max(0.1, min(0.5, test_size)))
        stratify = None
        warnings: List[str] = []

        if model_type in {"plsda", "svm", "random_forest", "decision_tree"}:
            y_labels = np.asarray(y, dtype=str)
            unique, counts = np.unique(y_labels, return_counts=True)
            if len(unique) < 2:
                return {
                    "status": "skipped",
                    "reason": "Need at least 2 classes for classification validation",
                    "limitations": [
                        "В датасете только один класс, классификационная задача не определена.",
                    ],
                    "hyperparameter_search": no_search_report,
                }, None, base_params, None
            if np.min(counts) >= 2:
                stratify = y_labels
            else:
                warnings.append("Not enough samples in at least one class for stratified split")

        _stage("train_test_split")
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=safe_test_size,
            random_state=int(random_state),
            stratify=stratify,
        )

        search_report = {"enabled": bool(hyperparameter_search), "status": "skipped", "reason": "hyperparameter_search disabled by user", "best_params_source": "manual_or_default"}
        tuned_params = dict(base_params)
        if hyperparameter_search:
            _stage("grid_search")
            search_start = time.perf_counter()
            search_report = run_hyperparameter_search(
                model_type=model_type,
                x_train=x_train,
                y_train=y_train,
                base_params=base_params,
                random_state=random_state,
                quick_mode=quick_mode,
            )
            search_report["search_time_sec"] = round(time.perf_counter() - search_start, 6)
            if search_report.get("status") == "ok":
                tuned_params.update(search_report.get("best_params") or {})

        _stage("fit")
        eval_model = create_model(model_type=model_type, n_components=tuned_params.get("n_components", n_components), model_params=tuned_params)
        eval_model.fit(x_train, y_train)

        kfold_start = time.perf_counter()
        if kfold_enabled:
            _stage("kfold")
            kfold_raw = self._kfold_validation(
                model_type=model_type,
                x=x,
                y=y,
                n_components=tuned_params.get("n_components", n_components),
                model_params=tuned_params,
                random_state=random_state,
                n_splits=n_splits,
            )
        else:
            kfold_raw = {"status": "skipped", "reason": "k-fold disabled by user"}
        kfold_time_sec = round(time.perf_counter() - kfold_start, 6)

        if model_type in {"pls", "svr"}:
            _stage("validation")
            y_true = np.asarray(y_test, dtype=float)
            y_pred = np.asarray(eval_model.predict(x_test), dtype=float)
            holdout_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
            kfold = self._format_kfold_regression(kfold_raw, kfold_time_sec)
            _stage("permutation")
            permutation_start = time.perf_counter()
            if model_type == "pls" and permutation_enabled is not False:
                permutation = self._permutation_test(
                    model_type=model_type,
                    x=x,
                    y=y,
                    n_components=tuned_params.get("n_components", n_components),
                    random_state=random_state,
                    n_permutations=n_permutations,
                )
                permutation["enabled"] = True
            elif model_type == "pls" and permutation_enabled is False:
                permutation = {"enabled": False, "status": "skipped", "reason": "permutation test disabled by user"}
            else:
                permutation = {
                    "enabled": False,
                    "status": "skipped",
                    "reason": "implemented only for PLS-DA and PLS Regression in current version",
                }
            permutation["time_sec"] = round(time.perf_counter() - permutation_start, 6)
            _stage("bootstrap")
            bootstrap_start = time.perf_counter()
            if bootstrap_enabled:
                bootstrap = self._bootstrap_ci_regression(y_true=y_true, y_pred=y_pred, random_state=random_state, n_bootstrap=n_bootstrap)
            else:
                bootstrap = {"enabled": False, "status": "skipped", "reason": "bootstrap disabled by user"}
            bootstrap["time_sec"] = round(time.perf_counter() - bootstrap_start, 6)
            return {
                "status": "ok",
                "mode": "holdout_regression",
                "train_samples": int(len(x_train)),
                "test_samples": int(len(x_test)),
                "metrics": holdout_metrics,
                "holdout_metrics": holdout_metrics,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "kfold": kfold,
                "permutation_test": permutation,
                "bootstrap_ci": bootstrap,
                "bootstrap": bootstrap,
                "hyperparameter_search": search_report,
                "limitations": self._build_limitations(
                    model_type=model_type,
                    sample_count=len(x),
                    class_labels=None,
                ),
                "warnings": warnings,
            }, eval_model, tuned_params, time.perf_counter() - validation_start

        _stage("validation")
        y_true_cls = np.asarray(y_test, dtype=str)
        y_pred_cls = np.asarray(eval_model.predict(x_test), dtype=str)
        labels = sorted(np.unique(np.concatenate([y_true_cls, y_pred_cls])).tolist())
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
        cm_total = self._confusion_matrix_total(cm)
        cm_accuracy = self._confusion_matrix_accuracy(cm)
        holdout_metrics = {
            "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
            "precision_macro": float(precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
        }
        validation_metrics = {
            **holdout_metrics,
            "confusion_matrix": cm.tolist(),
            "classes": labels,
            "y_true": y_true_cls.tolist(),
            "y_pred": y_pred_cls.tolist(),
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "validation_mode": "holdout_classification",
            "confusion_matrix_total": cm_total,
            "confusion_matrix_accuracy": cm_accuracy,
        }
        if cm_accuracy is not None and abs(validation_metrics["accuracy"] - cm_accuracy) > 1e-6:
            warnings.append("Validation confusion_matrix_accuracy differs from accuracy")
        kfold = self._format_kfold_classification(kfold_raw, kfold_time_sec)
        _stage("permutation")
        permutation_start = time.perf_counter()
        if model_type == "plsda" and permutation_enabled is not False:
            permutation = self._permutation_test(
                model_type=model_type,
                x=x,
                y=y,
                n_components=tuned_params.get("n_components", n_components),
                random_state=random_state,
                n_permutations=n_permutations,
            )
            permutation["enabled"] = True
        elif model_type == "plsda" and permutation_enabled is False:
            permutation = {"enabled": False, "status": "skipped", "reason": "permutation test disabled by user"}
        else:
            permutation = {
                "enabled": False,
                "status": "skipped",
                "reason": "implemented only for PLS-DA and PLS Regression in current version",
            }
        permutation["time_sec"] = round(time.perf_counter() - permutation_start, 6)
        _stage("bootstrap")
        bootstrap_start = time.perf_counter()
        if bootstrap_enabled:
            bootstrap = self._bootstrap_ci_classification(y_true=y_true_cls, y_pred=y_pred_cls, random_state=random_state, n_bootstrap=n_bootstrap)
        else:
            bootstrap = {"enabled": False, "status": "skipped", "reason": "bootstrap disabled by user"}
        bootstrap["time_sec"] = round(time.perf_counter() - bootstrap_start, 6)
        return {
            "status": "ok",
            "mode": "holdout_classification",
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "classes": labels,
            "confusion_matrix": cm.tolist(),
            "y_true": y_true_cls.tolist(),
            "y_pred": y_pred_cls.tolist(),
            "kfold": kfold,
            "permutation_test": permutation,
            "bootstrap_ci": bootstrap,
            "bootstrap": bootstrap,
            "hyperparameter_search": search_report,
            "limitations": self._build_limitations(
                model_type=model_type,
                sample_count=len(x),
                class_labels=np.asarray(y, dtype=str),
            ),
            "warnings": warnings,
        }, eval_model, tuned_params, time.perf_counter() - validation_start

    @staticmethod
    def _format_kfold_classification(kfold_raw: Dict[str, Any], time_sec: float) -> Dict[str, Any]:
        if kfold_raw.get("status") != "ok":
            return {"enabled": True, "status": kfold_raw.get("status", "skipped"), "reason": kfold_raw.get("reason"), "time_sec": time_sec}
        mean_std = kfold_raw.get("mean_std") or {}
        f1 = mean_std.get("f1_macro") or {}
        return {
            "enabled": True,
            "status": "ok",
            "n_splits": kfold_raw.get("n_splits"),
            "metric": "f1_macro",
            "mean": f1.get("mean"),
            "std": f1.get("std"),
            "fold_metrics": kfold_raw.get("fold_metrics"),
            "mean_std": mean_std,
            "time_sec": time_sec,
        }

    @staticmethod
    def _format_kfold_regression(kfold_raw: Dict[str, Any], time_sec: float) -> Dict[str, Any]:
        if kfold_raw.get("status") != "ok":
            return {"enabled": True, "status": kfold_raw.get("status", "skipped"), "reason": kfold_raw.get("reason"), "time_sec": time_sec}
        mean_std = kfold_raw.get("mean_std") or {}
        rmse = mean_std.get("rmse") or {}
        r2 = mean_std.get("r2") or {}
        return {
            "enabled": True,
            "status": "ok",
            "n_splits": kfold_raw.get("n_splits"),
            "rmse_mean": rmse.get("mean"),
            "rmse_std": rmse.get("std"),
            "r2_mean": r2.get("mean"),
            "r2_std": r2.get("std"),
            "fold_metrics": kfold_raw.get("fold_metrics"),
            "mean_std": mean_std,
            "time_sec": time_sec,
        }

    def _kfold_validation(
        self,
        model_type: str,
        x: np.ndarray,
        y: np.ndarray,
        n_components: Optional[int],
        model_params: Optional[Dict[str, Any]],
        random_state: int,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        n_samples = len(x)
        if n_samples < 4:
            return {"status": "skipped", "reason": "Too few samples for k-fold"}
        requested_splits = max(2, int(n_splits or 5))

        metrics_per_fold: List[Dict[str, float]] = []
        if model_type in {"plsda", "svm", "random_forest", "decision_tree"}:
            y_cls = np.asarray(y, dtype=str)
            _, counts = np.unique(y_cls, return_counts=True)
            n_splits = min(requested_splits, int(np.min(counts)))
            if n_splits < 2:
                return {"status": "skipped", "reason": "Not enough samples per class for stratified k-fold"}
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
            split_iter = splitter.split(x, y_cls)
        else:
            n_splits = min(requested_splits, n_samples)
            if n_splits < 2:
                return {"status": "skipped", "reason": "Not enough samples for k-fold"}
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
            split_iter = splitter.split(x)

        for train_idx, test_idx in split_iter:
            model = create_model(model_type=model_type, n_components=n_components, model_params=model_params)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            if model_type in {"pls", "svr"}:
                y_true = np.asarray(y_test, dtype=float)
                y_pred_num = np.asarray(y_pred, dtype=float)
                metrics_per_fold.append(
                    {
                        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_num))),
                        "mae": float(mean_absolute_error(y_true, y_pred_num)),
                        "r2": float(r2_score(y_true, y_pred_num)),
                    }
                )
            else:
                y_true_cls = np.asarray(y_test, dtype=str)
                y_pred_cls = np.asarray(y_pred, dtype=str)
                metrics_per_fold.append(
                    {
                        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
                        "f1_macro": float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
                    }
                )

        summary: Dict[str, Any] = {"status": "ok", "n_splits": n_splits, "fold_metrics": metrics_per_fold}
        if metrics_per_fold:
            keys = metrics_per_fold[0].keys()
            summary["mean_std"] = {
                key: {
                    "mean": float(np.mean([m[key] for m in metrics_per_fold])),
                    "std": float(np.std([m[key] for m in metrics_per_fold], ddof=0)),
                }
                for key in keys
            }
        return summary

    def _permutation_test(
        self,
        model_type: str,
        x: np.ndarray,
        y: np.ndarray,
        n_components: Optional[int],
        random_state: int,
        n_permutations: int = 200,
    ) -> Dict[str, Any]:
        if len(x) < 6:
            return {"status": "skipped", "reason": "Too few samples for permutation test"}

        rng = np.random.default_rng(int(random_state))
        true_cv = self._kfold_validation(
            model_type=model_type,
            x=x,
            y=y,
            n_components=n_components,
            model_params=None,
            random_state=random_state,
        )
        if true_cv.get("status") != "ok":
            return {"status": "skipped", "reason": "k-fold validation unavailable for permutation test"}

        if model_type == "pls":
            observed = true_cv["mean_std"]["r2"]["mean"]
            metric_name = "r2"
        else:
            observed = true_cv["mean_std"]["accuracy"]["mean"]
            metric_name = "accuracy"

        perm_scores: List[float] = []
        y_arr = np.asarray(y, dtype=object)
        for _ in range(n_permutations):
            y_perm = rng.permutation(y_arr)
            perm_cv = self._kfold_validation(
                model_type=model_type,
                x=x,
                y=y_perm,
                n_components=n_components,
                model_params=None,
                random_state=int(rng.integers(1, 1_000_000)),
            )
            if perm_cv.get("status") != "ok":
                continue
            perm_scores.append(float(perm_cv["mean_std"][metric_name]["mean"]))

        if not perm_scores:
            return {"status": "skipped", "reason": "Permutation test failed to produce scores"}

        if model_type == "pls":
            # for r2: larger is better
            p_value = (sum(score >= observed for score in perm_scores) + 1) / (len(perm_scores) + 1)
        else:
            p_value = (sum(score >= observed for score in perm_scores) + 1) / (len(perm_scores) + 1)

        return {
            "status": "ok",
            "metric": metric_name,
            "observed_score": float(observed),
            "p_value": float(p_value),
            "n_permutations": int(len(perm_scores)),
            "permuted_score_mean": float(np.mean(perm_scores)),
            "permuted_score_std": float(np.std(perm_scores, ddof=0)),
        }

    def _bootstrap_ci_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        random_state: int,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(random_state))
        n = len(y_true)
        if n < 3:
            return {"enabled": True, "status": "skipped", "reason": "Too few samples for bootstrap CI"}

        acc_scores: List[float] = []
        f1_scores: List[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            acc_scores.append(float(accuracy_score(y_t, y_p)))
            f1_scores.append(float(f1_score(y_t, y_p, average="macro", zero_division=0)))

        f1_ci = [float(np.percentile(f1_scores, 2.5)), float(np.percentile(f1_scores, 97.5))]
        return {
            "enabled": True,
            "status": "ok",
            "n_bootstrap": n_bootstrap,
            "n_iterations": n_bootstrap,
            "metric": "f1_macro",
            "ci_95": f1_ci,
            "accuracy_95ci": [float(np.percentile(acc_scores, 2.5)), float(np.percentile(acc_scores, 97.5))],
            "f1_macro_95ci": f1_ci,
        }

    def _bootstrap_ci_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        random_state: int,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(random_state))
        n = len(y_true)
        if n < 3:
            return {"enabled": True, "status": "skipped", "reason": "Too few samples for bootstrap CI"}

        rmse_scores: List[float] = []
        r2_scores: List[float] = []
        mae_scores: List[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            rmse_scores.append(float(np.sqrt(mean_squared_error(y_t, y_p))))
            mae_scores.append(float(mean_absolute_error(y_t, y_p)))
            r2_scores.append(float(r2_score(y_t, y_p)))

        rmse_ci = [float(np.percentile(rmse_scores, 2.5)), float(np.percentile(rmse_scores, 97.5))]
        return {
            "enabled": True,
            "status": "ok",
            "n_bootstrap": n_bootstrap,
            "n_iterations": n_bootstrap,
            "metric": "rmse",
            "ci_95": rmse_ci,
            "rmse_95ci": rmse_ci,
            "mae_95ci": [float(np.percentile(mae_scores, 2.5)), float(np.percentile(mae_scores, 97.5))],
            "r2_95ci": [float(np.percentile(r2_scores, 2.5)), float(np.percentile(r2_scores, 97.5))],
        }

    def _build_limitations(
        self,
        model_type: str,
        sample_count: int,
        class_labels: Optional[np.ndarray],
    ) -> List[str]:
        limitations = [
            f"Набор данных небольшой ({sample_count} спектров), возможна нестабильность оценок.",
            "Результаты валидны только для текущего домена измерений и протокола пробоподготовки.",
            "Нужна внешняя проверка на независимой выборке перед практическим применением.",
        ]
        if model_type == "plsda" and class_labels is not None:
            unique, counts = np.unique(class_labels, return_counts=True)
            if len(unique) > 0:
                limitations.append(
                    "Распределение классов: "
                    + ", ".join([f"{cls}={cnt}" for cls, cnt in zip(unique.tolist(), counts.tolist())])
                )
        limitations.append("Метрики являются вычислительной оценкой качества модели и требуют проверки на независимых данных.")
        return limitations

    @staticmethod
    def _metadata_axis(metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        raw_axis = metadata.get("spectral_axis") or metadata.get("axis")
        if raw_axis:
            axis = np.asarray(raw_axis, dtype=float)
            if axis.ndim == 1 and axis.size:
                return axis
        count = AnalysisService._expected_feature_count(metadata)
        axis_min = metadata.get("axis_min")
        axis_max = metadata.get("axis_max")
        if count and axis_min is not None and axis_max is not None:
            try:
                return np.linspace(float(axis_min), float(axis_max), int(count), dtype=float)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _raw_metadata_axis(metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        raw_axis = metadata.get("raw_spectral_axis")
        if not raw_axis:
            return None
        axis = np.asarray(raw_axis, dtype=float)
        return axis if axis.ndim == 1 and axis.size else None

    @staticmethod
    def _axis_close(left: np.ndarray, right: np.ndarray) -> bool:
        return left.shape == right.shape and bool(np.allclose(left, right, rtol=1e-6, atol=1e-8))

    @staticmethod
    def _expected_feature_count(metadata: Dict[str, Any]) -> int:
        for key in ("expected_feature_count", "feature_count", "n_features"):
            value = metadata.get(key)
            if value not in (None, ""):
                try:
                    count = int(value)
                    if count > 0:
                        return count
                except (TypeError, ValueError):
                    pass
        input_shape = metadata.get("input_shape")
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            try:
                count = int(input_shape[1])
                if count > 0:
                    return count
            except (TypeError, ValueError):
                pass
        axis = metadata.get("spectral_axis") or metadata.get("axis")
        if isinstance(axis, list) and axis:
            return len(axis)
        return 0

    @staticmethod
    def _model_requires_preprocessing(metadata: Dict[str, Any]) -> bool:
        version = str(metadata.get("dataset_version") or "").strip().lower()
        return bool(metadata.get("used_processed_data")) or version == "processed"

    @staticmethod
    def _metadata_preprocessing_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("preprocessing_config", "preprocessing"):
            value = metadata.get(key)
            if isinstance(value, dict) and value:
                return value
        return {}

    @staticmethod
    def _as_imported_dataset(dataset: SpectrumDataset) -> ImportedSpectralDataset:
        axis = np.asarray(dataset.spectral_axis if dataset.spectral_axis is not None else np.arange(dataset.feature_count), dtype=float)
        return ImportedSpectralDataset(
            X=np.asarray(dataset.x, dtype=float),
            axis=axis,
            sample_ids=list(dataset.sample_names),
            y=None,
            target_name=None,
            class_names=None,
            metadata=dict(dataset.metadata or {}),
            source_layout=dataset.source_format,
            source_files=[dataset.filename] if dataset.filename else [],
        )

    def _prepare_inference_dataset(self, dataset: SpectrumDataset, metadata: Dict[str, Any]) -> tuple[SpectrumDataset, Dict[str, Any], List[str]]:
        warnings: List[str] = []
        input_axis = np.asarray(dataset.spectral_axis if dataset.spectral_axis is not None else np.arange(dataset.feature_count), dtype=float)
        X = np.asarray(dataset.x, dtype=float)
        working_axis = input_axis
        preprocessing_config = self._metadata_preprocessing_config(metadata)
        requires_preprocessing = self._model_requires_preprocessing(metadata)
        preprocessing_applied = False
        raw_axis_interpolated = False

        expected_axis = self._metadata_axis(metadata)
        interpolated = False
        range_warning = False
        input_was_single_column = bool((dataset.metadata or {}).get("single_column_intensity_only"))
        if input_was_single_column and expected_axis is not None:
            expected_features_for_axis = self._expected_feature_count(metadata) or expected_axis.size
            if X.shape[1] == expected_features_for_axis:
                working_axis = expected_axis
                warnings.append("TXT-файл содержит только интенсивности без спектральной оси; использована spectral_axis сохранённой модели.")
            else:
                warnings.append(
                    f"TXT-файл содержит только интенсивности без спектральной оси; число точек ({X.shape[1]}) не совпадает с моделью ({expected_features_for_axis})."
                )
        if expected_axis is not None and not self._axis_close(working_axis, expected_axis):
            source_min = float(np.nanmin(working_axis))
            source_max = float(np.nanmax(working_axis))
            expected_min = float(np.nanmin(expected_axis))
            expected_max = float(np.nanmax(expected_axis))
            if source_min > expected_min or source_max < expected_max:
                range_warning = True
                warnings.append("Диапазон оси входного спектра не полностью покрывает ось модели; края будут заполнены ближайшими значениями.")
            X = np.vstack([np.interp(expected_axis, working_axis, row, left=float(row[0]), right=float(row[-1])) for row in X])
            working_axis = expected_axis
            interpolated = True
            warnings.append("Входной спектр интерполирован на спектральную ось сохранённой модели.")

        if requires_preprocessing:
            if not preprocessing_config:
                raise ValueError(
                    "Модель обучена на processed-данных, но в metadata модели нет preprocessing_config. "
                    "Невозможно корректно применить модель к raw TXT."
                )
            preprocessing_dataset = ImportedSpectralDataset(
                X=X,
                axis=working_axis,
                sample_ids=list(dataset.sample_names),
                y=None,
                target_name=None,
                class_names=None,
                metadata=dict(dataset.metadata or {}),
                source_layout=dataset.source_format,
                source_files=[dataset.filename] if dataset.filename else [],
            )
            processed = preprocess_matrix(preprocessing_dataset, preprocessing_config)
            X = np.asarray(processed["X"], dtype=float)
            working_axis = np.asarray(processed["axis"], dtype=float)
            preprocessing_applied = True
            warnings.extend([str(item) for item in processed.get("warnings", []) if item])

        expected_features = self._expected_feature_count(metadata)
        if expected_features and X.shape[1] != expected_features:
            raise ValueError(
                f"Число признаков после подготовки входа ({X.shape[1]}) не совпадает с ожидаемым числом признаков модели ({expected_features}). "
                "Проверьте формат файла, спектральную ось и настройки предобработки."
            )

        prepared = SpectrumDataset(
            x=X,
            spectral_axis=working_axis,
            sample_names=list(dataset.sample_names),
            source_format=dataset.source_format,
            filename=dataset.filename,
            file_format=dataset.file_format,
            metadata={**dict(dataset.metadata or {}), "prepared_for_model": metadata.get("model_id")},
        )
        report = {
            "raw_feature_count": int(dataset.feature_count),
            "input_feature_count": int(dataset.feature_count),
            "prepared_feature_count": int(prepared.feature_count),
            "expected_feature_count": expected_features or int(prepared.feature_count),
            "axis_min_input": float(np.nanmin(input_axis)) if input_axis.size else None,
            "axis_max_input": float(np.nanmax(input_axis)) if input_axis.size else None,
            "input_axis_min": float(np.nanmin(input_axis)) if input_axis.size else None,
            "input_axis_max": float(np.nanmax(input_axis)) if input_axis.size else None,
            "axis_min_model": float(np.nanmin(expected_axis)) if expected_axis is not None and expected_axis.size else None,
            "axis_max_model": float(np.nanmax(expected_axis)) if expected_axis is not None and expected_axis.size else None,
            "prepared_axis_min": float(np.nanmin(working_axis)) if working_axis.size else None,
            "prepared_axis_max": float(np.nanmax(working_axis)) if working_axis.size else None,
            "model_axis_min": float(np.nanmin(expected_axis)) if expected_axis is not None and expected_axis.size else None,
            "model_axis_max": float(np.nanmax(expected_axis)) if expected_axis is not None and expected_axis.size else None,
            "interpolated_to_model_axis": interpolated,
            "interpolated": interpolated,
            "interpolated_to_raw_training_axis": raw_axis_interpolated,
            "range_warning": range_warning,
            "requires_preprocessing": requires_preprocessing,
            "preprocessing_applied": preprocessing_applied,
            "preprocessing_config": preprocessing_config if preprocessing_applied else {},
            "warnings": list(warnings),
        }
        return prepared, report, warnings

    def infer(self, model_type: str, model_id: str, dataset: SpectrumDataset) -> Dict[str, Any]:
        model, metadata = self.model_manager.load(model_type=model_type, model_id=model_id)
        if self._normalize_model_type(metadata.get("model_type", model_type)) == "hca":
            raise ValueError(
                "This saved HCA result cannot be used for direct prediction on new spectra. "
                "Re-run clustering on the full dataset."
            )
        prepared, report, warnings = self._prepare_inference_dataset(dataset, metadata)
        expected_features = self._expected_feature_count(metadata)
        if expected_features and prepared.feature_count != expected_features:
            raise ValueError(
                f"Количество признаков во входном файле ({dataset.feature_count}) не совпадает "
                f"с обученной моделью ({expected_features})."
            )
        predict_start = time.perf_counter()
        result = model.inference_result(prepared.x)
        predict_time_sec = time.perf_counter() - predict_start
        plot = self._build_plot(metadata.get("model_type", model_type), result, phase="inference")
        predictions = self._prediction_rows(result, prepared.sample_names)
        sample_count = max(1, prepared.sample_count)
        return {
            "status": "success",
            "model_id": metadata.get("model_id", model_id),
            "model_type": metadata.get("model_type", model_type),
            "model_metadata": metadata,
            "inference_report": report,
            "result": result,
            "predictions": predictions,
            "plots": {"prediction_distribution": plot} if plot else {},
            "plot": plot,
            "warnings": warnings,
            "performance": {"predict_time_one_sample_ms": round((predict_time_sec / sample_count) * 1000.0, 4), "predict_time_total_sec": round(predict_time_sec, 6)},
        }

    def infer_many(
        self,
        model_type: str,
        model_id: str,
        datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        model, metadata = self.model_manager.load(model_type=model_type, model_id=model_id)
        if self._normalize_model_type(metadata.get("model_type", model_type)) == "hca":
            raise ValueError(
                "This saved HCA result cannot be used for direct prediction on new spectra. "
                "Re-run clustering on the full dataset."
            )
        expected_features = self._expected_feature_count(metadata)
        all_results: List[Dict[str, Any]] = []
        all_predicted_classes: List[str] = []
        all_predictions: List[float] = []
        reports: List[Dict[str, Any]] = []
        warnings: List[str] = []
        total_samples = 0
        predict_time_sec = 0.0

        for item in datasets:
            filename = item["filename"]
            dataset: SpectrumDataset = item["dataset"]
            prepared, report, item_warnings = self._prepare_inference_dataset(dataset, metadata)
            report["filename"] = filename
            reports.append(report)
            warnings.extend([f"{filename}: {warning}" for warning in item_warnings])
            if expected_features and prepared.feature_count != expected_features:
                raise ValueError(
                    f"Файл {filename}: число признаков ({dataset.feature_count}) не совпадает "
                    f"с обученной моделью ({expected_features})."
                )
            predict_start = time.perf_counter()
            result = model.inference_result(prepared.x)
            predict_time_sec += time.perf_counter() - predict_start
            total_samples += max(1, prepared.sample_count)
            row = {"filename": filename, "result": result, "inference_report": report}
            all_results.append(row)

            if "predicted_classes" in result:
                all_predicted_classes.extend([str(v) for v in result["predicted_classes"]])
            if "predictions" in result:
                all_predictions.extend([float(v) for v in result["predictions"]])

        aggregate_result: Dict[str, Any] = {}
        if all_predicted_classes:
            aggregate_result["predicted_classes"] = all_predicted_classes
            aggregate_result["classes"] = sorted(list(set(all_predicted_classes)))
        if all_predictions:
            aggregate_result["predictions"] = all_predictions

        plot = self._build_plot(metadata.get("model_type", model_type), aggregate_result, phase="inference")
        prediction_rows: List[Dict[str, Any]] = []
        for item in all_results:
            filename = item["filename"]
            result = item["result"]
            names = result.get("sample_ids") or []
            rows = self._prediction_rows(result, names)
            for row in rows:
                row["source_file"] = filename
            prediction_rows.extend(rows)
        if len(prediction_rows) > 1:
            unique_predictions = {str(row.get("predicted")) for row in prediction_rows}
            if len(unique_predictions) == 1:
                warnings.append("Все файлы отнесены к одному классу. Проверьте ось спектра, предобработку и применимость модели.")
        return {
            "status": "success",
            "model_id": metadata.get("model_id", model_id),
            "model_type": metadata.get("model_type", model_type),
            "model_metadata": metadata,
            "batch_results": all_results,
            "aggregate_result": aggregate_result,
            "inference_reports": reports,
            "predictions": prediction_rows,
            "plots": {"prediction_distribution": plot} if plot else {},
            "plot": plot,
            "warnings": warnings,
            "performance": {
                "predict_time_one_sample_ms": round((predict_time_sec / max(1, total_samples)) * 1000.0, 4),
                "predict_time_total_sec": round(predict_time_sec, 6),
            },
        }

    @staticmethod
    def _prediction_rows(result: Dict[str, Any], sample_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        values = result.get("predicted_classes")
        if values is None:
            values = result.get("predictions")
        if values is None:
            values = result.get("scores")
        if values is None:
            return []
        rows = []
        for idx, value in enumerate(values):
            predicted = value
            if isinstance(value, list):
                predicted = ";".join(str(v) for v in value)
            confidence = None
            if isinstance(result.get("confidence"), list) and idx < len(result["confidence"]):
                confidence = result["confidence"][idx]
            probabilities = {}
            if isinstance(result.get("probabilities"), list) and idx < len(result["probabilities"]):
                probabilities = result["probabilities"][idx]
            decision_score = None
            if isinstance(result.get("decision_score"), list) and idx < len(result["decision_score"]):
                decision_score = result["decision_score"][idx]
            rows.append({
                "sample_id": sample_ids[idx] if sample_ids and idx < len(sample_ids) else f"sample_{idx + 1}",
                "predicted": predicted,
                "confidence": confidence if confidence is not None else "вероятность недоступна",
                "confidence_kind": result.get("confidence_kind") or "unavailable",
                "probability_available": bool(result.get("probability_available")),
                "probabilities": probabilities,
                "decision_score": decision_score,
            })
        return rows

    @staticmethod
    def _training_prediction_rows(result: Dict[str, Any], sample_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if "y_true" not in result or "y_pred" not in result:
            return []
        rows: List[Dict[str, Any]] = []
        y_true = result.get("y_true") or []
        y_pred = result.get("y_pred") or []
        for idx, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
            try:
                residual = float(true_value) - float(pred_value)
            except (TypeError, ValueError):
                residual = None
            rows.append(
                {
                    "sample_id": sample_ids[idx] if sample_ids and idx < len(sample_ids) else f"sample_{idx + 1}",
                    "y_true": float(true_value) if isinstance(true_value, (int, float)) or str(true_value).replace(".", "", 1).replace("-", "", 1).isdigit() else true_value,
                    "y_pred": float(pred_value) if isinstance(pred_value, (int, float)) or str(pred_value).replace(".", "", 1).replace("-", "", 1).isdigit() else pred_value,
                    "residual": residual,
                }
            )
        return rows

    def _build_plots(self, model_type: str, result: Dict[str, Any], phase: str, dataset: Optional[SpectrumDataset] = None) -> List[Dict[str, Any]]:
        primary = self._build_plot(model_type, result, phase)
        plots = [primary] if primary else []
        normalized = self._normalize_model_type(model_type)
        if phase == "training" and normalized == "pca":
            plots.extend(self._pca_extra_plots(result))
            if dataset is not None:
                plots.extend(self._pca_metadata_plots(result, dataset))
        if phase == "training" and normalized in {"kmeans", "hca"} and dataset is not None:
            plots = self._cluster_plots(normalized, result, dataset) or plots
        if phase == "training" and normalized in {"pls", "svr"} and "y_true" in result and "y_pred" in result:
            y_true = np.asarray(result["y_true"], dtype=float)
            y_pred = np.asarray(result["y_pred"], dtype=float)
            residuals = y_true - y_pred
            samples = result.get("sample_ids") or [f"sample_{idx + 1}" for idx in range(len(y_true))]
            plots.append(
                {
                    "data": [
                        {
                            "type": "scatter",
                            "mode": "markers",
                            "x": y_pred.tolist(),
                            "y": residuals.tolist(),
                            "text": samples,
                            "hovertemplate": "sample_id: %{text}<br>y_pred: %{x:.4g}<br>residual: %{y:.4g}<extra></extra>",
                        },
                        {"type": "scatter", "mode": "lines", "x": [float(np.min(y_pred)), float(np.max(y_pred))], "y": [0, 0], "name": "0", "line": {"dash": "dash"}},
                    ],
                    "layout": {"title": "График остатков", "xaxis": {"title": "Предсказанное значение"}, "yaxis": {"title": "Остаток"}, "template": "plotly_white"},
                }
            )
            plots.append(
                {
                    "data": [{"type": "histogram", "x": residuals.tolist(), "name": "Остатки"}],
                    "layout": {"title": "Распределение остатков", "xaxis": {"title": "Остаток"}, "yaxis": {"title": "Количество"}, "template": "plotly_white"},
                }
            )
            plots.append(
                {
                    "data": [
                        {"type": "scatter", "mode": "lines+markers", "x": list(range(1, len(y_pred) + 1)), "y": y_pred.tolist(), "name": "y_pred"},
                        {"type": "scatter", "mode": "markers", "x": list(range(1, len(y_true) + 1)), "y": y_true.tolist(), "name": "y_true"},
                    ],
                    "layout": {"title": "Истинные и предсказанные значения по образцам", "xaxis": {"title": "Номер образца"}, "yaxis": {"title": "Target"}, "template": "plotly_white"},
                }
            )
            scores = np.asarray(result.get("x_scores", []), dtype=float)
            if normalized == "pls" and scores.ndim == 2 and scores.shape[0] == len(y_true) and scores.shape[1] >= 1:
                plots.append(self._scatter_scores_plot(scores, y_true.tolist(), "PLS: латентные переменные", "LV1", "LV2", samples))
        return plots

    def _pca_extra_plots(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        ratios = [float(v) for v in result.get("explained_variance_ratio", [])]
        if not ratios:
            return []
        x = [f"PC{i + 1}" for i in range(len(ratios))]
        cumulative = np.cumsum(ratios).tolist()
        return [
            {
                "data": [{"type": "bar", "x": x, "y": ratios, "name": "Доля дисперсии"}],
                "layout": {"title": "PCA: доля объяснённой дисперсии", "xaxis": {"title": "Компонента"}, "yaxis": {"title": "Explained variance ratio"}, "template": "plotly_white"},
            },
            {
                "data": [{"type": "scatter", "mode": "lines+markers", "x": x, "y": cumulative, "name": "Накопленная доля"}],
                "layout": {"title": "PCA: накопленная объяснённая дисперсия", "xaxis": {"title": "Компонента"}, "yaxis": {"title": "Cumulative explained variance"}, "template": "plotly_white"},
            },
        ]

    def _pca_metadata_plots(self, result: Dict[str, Any], dataset: SpectrumDataset) -> List[Dict[str, Any]]:
        scores = np.asarray(result.get("scores", []), dtype=float)
        if scores.ndim != 2 or scores.shape[0] == 0:
            return []
        plots: List[Dict[str, Any]] = []
        sample_ids = result.get("sample_ids") or dataset.sample_names
        for key, title in [("slit_width_um", "PCA: цвет по ширине щели"), ("grating_lines_mm", "PCA: цвет по решётке")]:
            labels = self._metadata_series(dataset, key)
            if len(labels) == scores.shape[0]:
                plots.append(self._scatter_scores_plot(scores, labels, title, "PC1", "PC2", sample_ids))
        return plots

    def _metadata_series(self, dataset: SpectrumDataset, key: str) -> List[Any]:
        meta = dataset.metadata or {}
        rows = meta.get("sample_metadata") or meta.get("sample_metadata_preview") or []
        if len(rows) == dataset.sample_count and any(isinstance(row, dict) and key in row for row in rows):
            return [row.get(key) if isinstance(row, dict) else None for row in rows]
        values = meta.get(f"{key}_values") or []
        if len(values) == 1:
            return [values[0]] * dataset.sample_count
        return []

    def _cluster_plots(self, model_type: str, result: Dict[str, Any], dataset: SpectrumDataset) -> List[Dict[str, Any]]:
        labels = result.get("cluster_labels") or []
        if not labels:
            return []
        labels_str = [str(v) for v in labels]
        plots: List[Dict[str, Any]] = []
        x = np.asarray(dataset.x, dtype=float)
        sample_ids = dataset.sample_names or [f"sample_{idx + 1}" for idx in range(x.shape[0])]
        if x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] >= 2:
            x_scaled = (x - np.nanmean(x, axis=0)) / np.where(np.nanstd(x, axis=0) == 0, 1, np.nanstd(x, axis=0))
            scores = PCA(n_components=2).fit_transform(np.nan_to_num(x_scaled))
            plots.append(self._scatter_scores_plot(scores, labels_str, f"{model_type}: PCA-проекция кластеров", "PC1", "PC2", sample_ids))
        plots.append(class_count_plot(labels_str, f"{model_type}: распределение кластеров"))
        axis = np.asarray(dataset.spectral_axis, dtype=float)
        traces = []
        for label in sorted(set(labels_str)):
            idx = [i for i, value in enumerate(labels_str) if value == label]
            if idx:
                traces.append({"type": "scatter", "mode": "lines", "x": axis.tolist(), "y": np.nanmean(x[idx], axis=0).tolist(), "name": f"Кластер {label}"})
        if traces:
            plots.append({"data": traces, "layout": {"title": "Средние спектры по кластерам", "xaxis": {"title": "Спектральная ось"}, "yaxis": {"title": "Интенсивность"}, "template": "plotly_white"}})
        slit = self._metadata_series(dataset, "slit_width_um")
        if len(slit) == len(labels_str):
            plots.append({
                "data": [{"type": "box", "x": labels_str, "y": [float(v) if v is not None else None for v in slit], "boxpoints": "all"}],
                "layout": {"title": "Ширина щели по кластерам", "xaxis": {"title": "Кластер"}, "yaxis": {"title": "Ширина щели, мкм"}, "template": "plotly_white"},
            })
        grating = self._metadata_series(dataset, "grating_lines_mm")
        if len(grating) == len(labels_str):
            cluster_values = sorted(set(labels_str))
            grating_values = sorted({str(v) for v in grating if v is not None})
            data = []
            for gr in grating_values:
                data.append({"type": "bar", "name": f"{gr} штр./мм", "x": cluster_values, "y": [sum(1 for lab, g in zip(labels_str, grating) if lab == cl and str(g) == gr) for cl in cluster_values]})
            plots.append({"data": data, "layout": {"title": "Решётка внутри кластеров", "barmode": "stack", "xaxis": {"title": "Кластер"}, "yaxis": {"title": "Количество спектров"}, "template": "plotly_white"}})
        target = result.get("target") or []
        if target and len(target) == len(labels_str):
            classes = sorted({str(v) for v in target})
            clusters = sorted(set(labels_str))
            matrix = [[sum(1 for t, c in zip(target, labels_str) if str(t) == cls and c == cl) for cl in clusters] for cls in classes]
            plots.append({"data": [{"type": "heatmap", "z": matrix, "x": clusters, "y": classes, "colorscale": "Blues", "text": matrix}], "layout": {"title": "Реальные классы и кластеры", "xaxis": {"title": "Кластер"}, "yaxis": {"title": "Реальный класс"}, "template": "plotly_white"}})
        return plots

    def _build_plot(self, model_type: str, result: Dict[str, Any], phase: str) -> Dict[str, Any]:
        normalized = self._normalize_model_type(model_type)

        if normalized == "pca":
            scores = np.asarray(result.get("scores", []), dtype=float)
            if scores.size == 0:
                return {}
            ratios = [float(v) for v in result.get("explained_variance_ratio", [])]
            x_title = f"PC1 ({ratios[0] * 100:.1f}% дисперсии)" if len(ratios) > 0 else "PC1"
            y_title = f"PC2 ({ratios[1] * 100:.1f}% дисперсии)" if len(ratios) > 1 else "PC2"
            return self._scatter_scores_plot(scores, result.get("target"), "PCA: PC1-PC2", x_title, y_title, result.get("sample_ids"))

        if normalized in {"pls", "svr"}:
            if phase == "training" and "y_true" in result and "y_pred" in result:
                y_true = np.asarray(result["y_true"], dtype=float)
                y_pred = np.asarray(result["y_pred"], dtype=float)
                title = "PLS Regression: предсказание vs истина" if normalized == "pls" else "SVR: предсказание vs истина"
                return pls_regression_plot(y_true, y_pred, title)
            if phase == "inference" and "predictions" in result:
                preds = [float(v) for v in result["predictions"]]
                return residual_plot(preds, f"{normalized.upper()}: значения предсказаний")
            return {}

        if normalized in {"plsda", "svm", "random_forest", "decision_tree"}:
            if phase == "training" and result.get("confusion_matrix"):
                return self._confusion_matrix_plot(result.get("confusion_matrix"), result.get("classes") or [], f"{normalized}: confusion matrix")
            labels = result.get("y_pred") if phase == "training" else result.get("predicted_classes")
            if not labels:
                return {}
            if phase == "training" and result.get("confusion_matrix"):
                return self._confusion_matrix_plot(result.get("confusion_matrix"), result.get("classes") or [], f"{normalized}: матрица ошибок")
            return class_count_plot(labels, f"{normalized}: распределение предсказанных классов")

        if normalized in {"kmeans", "hca"}:
            labels = result.get("cluster_labels")
            if not labels:
                return {}
            return class_count_plot([str(v) for v in labels], f"{normalized}: распределение кластеров")

        return {}

    def _scatter_scores_plot(
        self,
        scores: np.ndarray,
        labels: Optional[List[Any]],
        title: str,
        x_title: str,
        y_title: str,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        x = scores[:, 0].tolist() if scores.shape[1] >= 1 else [0.0] * scores.shape[0]
        y = scores[:, 1].tolist() if scores.shape[1] >= 2 else [0.0] * scores.shape[0]
        labels_list = [str(v) for v in labels] if labels and len(labels) == scores.shape[0] else ["sample"] * scores.shape[0]
        samples = sample_ids if sample_ids and len(sample_ids) == scores.shape[0] else [f"sample_{i + 1}" for i in range(scores.shape[0])]
        traces: List[Dict[str, Any]] = []
        for label in sorted(set(labels_list)):
            idx = [i for i, value in enumerate(labels_list) if value == label]
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": label,
                    "x": [x[i] for i in idx],
                    "y": [y[i] for i in idx],
                    "marker": {"size": 10, "opacity": 0.85},
                    "text": [
                        f"sample_id: {samples[i]}<br>target: {labels_list[i]}<br>PC1: {x[i]:.4g}<br>PC2: {y[i]:.4g}"
                        for i in idx
                    ],
                    "hovertemplate": "%{text}<extra></extra>",
                }
            )
        return {
            "data": traces,
            "layout": {
                "title": title,
                "xaxis": {"title": x_title},
                "yaxis": {"title": y_title},
                "template": "plotly_white",
            },
        }

    def _confusion_matrix_plot(self, matrix: List[List[int]], classes: List[str], title: str) -> Dict[str, Any]:
        return {
            "data": [
                {
                    "type": "heatmap",
                    "z": matrix,
                    "x": classes,
                    "y": classes,
                    "colorscale": "Blues",
                    "showscale": True,
                }
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": "Предсказанный класс"},
                "yaxis": {"title": "Истинный класс"},
                "template": "plotly_white",
            },
        }
