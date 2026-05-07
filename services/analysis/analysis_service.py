from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from services.analysis.model_manager import ModelManager
from services.analysis.models import create_model
from services.analysis.spectrum_loader import SpectrumDataset, parse_spectrum_file, parse_target_values
from services.analysis.visualization import class_count_plot, pca_plot, pls_regression_plot, residual_plot


class AnalysisService:
    def __init__(self, model_root: Path):
        self.model_manager = ModelManager(model_root)

    @staticmethod
    def parse_file(raw: bytes, filename: str) -> SpectrumDataset:
        return parse_spectrum_file(raw, filename)

    @staticmethod
    def _parse_supervised_targets(model_type: str, raw_targets: Optional[str], sample_count: int) -> Optional[np.ndarray]:
        normalized = model_type.lower()
        if normalized in {"pls", "plsda", "pls-da"}:
            return parse_target_values(raw_targets, sample_count)
        return None

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
    ) -> Dict[str, Any]:
        x = dataset.x
        y = self._parse_supervised_targets(model_type, raw_targets, dataset.sample_count)

        normalized_model_type = model_type.strip().lower().replace("-", "")
        validation = self._run_validation(
            model_type=normalized_model_type,
            x=x,
            y=y,
            n_components=n_components,
            do_validation=do_validation,
            test_size=test_size,
            random_state=random_state,
        )

        model = create_model(model_type=model_type, n_components=n_components)
        model.fit(x, y)
        result = model.training_result(x, y)

        metadata = {
            "model_name": (model_name or "").strip() or "Untitled model",
            "feature_count": dataset.feature_count,
            "sample_count": dataset.sample_count,
            "source_format": dataset.source_format,
            "params": model.get_params(),
            "validation_config": {
                "enabled": bool(do_validation),
                "test_size": float(max(0.1, min(0.5, test_size))),
                "random_state": int(random_state),
            },
        }

        saved = self.model_manager.save(model_type=model.model_type, model_obj=model, metadata=metadata)
        plot = self._build_plot(model.model_type, result, phase="training")

        return {
            "saved_model": saved,
            "result": result,
            "validation": validation,
            "plot": plot,
        }

    def _run_validation(
        self,
        model_type: str,
        x: np.ndarray,
        y: Optional[np.ndarray],
        n_components: Optional[int],
        do_validation: bool,
        test_size: float,
        random_state: int,
    ) -> Optional[Dict[str, Any]]:
        if not do_validation:
            return {"status": "skipped", "reason": "Validation disabled by user"}

        if model_type not in {"pls", "plsda", "mcr_als", "pca"}:
            return {"status": "skipped", "reason": "Unsupported model type for validation"}

        if model_type in {"pca", "mcr_als"}:
            return {
                "status": "skipped",
                "reason": f"{model_type.upper()} does not use supervised holdout metrics",
                "limitations": [
                    "Выбранный метод не является supervised-моделью для прогноза класса/концентрации.",
                    "Для надежной количественной валидации используйте PLS (регрессия) или PLS-DA (классификация).",
                ],
            }

        if y is None:
            return {
                "status": "skipped",
                "reason": "Target values are required for supervised validation",
                "limitations": [
                    "Для supervised-валидации необходимы истинные target-значения.",
                    "Без target можно выполнить только разведочный анализ.",
                ],
            }

        if len(x) < 6:
            return {
                "status": "skipped",
                "reason": "Need at least 6 spectra for stable train/test validation",
                "limitations": [
                    "Слишком мало спектров для устойчивой оценки качества модели.",
                    "Рекомендуется расширить выборку минимум до 20-30 образцов.",
                ],
            }

        safe_test_size = float(max(0.1, min(0.5, test_size)))
        stratify = None
        warnings = []

        if model_type == "plsda":
            y_labels = np.asarray(y, dtype=str)
            unique, counts = np.unique(y_labels, return_counts=True)
            if len(unique) < 2:
                return {
                    "status": "skipped",
                    "reason": "Need at least 2 classes for classification validation",
                    "limitations": [
                        "В датасете только один класс, классификационная задача не определена.",
                    ],
                }
            if np.min(counts) >= 2:
                stratify = y_labels
            else:
                warnings.append("Not enough samples in at least one class for stratified split")

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=safe_test_size,
            random_state=int(random_state),
            stratify=stratify,
        )

        eval_model = create_model(model_type=model_type, n_components=n_components)
        eval_model.fit(x_train, y_train)

        if model_type == "pls":
            y_true = np.asarray(y_test, dtype=float)
            y_pred = np.asarray(eval_model.predict(x_test), dtype=float)
            validation_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
            kfold = self._kfold_validation(
                model_type=model_type,
                x=x,
                y=y,
                n_components=n_components,
                random_state=random_state,
            )
            permutation = self._permutation_test(
                model_type=model_type,
                x=x,
                y=y,
                n_components=n_components,
                random_state=random_state,
            )
            bootstrap_ci = self._bootstrap_ci_regression(y_true=y_true, y_pred=y_pred, random_state=random_state)
            return {
                "status": "ok",
                "mode": "holdout_regression",
                "train_samples": int(len(x_train)),
                "test_samples": int(len(x_test)),
                "metrics": validation_metrics,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "kfold": kfold,
                "permutation_test": permutation,
                "bootstrap_ci": bootstrap_ci,
                "limitations": self._build_limitations(
                    model_type=model_type,
                    sample_count=len(x),
                    class_labels=None,
                ),
                "warnings": warnings,
            }

        y_true_cls = np.asarray(y_test, dtype=str)
        y_pred_cls = np.asarray(eval_model.predict(x_test), dtype=str)
        labels = sorted(np.unique(np.concatenate([y_true_cls, y_pred_cls])).tolist())
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
        validation_metrics = {
            "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
            "precision_macro": float(precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)),
        }
        kfold = self._kfold_validation(
            model_type=model_type,
            x=x,
            y=y,
            n_components=n_components,
            random_state=random_state,
        )
        permutation = self._permutation_test(
            model_type=model_type,
            x=x,
            y=y,
            n_components=n_components,
            random_state=random_state,
        )
        bootstrap_ci = self._bootstrap_ci_classification(y_true=y_true_cls, y_pred=y_pred_cls, random_state=random_state)
        return {
            "status": "ok",
            "mode": "holdout_classification",
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "metrics": validation_metrics,
            "classes": labels,
            "confusion_matrix": cm.tolist(),
            "y_true": y_true_cls.tolist(),
            "y_pred": y_pred_cls.tolist(),
            "kfold": kfold,
            "permutation_test": permutation,
            "bootstrap_ci": bootstrap_ci,
            "limitations": self._build_limitations(
                model_type=model_type,
                sample_count=len(x),
                class_labels=np.asarray(y, dtype=str),
            ),
            "warnings": warnings,
        }

    def _kfold_validation(
        self,
        model_type: str,
        x: np.ndarray,
        y: np.ndarray,
        n_components: Optional[int],
        random_state: int,
    ) -> Dict[str, Any]:
        n_samples = len(x)
        if n_samples < 4:
            return {"status": "skipped", "reason": "Too few samples for k-fold"}

        metrics_per_fold: List[Dict[str, float]] = []
        if model_type == "plsda":
            y_cls = np.asarray(y, dtype=str)
            _, counts = np.unique(y_cls, return_counts=True)
            n_splits = min(5, int(np.min(counts)))
            if n_splits < 2:
                return {"status": "skipped", "reason": "Not enough samples per class for stratified k-fold"}
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
            split_iter = splitter.split(x, y_cls)
        else:
            n_splits = min(5, n_samples)
            if n_splits < 2:
                return {"status": "skipped", "reason": "Not enough samples for k-fold"}
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
            split_iter = splitter.split(x)

        for train_idx, test_idx in split_iter:
            model = create_model(model_type=model_type, n_components=n_components)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            if model_type == "pls":
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
        true_cv = self._kfold_validation(model_type=model_type, x=x, y=y, n_components=n_components, random_state=random_state)
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
        n_bootstrap: int = 500,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(random_state))
        n = len(y_true)
        if n < 3:
            return {"status": "skipped", "reason": "Too few samples for bootstrap CI"}

        acc_scores: List[float] = []
        f1_scores: List[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            acc_scores.append(float(accuracy_score(y_t, y_p)))
            f1_scores.append(float(f1_score(y_t, y_p, average="macro", zero_division=0)))

        return {
            "status": "ok",
            "n_bootstrap": n_bootstrap,
            "accuracy_95ci": [float(np.percentile(acc_scores, 2.5)), float(np.percentile(acc_scores, 97.5))],
            "f1_macro_95ci": [float(np.percentile(f1_scores, 2.5)), float(np.percentile(f1_scores, 97.5))],
        }

    def _bootstrap_ci_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        random_state: int,
        n_bootstrap: int = 500,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(random_state))
        n = len(y_true)
        if n < 3:
            return {"status": "skipped", "reason": "Too few samples for bootstrap CI"}

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

        return {
            "status": "ok",
            "n_bootstrap": n_bootstrap,
            "rmse_95ci": [float(np.percentile(rmse_scores, 2.5)), float(np.percentile(rmse_scores, 97.5))],
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
        limitations.append("Метрики не являются медицинским диагнозом и требуют экспертной интерпретации.")
        return limitations

    def infer(self, model_type: str, model_id: str, dataset: SpectrumDataset) -> Dict[str, Any]:
        model, metadata = self.model_manager.load(model_type=model_type, model_id=model_id)
        result = model.inference_result(dataset.x)
        plot = self._build_plot(metadata.get("model_type", model_type), result, phase="inference")
        return {
            "model_metadata": metadata,
            "result": result,
            "plot": plot,
        }

    def infer_many(
        self,
        model_type: str,
        model_id: str,
        datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        model, metadata = self.model_manager.load(model_type=model_type, model_id=model_id)
        all_results: List[Dict[str, Any]] = []
        all_predicted_classes: List[str] = []
        all_predictions: List[float] = []

        for item in datasets:
            filename = item["filename"]
            dataset: SpectrumDataset = item["dataset"]
            result = model.inference_result(dataset.x)
            row = {"filename": filename, "result": result}
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
        return {
            "model_metadata": metadata,
            "batch_results": all_results,
            "aggregate_result": aggregate_result,
            "plot": plot,
        }

    def _build_plot(self, model_type: str, result: Dict[str, Any], phase: str) -> Dict[str, Any]:
        normalized = model_type.lower()

        if normalized == "pca":
            scores = np.asarray(result.get("scores", []), dtype=float)
            if scores.size == 0:
                return {}
            return pca_plot(scores, "PCA: распределение в пространстве компонент")

        if normalized == "pls":
            if phase == "training" and "y_true" in result and "y_pred" in result:
                y_true = np.asarray(result["y_true"], dtype=float)
                y_pred = np.asarray(result["y_pred"], dtype=float)
                return pls_regression_plot(y_true, y_pred, "PLS: предсказание vs истина")
            if phase == "inference" and "predictions" in result:
                preds = [float(v) for v in result["predictions"]]
                return residual_plot(preds, "PLS: значения предсказаний")
            return {}

        if normalized == "plsda":
            labels = result.get("y_pred") if phase == "training" else result.get("predicted_classes")
            if not labels:
                return {}
            return class_count_plot(labels, "PLS-DA: распределение предсказанных классов")

        if normalized == "mcr_als":
            residuals = result.get("mean_absolute_residual")
            if residuals:
                return residual_plot([float(v) for v in residuals], "MCR-ALS stub: residuals")
            return {}

        return {}
