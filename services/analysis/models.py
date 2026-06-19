from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from services.analysis.model_base import BaseAnalysisModel


@dataclass
class ModelConfig:
    n_components: Optional[int] = None
    n_clusters: Optional[int] = None
    random_state: int = 42
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    epsilon: float = 0.1
    n_estimators: int = 200
    class_weight: Optional[str] = None
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    linkage: str = "ward"


def _safe_components(n_components: Optional[int], x: np.ndarray, cap: Optional[int] = None) -> int:
    max_components = min(x.shape[0], x.shape[1])
    if cap is not None:
        max_components = min(max_components, cap)
    if max_components < 1:
        return 1
    if n_components is None:
        return min(2, max_components)
    return max(1, min(int(n_components), max_components))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = sorted(np.unique(np.concatenate([y_true.astype(str), y_pred.astype(str)])).tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classes": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _safe_clusters(configured: Optional[int], x: np.ndarray, y: Optional[np.ndarray]) -> int:
    if configured:
        return max(2, min(int(configured), int(x.shape[0])))
    if y is not None:
        unique = len(np.unique(np.asarray(y, dtype=str)))
        if unique >= 2:
            return min(unique, int(x.shape[0]))
    return min(2, int(x.shape[0]))


def _cluster_quality_metrics(x_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    unique_labels = np.unique(labels)
    if x_scaled.shape[0] < 3 or len(unique_labels) < 2 or len(unique_labels) >= x_scaled.shape[0]:
        return {
            "silhouette_score": None,
            "warnings": [
                "Silhouette score не рассчитан: нужно минимум 2 кластера и хотя бы один кластер с более чем одним объектом."
            ],
        }
    try:
        return {"silhouette_score": float(silhouette_score(x_scaled, labels)), "warnings": []}
    except Exception as exc:
        return {"silhouette_score": None, "warnings": [f"Silhouette score не рассчитан: {exc}"]}


def _require_target(y: Optional[np.ndarray], message: str) -> np.ndarray:
    if y is None:
        raise ValueError(message)
    y_arr = np.asarray(y)
    if y_arr.shape[0] == 0:
        raise ValueError(message)
    return y_arr


def _require_classes(y: Optional[np.ndarray], method_name: str) -> np.ndarray:
    y_arr = _require_target(y, f"Для {method_name} нужен категориальный target.")
    labels = y_arr.astype(str)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Для {method_name} target должен содержать минимум два класса.")
    return labels


def _require_numeric_target(y: Optional[np.ndarray], method_name: str) -> np.ndarray:
    y_arr = _require_target(y, f"Для {method_name} нужен числовой target, например концентрация.")
    try:
        return y_arr.astype(float).reshape(-1)
    except ValueError as exc:
        raise ValueError(f"Для {method_name} target должен быть числовым.") from exc


class PCAModel(BaseAnalysisModel):
    model_type = "pca"
    supervised = False

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model: Optional[PCA] = None

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        x_scaled = self.scaler.fit_transform(x)
        components = _safe_components(self.config.n_components, x)
        self.model = PCA(n_components=components)
        self.model.fit(x_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PCA model is not fitted")
        x_scaled = self.scaler.transform(x)
        return self.model.transform(x_scaled)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        scores = self.predict(x)
        assert self.model is not None
        return {
            "scores": scores.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "components": int(self.model.n_components_),
        }

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        scores = self.predict(x)
        return {
            "scores": scores.tolist(),
            "components": scores.shape[1],
        }

    def get_params(self) -> Dict[str, Any]:
        return {"n_components": self.config.n_components}


class PLSRegressionModel(BaseAnalysisModel):
    model_type = "pls"
    supervised = True

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model: Optional[PLSRegression] = None

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is None:
            raise ValueError("PLS regression requires target values")
        y_numeric = np.asarray(y, dtype=float).reshape(-1, 1)
        x_scaled = self.scaler.fit_transform(x)
        components = _safe_components(self.config.n_components, x, cap=x.shape[0] - 1)
        self.model = PLSRegression(n_components=components)
        self.model.fit(x_scaled, y_numeric)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PLS model is not fitted")
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled).reshape(-1)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_numeric = _require_numeric_target(y, "PLS-регрессии")
        predicted = self.predict(x)
        return {
            "y_true": y_numeric.tolist(),
            "y_pred": predicted.tolist(),
            "x_scores": self.model.x_scores_.tolist() if self.model is not None and hasattr(self.model, "x_scores_") else [],
            **_regression_metrics(y_numeric, predicted),
        }

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        prediction = self.predict(x)
        return {"predictions": prediction.tolist()}

    def get_params(self) -> Dict[str, Any]:
        return {"n_components": self.config.n_components}


class PLSDAClassifier(BaseAnalysisModel):
    model_type = "plsda"
    supervised = True

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model: Optional[PLSRegression] = None
        self.encoder = LabelEncoder()

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is None:
            raise ValueError("PLS-DA requires class labels")

        y_labels = np.asarray(y, dtype=str)
        y_encoded = self.encoder.fit_transform(y_labels)
        class_count = len(self.encoder.classes_)
        y_one_hot = np.eye(class_count)[y_encoded]

        x_scaled = self.scaler.fit_transform(x)
        components = _safe_components(self.config.n_components, x, cap=max(1, class_count))
        self.model = PLSRegression(n_components=components)
        self.model.fit(x_scaled, y_one_hot)

    def _predict_encoded(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PLS-DA model is not fitted")
        x_scaled = self.scaler.transform(x)
        y_scores = self.model.predict(x_scaled)
        return np.argmax(y_scores, axis=1)

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PLS-DA model is not fitted")
        x_scaled = self.scaler.transform(x)
        return np.asarray(self.model.predict(x_scaled), dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        encoded = self._predict_encoded(x)
        return self.encoder.inverse_transform(encoded)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_true = _require_classes(y, "PLS-DA")
        y_pred = self.predict(x)
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            **_classification_metrics(y_true, y_pred),
        }

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        scores = self.decision_scores(x)
        encoded = np.argmax(scores, axis=1)
        y_pred = self.encoder.inverse_transform(encoded)
        max_scores = np.max(scores, axis=1)
        score_table = [
            {str(cls): float(row[idx]) for idx, cls in enumerate(self.encoder.classes_)}
            for row in scores
        ]
        return {
            "predicted_classes": y_pred.tolist(),
            "classes": self.encoder.classes_.tolist(),
            "decision_scores": score_table,
            "decision_score": max_scores.astype(float).tolist(),
            "confidence": [f"score={float(value):.4g}; вероятность недоступна для PLS-DA" for value in max_scores],
            "confidence_kind": "decision_score",
            "probability_available": False,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "classes": self.encoder.classes_.tolist() if hasattr(self.encoder, "classes_") else [],
        }


class SklearnClassifierModel(BaseAnalysisModel):
    supervised = True
    estimator_name = "classifier"

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Any = None

    def _build_model(self) -> Any:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        labels = _require_classes(y, self.estimator_name)
        self.model = self._build_model()
        self.model.fit(x, labels)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError(f"{self.estimator_name} model is not fitted")
        return np.asarray(self.model.predict(x), dtype=str)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_true = _require_classes(y, self.estimator_name)
        y_pred = self.predict(x)
        result = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            **_classification_metrics(y_true, y_pred),
        }
        estimator = self.model
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        if hasattr(estimator, "feature_importances_"):
            result["feature_importances"] = np.asarray(estimator.feature_importances_, dtype=float).tolist()
        return result

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(x)
        classes = list(getattr(self.model, "classes_", []))
        estimator = self.model
        if isinstance(estimator, Pipeline):
            classes = list(getattr(estimator.steps[-1][1], "classes_", classes))
        result: Dict[str, Any] = {"predicted_classes": y_pred.tolist(), "classes": [str(v) for v in classes] or sorted(np.unique(y_pred).tolist())}
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(x), dtype=float)
            max_proba = np.max(proba, axis=1)
            result["probabilities"] = [
                {str(cls): float(row[idx]) for idx, cls in enumerate(result["classes"])}
                for row in proba
            ]
            result["confidence"] = max_proba.astype(float).tolist()
            result["confidence_kind"] = "probability"
            result["probability_available"] = True
        elif hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(x), dtype=float)
            score_values = scores if scores.ndim == 1 else np.max(scores, axis=1)
            result["decision_score"] = np.asarray(score_values, dtype=float).tolist()
            result["confidence"] = [f"score={float(value):.4g}; вероятность недоступна" for value in score_values]
            result["confidence_kind"] = "decision_score"
            result["probability_available"] = False
        else:
            result["confidence"] = ["вероятность недоступна"] * len(y_pred)
            result["confidence_kind"] = "unavailable"
            result["probability_available"] = False
        return result

    def get_params(self) -> Dict[str, Any]:
        return dict(self.config.__dict__)


class SVMClassifier(SklearnClassifierModel):
    model_type = "svm"
    estimator_name = "SVM"

    def _build_model(self) -> Any:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel=self.config.kernel,
                        C=float(self.config.C),
                        gamma=self.config.gamma,
                        class_weight=self.config.class_weight,
                        random_state=int(self.config.random_state),
                    ),
                ),
            ]
        )


class RandomForestClassifierModel(SklearnClassifierModel):
    model_type = "random_forest"
    estimator_name = "Random Forest"

    def _build_model(self) -> Any:
        return RandomForestClassifier(
            n_estimators=int(self.config.n_estimators),
            random_state=int(self.config.random_state),
            class_weight=self.config.class_weight,
            min_samples_split=int(self.config.min_samples_split or 2),
        )


class DecisionTreeClassifierModel(SklearnClassifierModel):
    model_type = "decision_tree"
    estimator_name = "Decision Tree"

    def _build_model(self) -> Any:
        return DecisionTreeClassifier(
            random_state=int(self.config.random_state),
            max_depth=self.config.max_depth,
            class_weight=self.config.class_weight,
            min_samples_split=int(self.config.min_samples_split or 2),
        )


class KMeansClusterModel(BaseAnalysisModel):
    model_type = "kmeans"
    supervised = False

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        x_scaled = self.scaler.fit_transform(x)
        n_clusters = _safe_clusters(self.config.n_clusters, x, y)
        self.model = KMeans(n_clusters=n_clusters, random_state=int(self.config.random_state), n_init=10)
        self.model.fit(x_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("k-means model is not fitted")
        return self.model.predict(self.scaler.transform(x))

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        labels = self.predict(x)
        assert self.model is not None
        quality = _cluster_quality_metrics(self.scaler.transform(x), labels)
        result: Dict[str, Any] = {
            "cluster_labels": labels.astype(int).tolist(),
            "n_clusters": int(self.model.n_clusters),
            "inertia": float(self.model.inertia_),
            "silhouette_score": quality["silhouette_score"],
            "cluster_distribution": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
        if quality["warnings"]:
            result["warnings"] = quality["warnings"]
        if y is not None and len(np.unique(np.asarray(y, dtype=str))) >= 2:
            result["adjusted_rand_score"] = float(adjusted_rand_score(np.asarray(y, dtype=str), labels))
            result["normalized_mutual_info_score"] = float(normalized_mutual_info_score(np.asarray(y, dtype=str), labels))
        return result

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        labels = self.predict(x)
        return {"cluster_labels": labels.astype(int).tolist()}

    def get_params(self) -> Dict[str, Any]:
        return dict(self.config.__dict__)


class HCAClusterModel(BaseAnalysisModel):
    model_type = "hca"
    supervised = False

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model: Optional[AgglomerativeClustering] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        x_scaled = self.scaler.fit_transform(x)
        n_clusters = _safe_clusters(self.config.n_clusters, x, y)
        linkage = self.config.linkage if self.config.linkage in {"ward", "complete", "average", "single"} else "ward"
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = self.model.fit_predict(x_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.labels_ is None:
            raise ValueError("HCA model is not fitted")
        if x.shape[0] == len(self.labels_):
            return self.labels_
        raise ValueError("HCA не поддерживает прямое предсказание для новых объектов в sklearn AgglomerativeClustering.")

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        labels = self.predict(x)
        quality = _cluster_quality_metrics(self.scaler.transform(x), labels)
        result: Dict[str, Any] = {
            "cluster_labels": labels.astype(int).tolist(),
            "n_clusters": int(len(np.unique(labels))),
            "silhouette_score": quality["silhouette_score"],
            "cluster_distribution": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "linkage": self.config.linkage,
        }
        if quality["warnings"]:
            result["warnings"] = quality["warnings"]
        if y is not None and len(np.unique(np.asarray(y, dtype=str))) >= 2:
            result["adjusted_rand_score"] = float(adjusted_rand_score(np.asarray(y, dtype=str), labels))
            result["normalized_mutual_info_score"] = float(normalized_mutual_info_score(np.asarray(y, dtype=str), labels))
        return result

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        raise ValueError(
            "This saved HCA result cannot be used for direct prediction on new spectra. "
            "Re-run clustering on the full dataset."
        )

    def get_params(self) -> Dict[str, Any]:
        params = dict(self.config.__dict__)
        params["can_infer_new_samples"] = False
        params["limitations"] = [
            "HCA does not support direct inference for new spectra because AgglomerativeClustering has no predict method."
        ]
        return params


class SVRRegressionModel(BaseAnalysisModel):
    model_type = "svr"
    supervised = True

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svr",
                    SVR(
                        kernel=self.config.kernel,
                        C=float(self.config.C),
                        epsilon=float(self.config.epsilon),
                        gamma=self.config.gamma,
                    ),
                ),
            ]
        )

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        y_numeric = _require_numeric_target(y, "SVR")
        self.model.fit(x, y_numeric)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x), dtype=float)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_true = _require_numeric_target(y, "SVR")
        y_pred = self.predict(x)
        return {"y_true": y_true.tolist(), "y_pred": y_pred.tolist(), **_regression_metrics(y_true, y_pred)}

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        return {"predictions": self.predict(x).tolist()}

    def get_params(self) -> Dict[str, Any]:
        return dict(self.config.__dict__)


def create_model(
    model_type: str,
    n_components: Optional[int] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> BaseAnalysisModel:
    params = dict(model_params or {})
    if n_components is not None:
        params["n_components"] = n_components
    config_fields = set(ModelConfig.__dataclass_fields__.keys())
    config_values = {key: value for key, value in params.items() if key in config_fields}
    config = ModelConfig(**config_values)
    normalized = model_type.strip().lower()
    registry = {
        "pca": PCAModel,
        "pls": PLSRegressionModel,
        "pls_regression": PLSRegressionModel,
        "pls-regression": PLSRegressionModel,
        "plsda": PLSDAClassifier,
        "pls-da": PLSDAClassifier,
        "svm": SVMClassifier,
        "svc": SVMClassifier,
        "random_forest": RandomForestClassifierModel,
        "random-forest": RandomForestClassifierModel,
        "rf": RandomForestClassifierModel,
        "decision_tree": DecisionTreeClassifierModel,
        "decision-tree": DecisionTreeClassifierModel,
        "dt": DecisionTreeClassifierModel,
        "kmeans": KMeansClusterModel,
        "k-means": KMeansClusterModel,
        "hca": HCAClusterModel,
        "hierarchical": HCAClusterModel,
        "svr": SVRRegressionModel,
    }
    model_cls = registry.get(normalized)
    if model_cls is None:
        if normalized in {"mcr_als", "mcr-als"}:
            raise ValueError("MCR-ALS is not implemented in current version")
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Неизвестный тип модели '{model_type}'. Поддерживается: {supported}")
    return model_cls(config=config)


def supported_model_types() -> List[str]:
    return ["pca", "pls", "plsda", "svm", "random_forest", "decision_tree", "kmeans", "hca", "svr"]
