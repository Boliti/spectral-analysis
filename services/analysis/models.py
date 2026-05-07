from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from services.analysis.model_base import BaseAnalysisModel


@dataclass
class ModelConfig:
    n_components: Optional[int] = None


def _safe_components(n_components: Optional[int], x: np.ndarray, cap: Optional[int] = None) -> int:
    max_components = min(x.shape[0], x.shape[1])
    if cap is not None:
        max_components = min(max_components, cap)
    if max_components < 1:
        return 1
    if n_components is None:
        return min(2, max_components)
    return max(1, min(int(n_components), max_components))


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
        if y is None:
            raise ValueError("PLS regression requires target values")
        y_numeric = np.asarray(y, dtype=float).reshape(-1)
        predicted = self.predict(x)
        rmse = float(np.sqrt(mean_squared_error(y_numeric, predicted)))
        return {
            "y_true": y_numeric.tolist(),
            "y_pred": predicted.tolist(),
            "r2": float(r2_score(y_numeric, predicted)),
            "rmse": rmse,
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        encoded = self._predict_encoded(x)
        return self.encoder.inverse_transform(encoded)

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if y is None:
            raise ValueError("PLS-DA requires class labels")
        y_true = np.asarray(y, dtype=str)
        y_pred = self.predict(x)
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classes": self.encoder.classes_.tolist(),
        }

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(x)
        return {
            "predicted_classes": y_pred.tolist(),
            "classes": self.encoder.classes_.tolist(),
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "classes": self.encoder.classes_.tolist() if hasattr(self.encoder, "classes_") else [],
        }


class MCRALSStubModel(BaseAnalysisModel):
    model_type = "mcr_als"
    supervised = False

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.mean_spectrum: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.mean_spectrum = np.mean(x, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_spectrum is None:
            raise ValueError("MCR-ALS stub model is not fitted")
        return np.tile(self.mean_spectrum, (x.shape[0], 1))

    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        reconstructed = self.predict(x)
        residual = np.mean(np.abs(x - reconstructed), axis=1)
        return {
            "message": "MCR-ALS реализован как расширяемый stub. Добавьте полноценный ALS solver в будущем.",
            "mean_absolute_residual": residual.tolist(),
        }

    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        reconstructed = self.predict(x)
        return {
            "message": "MCR-ALS stub inference выполнен.",
            "reconstructed_preview": reconstructed[: min(3, len(reconstructed))].tolist(),
        }

    def get_params(self) -> Dict[str, Any]:
        return {"stub": True, "n_components": self.config.n_components}


def create_model(model_type: str, n_components: Optional[int] = None) -> BaseAnalysisModel:
    config = ModelConfig(n_components=n_components)
    normalized = model_type.strip().lower()
    registry = {
        "pca": PCAModel,
        "pls": PLSRegressionModel,
        "plsda": PLSDAClassifier,
        "pls-da": PLSDAClassifier,
        "mcr_als": MCRALSStubModel,
        "mcr-als": MCRALSStubModel,
    }
    model_cls = registry.get(normalized)
    if model_cls is None:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Неизвестный тип модели '{model_type}'. Поддерживается: {supported}")
    return model_cls(config=config)


def supported_model_types() -> List[str]:
    return ["pca", "pls", "plsda", "mcr_als"]
