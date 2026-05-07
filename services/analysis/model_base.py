from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAnalysisModel(ABC):
    """Common interface for all spectral analysis models."""

    model_type: str
    supervised: bool = False

    @abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def training_result(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def inference_result(self, x: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError
