"""Real GridSearchCV-based hyperparameter search for the /analysis/train pipeline.

Used by AnalysisService.train_and_save: search runs ONLY on the train split
(never on the full dataset or on the held-out test part), so the resulting
best_params and best_score are not leaked from validation/test data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

SEARCHABLE_MODEL_TYPES = {"plsda", "svm", "random_forest", "decision_tree", "pls", "svr"}

_RMSE_SCORER = make_scorer(lambda y_true, y_pred: -float(np.sqrt(mean_squared_error(y_true, y_pred))))


class _PLSDASearchEstimator(BaseEstimator, ClassifierMixin):
    """sklearn-compatible PLS-DA wrapper used only for GridSearchCV scoring.

    Mirrors services.analysis.models.PLSDAClassifier numerically; kept separate
    because BaseAnalysisModel subclasses are not sklearn-clone-compatible.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: Sequence[Any]) -> "_PLSDASearchEstimator":
        self.scaler_ = StandardScaler()
        self.encoder_ = LabelEncoder()
        labels = np.asarray(y, dtype=str)
        encoded = self.encoder_.fit_transform(labels)
        class_count = len(self.encoder_.classes_)
        max_components = max(1, min(int(self.n_components), X.shape[0] - 1, X.shape[1], class_count))
        self.model_ = PLSRegression(n_components=max_components)
        x_scaled = self.scaler_.fit_transform(X)
        self.model_.fit(x_scaled, np.eye(class_count)[encoded])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.model_.predict(self.scaler_.transform(X))
        return self.encoder_.inverse_transform(np.argmax(scores, axis=1))


def _clean_best_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
    return {key.split("__")[-1]: value for key, value in best_params.items()}


def _n_splits_for_classification(y: np.ndarray, requested: int = 5) -> int:
    _, counts = np.unique(np.asarray(y, dtype=str), return_counts=True)
    return min(requested, int(np.min(counts)))


def _n_splits_for_regression(n_samples: int, requested: int = 5) -> int:
    return min(requested, n_samples)


def _build_search(model_type: str, x_train: np.ndarray, y_train: np.ndarray, base_params: Dict[str, Any], random_state: int, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    n_samples, n_features = x_train.shape
    class_weight = base_params.get("class_weight")

    if model_type == "plsda":
        class_count = len(np.unique(np.asarray(y_train, dtype=str)))
        n_splits = _n_splits_for_classification(y_train)
        if n_splits < 2:
            return None
        max_components = max(1, min(n_samples - 1, n_features, class_count))
        grid = {"n_components": [v for v in [2, 3, 5, 10] if v <= max_components] or [1]}
        return {
            "estimator": _PLSDASearchEstimator(),
            "param_grid": grid,
            "cv": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)),
            "scoring": "f1_macro",
            "n_splits": n_splits,
        }

    if model_type == "svm":
        n_splits = _n_splits_for_classification(y_train)
        if n_splits < 2:
            return None
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight=class_weight, random_state=int(random_state))),
        ])
        if quick_mode:
            grid = {"svc__C": [1, 10], "svc__kernel": ["rbf"], "svc__gamma": ["scale"]}
        else:
            grid = {"svc__C": [0.1, 1, 10, 100], "svc__kernel": ["linear", "rbf"], "svc__gamma": ["scale", "auto"]}
        return {"estimator": estimator, "param_grid": grid, "cv": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), "scoring": "f1_macro", "n_splits": n_splits}

    if model_type == "random_forest":
        n_splits = _n_splits_for_classification(y_train)
        if n_splits < 2:
            return None
        estimator = RandomForestClassifier(random_state=int(random_state), class_weight=class_weight)
        if quick_mode:
            grid = {"n_estimators": [100], "max_depth": [None, 5], "min_samples_split": [2]}
        else:
            grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5]}
        return {"estimator": estimator, "param_grid": grid, "cv": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), "scoring": "f1_macro", "n_splits": n_splits}

    if model_type == "decision_tree":
        n_splits = _n_splits_for_classification(y_train)
        if n_splits < 2:
            return None
        estimator = DecisionTreeClassifier(random_state=int(random_state), class_weight=class_weight)
        grid = {"max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5]}
        return {"estimator": estimator, "param_grid": grid, "cv": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), "scoring": "f1_macro", "n_splits": n_splits}

    if model_type == "pls":
        n_splits = _n_splits_for_regression(n_samples)
        if n_splits < 2:
            return None
        max_components = max(1, min(n_samples - 1, n_features))
        estimator = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
        grid = {"pls__n_components": [v for v in [2, 3, 5, 10] if v <= max_components] or [1]}
        return {
            "estimator": estimator,
            "param_grid": grid,
            "cv": KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)),
            "scoring": {"rmse": _RMSE_SCORER, "r2": "r2"},
            "refit": "rmse",
            "n_splits": n_splits,
        }

    if model_type == "svr":
        n_splits = _n_splits_for_regression(n_samples)
        if n_splits < 2:
            return None
        estimator = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
        if quick_mode:
            grid = {"svr__C": [1, 10], "svr__gamma": ["scale"], "svr__epsilon": [0.1], "svr__kernel": ["rbf"]}
        else:
            grid = {"svr__C": [0.1, 1, 10, 100], "svr__gamma": ["scale", "auto"], "svr__epsilon": [0.01, 0.1, 0.2], "svr__kernel": ["rbf", "linear"]}
        return {
            "estimator": estimator,
            "param_grid": grid,
            "cv": KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)),
            "scoring": {"rmse": _RMSE_SCORER, "r2": "r2"},
            "refit": "rmse",
            "n_splits": n_splits,
        }

    return None


def run_hyperparameter_search(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_params: Dict[str, Any],
    random_state: int,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """Run GridSearchCV on the train split only. Returns a JSON-serializable report."""
    if model_type not in SEARCHABLE_MODEL_TYPES:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"hyperparameter_search is only supported for {sorted(SEARCHABLE_MODEL_TYPES)}",
            "best_params_source": "manual_or_default",
        }

    spec = _build_search(model_type, x_train, y_train, base_params, random_state, quick_mode=quick_mode)
    if spec is None:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "Not enough samples per class/fold for GridSearchCV on the train split",
            "best_params_source": "manual_or_default",
        }

    search_kwargs: Dict[str, Any] = {
        "estimator": spec["estimator"],
        "param_grid": spec["param_grid"],
        "cv": spec["cv"],
        "scoring": spec["scoring"],
        "error_score": np.nan,
    }
    if "refit" in spec:
        search_kwargs["refit"] = spec["refit"]
    search = GridSearchCV(**search_kwargs)
    search.fit(x_train, y_train)

    best_params = _clean_best_params(search.best_params_)
    best_score = float(search.best_score_)
    extra: Dict[str, Any] = {}
    if isinstance(spec["scoring"], dict):
        scoring_name = spec.get("refit", "rmse")
        r2_key = "mean_test_r2"
        if r2_key in search.cv_results_:
            extra["r2_at_best"] = float(search.cv_results_[r2_key][search.best_index_])
        score_name = f"neg_{scoring_name}" if scoring_name == "rmse" else scoring_name
    else:
        score_name = spec["scoring"]

    return {
        "enabled": True,
        "status": "ok",
        "method": "GridSearchCV",
        "cv": spec["n_splits"],
        "scoring": score_name,
        "param_grid": {key.split("__")[-1]: value for key, value in spec["param_grid"].items()},
        "best_params": best_params,
        "best_score": best_score,
        "best_params_source": "grid_search",
        "quick_mode": bool(quick_mode),
        **extra,
    }
