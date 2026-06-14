from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, mean_squared_error, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier

from experiment_utils import PLSDAEstimator, fail_missing_input, param_grid, read_table, save_csv, split_features_target, stratify_if_possible


def infer_task(y: np.ndarray | None, requested: str) -> str:
    if requested != "auto":
        return requested
    if y is None:
        return "clustering"
    try:
        numeric = np.asarray(y, dtype=float)
        return "regression" if len(np.unique(numeric)) > max(10, int(len(numeric) * 0.1)) else "classification"
    except ValueError:
        return "classification"


def clean_params(params: dict) -> dict:
    return {key.split("__")[-1]: value for key, value in params.items()}


def run_grid(model_name: str, estimator, grid: dict, X: np.ndarray, y: np.ndarray, task: str, cv_folds: int) -> dict:
    warning = ""
    if task == "classification":
        scorer = "f1_macro"
        labels = np.asarray(y, dtype=str)
        _, counts = np.unique(labels, return_counts=True)
        folds = max(2, min(cv_folds, int(np.min(counts))))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        scorer = make_scorer(lambda yt, yp: -float(np.sqrt(mean_squared_error(yt, yp))))
        folds = max(2, min(cv_folds, len(X)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        search = GridSearchCV(estimator, grid, scoring=scorer, cv=cv, error_score=np.nan)
        search.fit(X, y)
        best_params = clean_params(search.best_params_)
        best_score = float(search.best_score_)
    except Exception as exc:
        best_params = {}
        best_score = None
        folds = 0
        warning = str(exc)
    return {
        "task_type": task,
        "model": model_name,
        "param_grid": json.dumps(grid, ensure_ascii=False),
        "best_params": json.dumps(best_params, ensure_ascii=False),
        "best_score": best_score,
        "score_name": "f1_macro" if task == "classification" else "neg_rmse",
        "cv_folds": folds,
        "warning": warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter selection for spectral analysis models.")
    parser.add_argument("--input", required=True, help="CSV/XLSX dataset.")
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--task", choices=["auto", "classification", "regression", "clustering"], default="auto")
    parser.add_argument("--output", default="results/experiments", help="Output directory.")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    fail_missing_input(input_path, "a CSV/XLSX dataset for classification, regression or clustering")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    X, y, _features = split_features_target(df, args.target)
    task = infer_task(y, args.task)
    rows = []

    if task == "classification":
        if y is None:
            raise SystemExit(f"Target column '{args.target}' is required for classification.")
        y = np.asarray(y, dtype=str)
        max_components = list(range(1, min(10, X.shape[0] - 1, X.shape[1], len(np.unique(y))) + 1))
        rows.append(run_grid("PLS-DA", PLSDAEstimator(), {"n_components": max_components}, X, y, task, args.cv_folds))
        rows.append(run_grid("SVM", Pipeline([("scaler", StandardScaler()), ("svc", SVC())]), {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"]}, X, y, task, args.cv_folds))
        rows.append(run_grid("Random Forest", RandomForestClassifier(random_state=42), {"n_estimators": [100, 200, 500], "max_depth": [None, 5, 10]}, X, y, task, args.cv_folds))
        rows.append(run_grid("Decision Tree", DecisionTreeClassifier(random_state=42), {"max_depth": [None, 3, 5, 10]}, X, y, task, args.cv_folds))
    elif task == "regression":
        if y is None:
            raise SystemExit(f"Target column '{args.target}' is required for regression.")
        y_num = np.asarray(y, dtype=float)
        max_components = list(range(1, min(10, X.shape[0] - 1, X.shape[1]) + 1))
        rows.append(run_grid("PLS Regression", Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())]), {"pls__n_components": max_components}, X, y_num, task, args.cv_folds))
        rows.append(run_grid("SVR", Pipeline([("scaler", StandardScaler()), ("svr", SVR())]), {"svr__C": [0.1, 1, 10], "svr__epsilon": [0.01, 0.1, 0.5], "svr__kernel": ["linear", "rbf"]}, X, y_num, task, args.cv_folds))
    else:
        X_scaled = StandardScaler().fit_transform(X)
        for params in param_grid({"n_clusters": list(range(2, min(6, len(X) - 1) + 1))}):
            warning = ""
            score = None
            try:
                labels = KMeans(n_clusters=params["n_clusters"], random_state=42, n_init=10).fit_predict(X_scaled)
                if 1 < len(np.unique(labels)) < len(labels):
                    score = float(silhouette_score(X_scaled, labels))
                else:
                    warning = "Silhouette score is undefined for this clustering result."
            except Exception as exc:
                warning = str(exc)
            rows.append(
                {
                    "task_type": "clustering",
                    "model": "k-means",
                    "param_grid": json.dumps({"n_clusters": "2...6"}, ensure_ascii=False),
                    "best_params": json.dumps(params, ensure_ascii=False),
                    "best_score": score,
                    "score_name": "silhouette_score",
                    "cv_folds": 0,
                    "warning": warning,
                }
            )
        valid = [row for row in rows if row["best_score"] is not None]
        if valid:
            best = max(valid, key=lambda row: row["best_score"])
            rows = [{**row, "warning": (row["warning"] + ("; " if row["warning"] else "") + ("best" if row is best else "")).strip("; ")} for row in rows]

    save_csv(rows, output / "hyperparameter_selection.csv")
    print(f"Saved: {output / 'hyperparameter_selection.csv'}")


if __name__ == "__main__":
    main()
