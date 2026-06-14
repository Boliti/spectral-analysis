from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiment_utils import WORKING_DATASET_MESSAGE, classification_models, fail_missing_input, read_table, safe_train_test_split, save_csv, split_features_target


def measure(operation: str, model: str, dataset_name: str, n_samples: int, n_features: int, fn: Callable[[], Any]) -> Dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    warning = ""
    result = None
    try:
        result = fn()
    except Exception as exc:
        warning = str(exc)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "operation": operation,
        "model": model,
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "time_seconds": elapsed,
        "inference_ms_per_spectrum": None,
        "peak_memory_mb": peak / (1024 * 1024),
        "warning": warning,
        "_result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance and memory benchmark for the analysis module models.")
    parser.add_argument("--input", required=True, help="CSV/XLSX classification dataset.")
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--output", default="results/experiments", help="Output directory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    fail_missing_input(input_path, "a CSV/XLSX dataset with spectral feature columns")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    imported_df = None
    row = measure("import", "", input_path.name, 0, 0, lambda: read_table(input_path))
    imported_df = row.pop("_result")
    rows.append(row)
    if imported_df is None:
        save_csv(rows, output / "performance_benchmark.csv")
        raise SystemExit("Import failed; see performance_benchmark.csv.")

    X, y, _features = split_features_target(imported_df, args.target)
    n_samples, n_features = X.shape
    row = measure("preprocessing", "StandardScaler", input_path.name, n_samples, n_features, lambda: StandardScaler().fit_transform(X))
    X_processed = row.pop("_result") if row.get("_result") is not None else X
    rows.append(row)

    if y is None:
        rows.append(
            {
                "operation": "training",
                "model": "classification models",
                "dataset_name": input_path.name,
                "n_samples": n_samples,
                "n_features": n_features,
                "time_seconds": None,
                "inference_ms_per_spectrum": None,
                "peak_memory_mb": None,
                "warning": WORKING_DATASET_MESSAGE,
            }
        )
    else:
        y = np.asarray(y, dtype=str)
        X_train, X_test, y_train, _y_test = safe_train_test_split(X_processed, y, random_state=42)
        for model_name, model in classification_models(random_state=42, include_tree=True).items():
            row = measure("training", model_name, input_path.name, n_samples, n_features, lambda m=model: m.fit(X_train, y_train))
            trained_model = row.pop("_result")
            rows.append(row)
            if trained_model is not None:
                sample = X_test[:1]
                inf = measure("inference", model_name, input_path.name, n_samples, n_features, lambda m=trained_model: m.predict(sample))
                inf.pop("_result", None)
                inf["inference_ms_per_spectrum"] = inf["time_seconds"] * 1000.0
                rows.append(inf)

    save_csv(rows, output / "performance_benchmark.csv")
    print(f"Saved: {output / 'performance_benchmark.csv'}")


if __name__ == "__main__":
    main()
