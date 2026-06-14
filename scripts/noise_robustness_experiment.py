from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiment_utils import (
    classification_metric_row,
    classification_models,
    fail_missing_input,
    read_table,
    safe_train_test_split,
    save_csv,
    simple_line_png,
    split_features_target,
    WORKING_DATASET_MESSAGE,
)


NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]


def add_noise(X: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    if level == 0:
        return X.copy()
    scale = np.std(X, axis=0, ddof=0)
    scale[scale == 0] = np.mean(scale[scale > 0]) if np.any(scale > 0) else 1.0
    return X + rng.normal(0.0, level * scale, size=X.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classification robustness experiment with additive Gaussian noise.")
    parser.add_argument("--input", required=True, help="CSV/XLSX dataset: rows are spectra, target column contains classes.")
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--output", default="results/experiments", help="Output directory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    fail_missing_input(input_path, "a classification CSV/XLSX dataset with a target column")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    X, y, _features = split_features_target(df, args.target)
    if y is None:
        raise SystemExit(WORKING_DATASET_MESSAGE)
    y = np.asarray(y, dtype=str)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, random_state=42)

    rows = []
    rng = np.random.default_rng(42)
    for level in NOISE_LEVELS:
        noisy_train = add_noise(X_train, level, rng)
        noisy_test = add_noise(X_test, level, rng)
        for model_name, model in classification_models(random_state=42, include_tree=True).items():
            warning = ""
            try:
                model.fit(noisy_train, y_train)
                pred = model.predict(noisy_test)
                metrics = classification_metric_row(y_test, pred)
            except Exception as exc:
                metrics = {"accuracy": None, "precision_macro": None, "recall_macro": None, "f1_macro": None, "confusion_matrix": "[]"}
                warning = str(exc)
            rows.append({"noise_level": level, "model": model_name, **metrics, "warning": warning})

    save_csv(rows, output / "noise_robustness.csv")
    series = []
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name and row["f1_macro"] is not None]
        series.append({"label": model_name, "x": [row["noise_level"] for row in model_rows], "y": [row["f1_macro"] for row in model_rows]})
    simple_line_png(output / "noise_robustness_f1.png", series)
    print(f"Saved: {output / 'noise_robustness.csv'}")
    print(f"Saved: {output / 'noise_robustness_f1.png'}")


if __name__ == "__main__":
    main()
