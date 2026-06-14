from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiment_utils import (
    classification_metric_row,
    classification_models,
    fail_missing_input,
    read_table,
    safe_train_test_split,
    save_csv,
    simple_line_png,
    split_features_target,
)


def moving_average(X: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return X.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.vstack([np.convolve(np.pad(row, (window // 2, window - 1 - window // 2), mode="edge"), kernel, mode="valid") for row in X])


def downsample(X: np.ndarray, factor: int) -> np.ndarray:
    return X[:, :: max(1, int(factor))]


def transform_resolution(X: np.ndarray, mode: str, value: int | float) -> np.ndarray:
    # These are computational simulations of resolution loss, not real spectrometer settings.
    if mode == "gaussian_sigma":
        return gaussian_filter1d(X, sigma=float(value), axis=1) if float(value) > 0 else X.copy()
    if mode == "moving_average_window":
        return moving_average(X, int(value))
    if mode == "downsampling_factor":
        return downsample(X, int(value))
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classification robustness experiment under simulated spectral resolution loss.")
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
        raise SystemExit(f"Target column '{args.target}' was not found.")
    y = np.asarray(y, dtype=str)

    modes = [("gaussian_sigma", v) for v in [0, 1, 2, 3, 5]]
    modes += [("moving_average_window", v) for v in [1, 3, 5, 9]]
    modes += [("downsampling_factor", v) for v in [1, 2, 4]]

    rows = []
    for mode, value in modes:
        X_changed = transform_resolution(X, mode, value)
        X_train, X_test, y_train, y_test = safe_train_test_split(X_changed, y, random_state=42)
        for model_name, model in classification_models(random_state=42, include_tree=False).items():
            warning = ""
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                metrics = classification_metric_row(y_test, pred)
            except Exception as exc:
                metrics = {"accuracy": None, "precision_macro": None, "recall_macro": None, "f1_macro": None}
                warning = str(exc)
            rows.append(
                {
                    "resolution_mode": mode,
                    "simulation_parameter": value,
                    "model": model_name,
                    **metrics,
                    "warning": warning,
                }
            )

    save_csv(rows, output / "resolution_robustness.csv")
    series = []
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name and row["f1_macro"] is not None]
        series.append({"label": model_name, "x": list(range(len(model_rows))), "y": [row["f1_macro"] for row in model_rows]})
    simple_line_png(output / "resolution_robustness_f1.png", series)
    print(f"Saved: {output / 'resolution_robustness.csv'}")
    print(f"Saved: {output / 'resolution_robustness_f1.png'}")


if __name__ == "__main__":
    main()
