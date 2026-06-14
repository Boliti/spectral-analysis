from __future__ import annotations

import itertools
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SERVICE_COLUMNS = {"sample_id", "sample", "id", "source_file", "source_sheet", "filename", "file_name"}


class PLSDAEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.model: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: Sequence[Any]) -> "PLSDAEstimator":
        labels = np.asarray(y, dtype=str)
        encoded = self.encoder.fit_transform(labels)
        class_count = len(self.encoder.classes_)
        max_components = max(1, min(int(self.n_components), X.shape[0] - 1, X.shape[1], class_count))
        self.model = PLSRegression(n_components=max_components)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, np.eye(class_count)[encoded])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("PLS-DA estimator is not fitted")
        scores = self.model.predict(self.scaler.transform(X))
        return self.encoder.inverse_transform(np.argmax(scores, axis=1))


def fail_missing_input(path: Path, description: str) -> None:
    if not path.exists():
        print(
            f"Input file or directory was not found: {path}\n"
            f"Provide {description} and rerun the command.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise SystemExit("Excel input requires openpyxl. Install project requirements with Excel support.") from exc
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported dataset format: {suffix}. Use CSV, XLSX or XLS.")


def split_features_target(df: pd.DataFrame, target: str = "target") -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    columns = [str(col).strip() for col in df.columns]
    df = df.copy()
    df.columns = columns
    y = df[target].to_numpy() if target in df.columns else None
    ignored = {target.lower()} | SERVICE_COLUMNS
    feature_cols: List[str] = []
    for col in df.columns:
        if col.lower() in ignored:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= max(1, int(len(df) * 0.8)):
            feature_cols.append(col)
    if not feature_cols:
        raise SystemExit("No numeric spectral feature columns were found.")
    X_frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_frame = X_frame.fillna(X_frame.median(numeric_only=True)).fillna(0.0)
    return X_frame.to_numpy(dtype=float), y, feature_cols


def stratify_if_possible(y: np.ndarray) -> Optional[np.ndarray]:
    labels, counts = np.unique(np.asarray(y, dtype=str), return_counts=True)
    if len(labels) >= 2 and np.min(counts) >= 2:
        return np.asarray(y, dtype=str)
    return None


def classification_models(random_state: int = 42, include_tree: bool = True) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "PLS-DA": PLSDAEstimator(n_components=2),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }
    if include_tree:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=random_state)
    return models


def classification_metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)])).tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": json.dumps(
            pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"), dropna=False)
            .reindex(index=labels, columns=labels, fill_value=0)
            .values.tolist(),
            ensure_ascii=False,
        ),
    }


def safe_train_test_split(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 4:
        raise SystemExit("At least 4 samples are required for a train/test experiment.")
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify_if_possible(y),
    )


def param_grid(items: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(items.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*[items[key] for key in keys])]


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _write_png_rgb(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    data += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def simple_line_png(path: Path, series: List[Dict[str, Any]], width: int = 900, height: int = 560) -> None:
    pixels = bytearray([255] * width * height * 3)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    margin = 56

    def put(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    put(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    for x in range(margin, width - margin):
        put(x, height - margin, (80, 80, 80))
    for y in range(margin, height - margin):
        put(margin, y, (80, 80, 80))

    points = []
    for item in series:
        xs = np.asarray(item.get("x", []), dtype=float)
        ys = np.asarray(item.get("y", []), dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.any(mask):
            points.append((xs[mask], ys[mask]))
    if not points:
        _write_png_rgb(path, width, height, pixels)
        return

    all_x = np.concatenate([p[0] for p in points])
    all_y = np.concatenate([p[1] for p in points])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def scale_x(v: float) -> int:
        return int(margin + (v - x_min) / (x_max - x_min) * (width - 2 * margin))

    def scale_y(v: float) -> int:
        return int(height - margin - (v - y_min) / (y_max - y_min) * (height - 2 * margin))

    for idx, (xs, ys) in enumerate(points):
        color = colors[idx % len(colors)]
        order = np.argsort(xs)
        px = [scale_x(float(v)) for v in xs[order]]
        py = [scale_y(float(v)) for v in ys[order]]
        for a, b in zip(range(len(px) - 1), range(1, len(px))):
            line(px[a], py[a], px[b], py[b], color)
        for x, y in zip(px, py):
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx * dx + dy * dy <= 9:
                        put(x + dx, y + dy, color)

    _write_png_rgb(path, width, height, pixels)
