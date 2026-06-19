from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _scatter(x: List[float], y: List[float], text: List[str], name: str) -> Dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "markers",
        "x": x,
        "y": y,
        "text": text,
        "name": name,
        "marker": {"size": 10, "opacity": 0.85},
    }


def pca_plot(scores: np.ndarray, title: str) -> Dict[str, Any]:
    # If scores has two or more components, plot PC1 vs PC2 with axis labels.
    if scores.ndim == 2 and scores.shape[1] >= 2:
        x = scores[:, 0].tolist()
        y = scores[:, 1].tolist()
        text = [f"sample_{i + 1}" for i in range(scores.shape[0])]
        return {
            "data": [_scatter(x, y, text, "PCA scores")],
            "layout": {
                "title": title,
                "xaxis": {"title": "PC1"},
                "yaxis": {"title": "PC2"},
                "template": "plotly_white",
            },
        }
    # If only one component, produce a simple 1D scatter of PC1 across samples.
    if scores.ndim == 2 and scores.shape[1] == 1:
        pc1 = scores[:, 0].tolist()
        text = [f"sample_{i + 1}" for i in range(scores.shape[0])]
        return {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": list(range(1, len(pc1) + 1)),
                    "y": pc1,
                    "text": text,
                    "name": "PC1",
                    "marker": {"size": 10, "opacity": 0.85},
                }
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": "Sample index"},
                "yaxis": {"title": "PC1"},
                "template": "plotly_white",
            },
        }
    # Fallback empty plot
    return {"data": [], "layout": {"title": title, "template": "plotly_white"}}


def pls_regression_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Dict[str, Any]:
    lo = float(min(np.min(y_true), np.min(y_pred))) if len(y_true) else 0.0
    hi = float(max(np.max(y_true), np.max(y_pred))) if len(y_true) else 1.0
    return {
        "data": [
            _scatter(y_true.tolist(), y_pred.tolist(), [f"sample_{i + 1}" for i in range(len(y_true))], "Prediction"),
            {
                "type": "scatter",
                "mode": "lines",
                "x": [lo, hi],
                "y": [lo, hi],
                "name": "y=x",
                "line": {"dash": "dash", "color": "#64748b"},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "True y"},
            "yaxis": {"title": "Predicted y"},
            "template": "plotly_white",
        },
    }


def class_count_plot(labels: List[str], title: str) -> Dict[str, Any]:
    unique, counts = np.unique(np.array(labels, dtype=str), return_counts=True)
    return {
        "data": [
            {
                "type": "bar",
                "x": unique.tolist(),
                "y": counts.tolist(),
                "marker": {"opacity": 0.9},
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "Class"},
            "yaxis": {"title": "Count"},
            "template": "plotly_white",
        },
    }


def residual_plot(residuals: List[float], title: str) -> Dict[str, Any]:
    x = list(range(1, len(residuals) + 1))
    return {
        "data": [
            {
                "type": "bar",
                "x": x,
                "y": residuals,
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "Spectrum index"},
            "yaxis": {"title": "Residual"},
            "template": "plotly_white",
        },
    }
