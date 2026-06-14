from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from services.analysis.spectrum_loader import SpectrumDataset, SpectrumValidationError
from data_processing import (
    baseline_als,
    baseline_polyfit,
    baseline_rolling_ball,
    gaussian_smooth,
    normalize_max,
    normalize_snv,
    smooth_signal,
    whittaker_smooth,
)


NUMERIC_TARGET_COLUMN_HINTS = {"conc (y)", "total_protein (y)", "conc_total_protein (y)", "concentration", "target", "y"}
CLASS_TARGET_COLUMN_HINTS = {"conc (class)", "class", "group", "label", "diagnosis"}
TARGET_COLUMN_HINTS = NUMERIC_TARGET_COLUMN_HINTS | CLASS_TARGET_COLUMN_HINTS
ID_COLUMN_HINTS = {"id", "sample", "sample_id", "name", "filename"}
AXIS_COLUMN_HINTS = {"wavenumber", "wave", "#wave", "raman", "shift", "x", "cm-1", "cm^-1"}
ROW_METADATA_HINTS = {"date", "seria", "cuvetta"}


@dataclass
class UploadedFileData:
    filename: str
    content: bytes


@dataclass
class SpectralDataset:
    X: np.ndarray
    axis: np.ndarray
    sample_ids: List[str]
    y: Optional[List[str]]
    target_name: Optional[str]
    class_names: Optional[List[str]]
    metadata: Dict[str, Any]
    source_layout: str
    source_files: List[str]
    group_ids: Optional[List[str]] = None
    source_sheets: Optional[List[str]] = None
    source_items: Optional[List[str]] = None
    sample_metadata: Optional[List[Dict[str, Any]]] = None
    processed_X: Optional[np.ndarray] = None
    processed_axis: Optional[np.ndarray] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    preprocessing_warnings: List[str] = field(default_factory=list)

    @property
    def dataset_id(self) -> str:
        return str(self.metadata.get("dataset_id", ""))

    def _matrix_for_version(self, version: str = "raw") -> Tuple[np.ndarray, np.ndarray]:
        if version == "processed" and self.processed_X is not None and self.processed_axis is not None:
            return self.processed_X, self.processed_axis
        return self.X, self.axis

    def summary(self, version: str = "raw") -> Dict[str, Any]:
        X, axis = self._matrix_for_version(version)
        target_type = str(self.metadata.get("target_type") or _target_type(self.y))
        distribution = _target_distribution(self.y, target_type)
        target_stats = _target_stats(self.y, target_type)
        structure = analyze_dataset_structure(X, axis)
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.metadata.get("created_at"),
            "version": version,
            "has_processed": self.processed_X is not None,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "axis_min": float(np.min(axis)) if axis.size else None,
            "axis_max": float(np.max(axis)) if axis.size else None,
            "sample_ids_preview": self.sample_ids[:5],
            "target_name": self.target_name,
            "target_type": target_type,
            "target_range": target_stats.get("range"),
            "target_min": target_stats.get("min"),
            "target_max": target_stats.get("max"),
            "target_unique_count": target_stats.get("unique_count"),
            "target_distribution": distribution,
            "task_recommendation": "regression" if target_type == "numeric" else ("classification" if target_type == "categorical" else "exploratory"),
            "alternative_targets": self.metadata.get("alternative_targets") or [],
            "class_names": self.class_names,
            "class_distribution": distribution,
            "source_layout": self.source_layout,
            "source_files": self.source_files,
            "selected_sheets": self.metadata.get("selected_sheets"),
            "sheet_mode": self.metadata.get("sheet_mode"),
            "measurement_metadata": self.metadata.get("measurement_metadata") or {},
            "preprocessing_status": self.metadata.get("preprocessing_status", "not_applied"),
            "preprocessing": self.preprocessing_config if version == "processed" else None,
            "recommended_methods": suggest_analysis_methods_from_summary(
                n_samples=int(X.shape[0]),
                n_features=int(X.shape[1]),
                target_name=self.target_name,
                target_type=target_type,
                distribution=distribution,
                has_processed=self.processed_X is not None,
                duplicate_ids=bool(self.metadata.get("duplicate_ids")),
            ),
            "metadata": self.metadata,
            "quality": structure,
            "missing_fraction": structure["missing_fraction"],
            "nan_count": structure["nan_count"],
            "empty_spectra_count": structure["empty_spectra_count"],
            "empty_features_count": structure["empty_features_count"],
            "has_negative_intensity": structure["has_negative_intensity"],
            "snr": structure["snr"],
            "technical_status": structure["status"],
            "technical_warnings": structure["warnings"],
        }

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "X": self.X.tolist(),
            "axis": self.axis.tolist(),
            "sample_ids": self.sample_ids,
            "y": self.y,
            "target_name": self.target_name,
            "class_names": self.class_names,
            "metadata": self.metadata,
            "source_layout": self.source_layout,
            "source_files": self.source_files,
            "group_ids": self.group_ids,
        }

    def to_analysis_dataset(self, version: str = "raw") -> SpectrumDataset:
        X, axis = self._matrix_for_version(version)
        return SpectrumDataset(
            x=X,
            spectral_axis=axis,
            sample_names=self.sample_ids,
            source_format=self.source_layout,
            filename=";".join(self.source_files[:3]),
            file_format=self.source_layout,
            metadata={
                **dict(self.metadata),
                "sample_metadata": self.sample_metadata or [],
                "dataset_version": version,
                "preprocessing": self.preprocessing_config if version == "processed" else {},
            },
        )

    def import_config(self) -> Dict[str, Any]:
        config = self.metadata.get("import_config")
        return dict(config) if isinstance(config, dict) else {}

    def metadata_export(self, version: str = "raw") -> Dict[str, Any]:
        summary = self.summary(version=version)
        return {
            "dataset_id": summary.get("dataset_id"),
            "created_at": summary.get("created_at"),
            "user_id": self.metadata.get("user_id"),
            "username": self.metadata.get("username"),
            "n_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "axis_min": summary.get("axis_min"),
            "axis_max": summary.get("axis_max"),
            "target_name": summary.get("target_name"),
            "target_type": summary.get("target_type"),
            "class_distribution": summary.get("class_distribution"),
            "source_layout": summary.get("source_layout"),
            "source_files": summary.get("source_files"),
            "source_sheets": summary.get("selected_sheets"),
            "spectrometer_parameters": summary.get("measurement_metadata"),
            "detected_metadata": self.metadata.get("sample_metadata_preview") or [],
            "slit_width_um_values": self.metadata.get("slit_width_um_values") or [],
            "grating_lines_mm_values": self.metadata.get("grating_lines_mm_values") or [],
            "exposure_time_s_values": self.metadata.get("exposure_time_s_values") or [],
            "accumulations_values": self.metadata.get("accumulations_values") or [],
            "power_mw_values": self.metadata.get("power_mw_values") or [],
            "sample_name_values": self.metadata.get("sample_name_values") or [],
            "recommended_methods": summary.get("recommended_methods"),
            "task_recommendation": summary.get("task_recommendation"),
            "used_version": version,
        }

    def to_export_frame(self, version: str = "raw") -> pd.DataFrame:
        X, axis = self._matrix_for_version(version)
        axis_columns = [f"{float(value):.6g}" for value in axis]
        data: Dict[str, Any] = {"sample_id": self.sample_ids}
        if self.y is not None:
            data["target"] = self.y
        if self.sample_metadata:
            metadata_keys: List[str] = []
            for row in self.sample_metadata:
                for key in row.keys():
                    if key not in metadata_keys:
                        metadata_keys.append(key)
            for key in metadata_keys:
                if key not in {"sample_id", "target"}:
                    data[key] = [row.get(key) for row in self.sample_metadata]
        if self.source_sheets is not None and "source_sheet" not in data:
            data["source_sheet"] = self.source_sheets
        if self.source_items is not None and "source_file" not in data:
            data["source_file"] = self.source_items
        elif self.source_files:
            data["source_file"] = [self.source_files[0]] * len(self.sample_ids)
        frame = pd.DataFrame(data)
        spectra = pd.DataFrame(X, columns=axis_columns)
        return pd.concat([frame, spectra], axis=1)


class DatasetImportError(ValueError):
    pass


@dataclass
class DatasetStore:
    _items: Dict[str, SpectralDataset] = field(default_factory=dict)

    def put(self, dataset: SpectralDataset) -> str:
        dataset_id = uuid.uuid4().hex
        dataset.metadata["dataset_id"] = dataset_id
        dataset.metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self._items[dataset_id] = dataset
        return dataset_id

    def get(self, dataset_id: str) -> SpectralDataset:
        dataset = self._items.get(dataset_id)
        if dataset is None:
            raise KeyError("Dataset not found")
        return dataset


dataset_store = DatasetStore()


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _is_excel(filename: str) -> bool:
    return _extension(filename) in {"xlsx", "xls"}


def _is_zip(filename: str) -> bool:
    return _extension(filename) == "zip"


def _safe_preview_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _read_excel_sheets(file_data: UploadedFileData, header: int | None = 0) -> Dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(io.BytesIO(file_data.content), sheet_name=None, header=header)
    except ImportError as exc:
        raise DatasetImportError("Для чтения Excel нужен openpyxl. Установите зависимости из requirements.txt.") from exc
    except Exception as exc:
        raise DatasetImportError(f"Не удалось прочитать Excel-файл {file_data.filename}: {exc}") from exc


def _column_name_text(value: Any) -> str:
    return str(value).strip()


def _looks_like_axis_column(name: Any) -> bool:
    text = _column_name_text(name).lower()
    return any(hint in text for hint in AXIS_COLUMN_HINTS)


def _looks_like_target_column(name: Any) -> bool:
    return _column_name_text(name).lower() in TARGET_COLUMN_HINTS


def _looks_like_numeric_target_column(name: Any) -> bool:
    return _column_name_text(name).lower() in NUMERIC_TARGET_COLUMN_HINTS


def _looks_like_class_target_column(name: Any) -> bool:
    return _column_name_text(name).lower() in CLASS_TARGET_COLUMN_HINTS


def _looks_like_id_column(name: Any) -> bool:
    text = _column_name_text(name).lower()
    return text in ID_COLUMN_HINTS or text.endswith("_id")


def _numeric_header(value: Any) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    result: List[str] = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0 and series.notna().sum() >= max(1, int(len(df) * 0.8)):
            result.append(str(col))
    return result


def _sheet_preview(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    clean = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    columns = [str(col) for col in clean.columns]
    numeric_target_candidates = [
        col for col in columns
        if _looks_like_numeric_target_column(col) and pd.to_numeric(clean[col], errors="coerce").notna().sum() >= max(1, int(len(clean) * 0.8))
    ]
    class_target_candidates = [col for col in columns if _looks_like_class_target_column(col)]
    numeric_header_count = sum(_numeric_header(col) for col in columns)
    sheet_role = "dataset" if numeric_header_count >= 3 and (numeric_target_candidates or class_target_candidates) else "auxiliary/reference"
    recommendation = "рекомендуется" if sheet_role == "dataset" else "вспомогательный лист, target не найден"
    return {
        "name": name,
        "rows": int(clean.shape[0]),
        "columns": columns,
        "numeric_columns": _numeric_columns(clean),
        "target_candidates": numeric_target_candidates + [col for col in class_target_candidates if col not in numeric_target_candidates],
        "numeric_target_candidates": numeric_target_candidates,
        "class_target_candidates": class_target_candidates,
        "id_candidates": [col for col in columns if _looks_like_id_column(col)],
        "axis_candidates": [col for col in columns if _looks_like_axis_column(col)],
        "sheet_role": sheet_role,
        "recommendation": recommendation,
        "head": [
            {str(k): _safe_preview_value(v) for k, v in row.items()}
            for row in clean.head(8).to_dict(orient="records")
        ],
    }


def _detect_excel_layout(sheets: Dict[str, pd.DataFrame]) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []
    first_name, first_df = next(iter(sheets.items()))
    first_clean = first_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    first_columns = [str(col) for col in first_clean.columns]
    first_numeric_header_count = sum(_numeric_header(col) for col in first_columns)
    first_has_target = any(_looks_like_target_column(col) for col in first_columns)
    first_has_id = any(_looks_like_id_column(col) for col in first_columns)
    if first_has_target and first_numeric_header_count >= 3:
        reasons.append(f"Лист '{first_name}' содержит target-колонку и числовые заголовки спектральных признаков.")
        return "excel_rows", 0.9 if first_has_id else 0.8, reasons

    if len(sheets) > 1:
        column_like = 0
        for sheet_name, df in sheets.items():
            clean = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if clean.shape[1] >= 2 and (_looks_like_axis_column(clean.columns[0]) or _numeric_columns(clean)[0:1]):
                numeric_after_first = 0
                for col in clean.columns[1:]:
                    numeric_after_first += int(pd.to_numeric(clean[col], errors="coerce").notna().sum() >= max(2, len(clean) // 2))
                if numeric_after_first >= 1:
                    column_like += 1
                    reasons.append(f"Лист '{sheet_name}' похож на таблицу wavenumber + спектры по столбцам.")
        if column_like == len(sheets):
            return "excel_columns", 0.9, reasons

    first_name, first_df = next(iter(sheets.items()))
    columns = [str(col) for col in first_df.columns]
    numeric_header_count = sum(_numeric_header(col) for col in columns)
    has_target = any(_looks_like_target_column(col) for col in columns)
    has_id = any(_looks_like_id_column(col) for col in columns)
    if has_target and numeric_header_count >= 3:
        reasons.append(
            f"Лист '{first_name}' содержит target-колонку и много числовых заголовков признаков."
        )
        return "excel_rows", 0.85 if has_id else 0.75, reasons

    if len(sheets) == 1 and first_df.shape[1] >= 2 and _looks_like_axis_column(first_df.columns[0]):
        return "excel_columns", 0.65, ["Первая колонка похожа на спектральную ось."]

    return "manual", 0.2, ["Структура Excel не распознана однозначно."]


def _zip_file_entries(file_data: UploadedFileData) -> Tuple[List[str], List[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(file_data.content)) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
    except zipfile.BadZipFile as exc:
        raise DatasetImportError("ZIP-архив поврежден или имеет неверный формат.") from exc
    spectra = [name for name in names if _extension(name) in {"txt", "csv", "esp", "asc"}]
    nested_archives = [name for name in names if _extension(name) in {"zip", "rar", "7z"}]
    return spectra, nested_archives


def auto_detect_layout(files: Sequence[UploadedFileData]) -> Dict[str, Any]:
    if not files:
        return {"layout": "manual", "confidence": 0.0, "reasons": ["Файлы не переданы."]}
    if len(files) == 1 and _is_excel(files[0].filename):
        sheets = _read_excel_sheets(files[0], header=0)
        layout, confidence, reasons = _detect_excel_layout(sheets)
        return {"layout": layout, "confidence": confidence, "reasons": reasons}
    if len(files) == 1 and _is_zip(files[0].filename):
        spectra, nested_archives = _zip_file_entries(files[0])
        reasons = [f"В ZIP найдено спектральных файлов: {len(spectra)}."]
        if nested_archives:
            reasons.append("Найдены вложенные архивы: " + ", ".join(nested_archives[:10]))
        return {"layout": "txt_folder" if spectra else "manual", "confidence": 0.9 if spectra else 0.2, "reasons": reasons}
    if all(_extension(file.filename) in {"txt", "csv", "esp", "asc"} for file in files):
        return {
            "layout": "txt_folder",
            "confidence": 0.75,
            "reasons": ["Переданы отдельные TXT/CSV/ESP-файлы; используется режим один файл = один спектр."],
        }
    return {"layout": "manual", "confidence": 0.1, "reasons": ["Структура не распознана автоматически."]}


def preview_files(files: Sequence[UploadedFileData]) -> Dict[str, Any]:
    if not files:
        raise DatasetImportError("Не переданы файлы для предпросмотра.")
    detected = auto_detect_layout(files)
    file = files[0]
    result: Dict[str, Any] = {
        "status": "ok",
        "files": [{"filename": item.filename, "size": len(item.content), "format": _extension(item.filename)} for item in files],
        "suggested_layout": detected,
        "layouts": ["excel_columns", "excel_rows", "txt_folder", "long_table"],
        "target_options": ["column", "sheet_name", "folder_name", "filename_regex", "file_name", "manual", "none", "target_file"],
    }
    if len(files) == 1 and _is_excel(file.filename):
        sheets = _read_excel_sheets(file, header=0)
        result["excel"] = {"sheets": [_sheet_preview(name, df) for name, df in sheets.items()]}
    elif len(files) == 1 and _is_zip(file.filename):
        spectra, nested_archives = _zip_file_entries(file)
        preview_metadata = [_metadata_from_spectrum_filename(name) for name in spectra[:500]]
        result["zip"] = {
            "spectral_files": spectra[:200],
            "spectral_file_count": len(spectra),
            "nested_archives": nested_archives,
            "class_candidates": sorted({Path(name).parts[0] for name in spectra if len(Path(name).parts) > 1}),
            "metadata": _detected_measurement_metadata(preview_metadata),
            "sample_metadata_preview": preview_metadata[:10],
        }
    else:
        result["separate_files"] = [item.filename for item in files]
    return result


def _read_config(raw_config: str | None) -> Dict[str, Any]:
    if not raw_config:
        return {}
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise DatasetImportError("import_config должен быть валидным JSON.") from exc


def _to_numeric_frame(df: pd.DataFrame, context: str) -> pd.DataFrame:
    clean = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    numeric = clean.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise DatasetImportError(f"{context}: обнаружены нечисловые значения в выбранной числовой матрице.")
    return numeric.astype(float)


def _class_names(y: Optional[List[str]]) -> Optional[List[str]]:
    return sorted(Counter(y).keys()) if y else None


def _target_type(y: Optional[List[str]]) -> str:
    if not y:
        return "none"
    numeric = 0
    for value in y:
        try:
            float(str(value).replace(",", "."))
            numeric += 1
        except ValueError:
            pass
    if numeric == len(y):
        return "numeric"
    return "categorical"


def _target_distribution(y: Optional[List[str]], target_type: str) -> Dict[str, int]:
    if not y:
        return {}
    counts = dict(Counter([str(value) for value in y]))
    if target_type == "numeric" and len(counts) > 20:
        return {}
    return counts


def _target_stats(y: Optional[List[str]], target_type: str) -> Dict[str, Any]:
    if not y:
        return {"unique_count": 0}
    unique_count = len(set(str(value) for value in y))
    if target_type != "numeric":
        return {"unique_count": unique_count}
    values = np.asarray([float(str(value).replace(",", ".")) for value in y], dtype=float)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": [float(np.min(values)), float(np.max(values))],
        "unique_count": unique_count,
    }


def estimate_snr(X: np.ndarray) -> Dict[str, Any]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return {"mean": None, "min": None, "max": None, "low_snr_indices": []}
    snr_values: List[float] = []
    low_indices: List[int] = []
    for idx, row in enumerate(matrix):
        finite = row[np.isfinite(row)]
        if finite.size < 5:
            snr_values.append(float("nan"))
            low_indices.append(idx)
            continue
        signal = float(np.nanmax(finite) - np.nanmin(finite))
        if finite.size >= 7:
            window = min(15, finite.size if finite.size % 2 == 1 else finite.size - 1)
            window = max(5, window)
            try:
                smoothed = smooth_signal(finite, window_length=window, polyorder=min(3, window - 1))
                noise = float(np.nanstd(finite - smoothed))
            except Exception:
                noise = float(np.nanstd(np.diff(finite)))
        else:
            noise = float(np.nanstd(np.diff(finite)))
        if noise <= 1e-12:
            snr = float("inf") if signal > 0 else 0.0
        else:
            snr = signal / noise
        snr_values.append(float(snr))
        if np.isfinite(snr) and snr < 5:
            low_indices.append(idx)
    finite_snr = np.asarray([v for v in snr_values if np.isfinite(v)], dtype=float)
    return {
        "mean": float(np.mean(finite_snr)) if finite_snr.size else None,
        "min": float(np.min(finite_snr)) if finite_snr.size else None,
        "max": float(np.max(finite_snr)) if finite_snr.size else None,
        "low_snr_indices": low_indices[:20],
        "method": "signal_range / std(residual_after_smoothing)",
    }


def analyze_dataset_structure(X: np.ndarray, axis: Optional[np.ndarray] = None) -> Dict[str, Any]:
    matrix = np.asarray(X, dtype=float)
    warnings: List[str] = []
    nan_mask = ~np.isfinite(matrix)
    nan_count = int(np.sum(nan_mask))
    missing_fraction = float(np.mean(nan_mask)) if matrix.size else 0.0
    empty_spectra_count = int(np.sum(np.all(nan_mask, axis=1))) if matrix.ndim == 2 else 0
    empty_features_count = int(np.sum(np.all(nan_mask, axis=0))) if matrix.ndim == 2 else 0
    has_negative = bool(np.any(matrix[np.isfinite(matrix)] < 0)) if matrix.size else False
    if nan_count:
        warnings.append(f"Найдено NaN/Inf значений: {nan_count}.")
    if empty_spectra_count:
        warnings.append(f"Полностью пустых спектров: {empty_spectra_count}.")
    if empty_features_count:
        warnings.append(f"Полностью пустых спектральных точек: {empty_features_count}.")
    if axis is None or np.asarray(axis).size == 0:
        warnings.append("Спектральная ось не задана.")
    else:
        axis_arr = np.asarray(axis, dtype=float)
        if not np.all(np.isfinite(axis_arr)):
            warnings.append("Спектральная ось содержит NaN/Inf.")
        if axis_arr.ndim != 1:
            warnings.append("Спектральная ось должна быть одномерной.")
        elif axis_arr.size != matrix.shape[1]:
            warnings.append("Длина спектральной оси не совпадает с числом признаков.")
    snr = estimate_snr(matrix)
    if snr.get("mean") is not None and snr["mean"] < 5:
        warnings.append("Средний SNR ниже 5: данные могут быть шумными.")
    status = "error" if empty_spectra_count or empty_features_count or missing_fraction > 0.2 else ("warning" if warnings else "ok")
    return {
        "status": status,
        "n_samples": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
        "n_features": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "missing_fraction": missing_fraction,
        "nan_count": nan_count,
        "empty_spectra_count": empty_spectra_count,
        "empty_features_count": empty_features_count,
        "has_negative_intensity": has_negative,
        "snr": snr,
        "warnings": warnings,
    }


def suggest_analysis_methods_from_summary(
    *,
    n_samples: int,
    n_features: int,
    target_name: Optional[str],
    target_type: str,
    distribution: Dict[str, int],
    has_processed: bool,
    duplicate_ids: bool,
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if target_type == "none" or not target_name:
        suggestions.append({"model_type": "pca", "title": "PCA", "reason": "target не задан: PCA подходит для визуального анализа структуры данных."})
        suggestions.append({"model_type": "kmeans", "title": "k-means", "reason": "Можно искать естественные группы без target."})
        suggestions.append({"model_type": "hca", "title": "HCA", "reason": "Иерархическая кластеризация без заранее заданных классов."})
        return suggestions

    counts = list(distribution.values())
    imbalance = len(counts) >= 2 and min(counts) / max(counts) < 0.3
    if target_type == "numeric":
        suggestions.append({"model_type": "pca", "title": "PCA", "reason": "Полезно проверить структуру и выбросы перед регрессией."})
        suggestions.append({"model_type": "pls", "title": "PLS Regression", "reason": "Числовой target: базовая модель для предсказания концентрации."})
        suggestions.append({"model_type": "svr", "title": "SVR", "reason": "Нелинейная регрессионная модель для сравнения с PLS."})
    else:
        suggestions.append({"model_type": "pca", "title": "PCA", "reason": "Сначала удобно визуально проверить разделимость классов."})
        suggestions.append({"model_type": "plsda", "title": "PLS-DA", "reason": "target категориальный: подходит для классификации спектров."})
        if n_samples >= 30:
            suggestions.append({"model_type": "svm", "title": "SVM", "reason": "Можно использовать для сравнения качества классификации."})
            suggestions.append({"model_type": "random_forest", "title": "Random Forest", "reason": "Дополнительная baseline-модель классификации и оценка важности признаков."})
            suggestions.append({"model_type": "decision_tree", "title": "Decision Tree", "reason": "Простая интерпретируемая классификационная модель."})
        suggestions.append({"model_type": "kmeans", "title": "k-means", "reason": "Можно сравнить найденные кластеры с известными классами."})
        suggestions.append({"model_type": "hca", "title": "HCA", "reason": "Иерархическая кластеризация для проверки естественной группировки."})
    if n_samples < max(20, n_features // 100):
        suggestions.append({"model_type": "warning", "title": "Риск переобучения", "reason": "Образцов мало относительно числа признаков; начните с PCA/PLS и осторожной валидации."})
    if imbalance:
        suggestions.append({"model_type": "warning", "title": "Несбалансированные классы", "reason": "Рекомендуется stratified split; для поддерживающих моделей class_weight='balanced'."})
    if duplicate_ids:
        suggestions.append({"model_type": "warning", "title": "Повторяющиеся sample_id", "reason": "Если это повторные измерения, для строгой оценки нужен group split."})
    if not has_processed:
        suggestions.append({"model_type": "preprocessing", "title": "Предобработка", "reason": "Для Raman/SERS обычно полезны baseline correction, сглаживание и SNV."})
    return suggestions


def _measurement_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("measurement_metadata") or {}
    allowed = [
        "slit_width",
        "grating_lines_per_mm",
        "laser_wavelength",
        "integration_time",
        "accumulation_count",
        "spectrometer_name",
    ]
    return {key: value for key in allowed if (value := raw.get(key)) not in (None, "")}


def _unique_sample_id(sample_id: str, sheet_name: str, seen: set[str], force_prefix: bool = False) -> str:
    base = str(sample_id)
    candidate = f"{sheet_name}_{base}" if force_prefix else base
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    prefixed = f"{sheet_name}_{base}"
    candidate = prefixed
    suffix = 2
    while candidate in seen:
        candidate = f"{prefixed}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _sample_value(row: pd.Series, name: str) -> str:
    for col in row.index:
        if str(col).strip().lower() == name:
            value = row[col]
            if pd.isna(value):
                return "na"
            return str(value).strip().replace(" ", "_")
    return "na"


def _auto_sample_id(filename: str, sheet_name: str, row: pd.Series, index: int) -> str:
    stem = Path(filename).stem
    lower_stem = stem.lower()
    conc_y = _sample_value(row, "conc (y)")
    seria = _sample_value(row, "seria")
    if "albumin" in lower_stem or "voda" in lower_stem:
        return f"albumin_seria{seria}_conc{conc_y}_row{index + 1}"
    date = _sample_value(row, "date")
    cuvetta = _sample_value(row, "cuvetta")
    return f"{stem}_date{date}_seria{seria}_cuvetta{cuvetta}_row{index + 1}"


def _interpolate_matrix_to_grid(
    rows: Sequence[np.ndarray],
    axes: Sequence[np.ndarray],
    grid: np.ndarray,
) -> np.ndarray:
    return np.vstack([np.interp(grid, axis, row) for row, axis in zip(rows, axes)])


class ExcelColumnSpectraParser:
    def parse(self, file_data: UploadedFileData, config: Dict[str, Any]) -> SpectralDataset:
        sheets = _read_excel_sheets(file_data, header=0)
        selected_sheets = config.get("sheets") or list(sheets.keys())
        axis_column = config.get("axis_column")
        target_source = config.get("target_source", "sheet_name")
        manual_target = config.get("manual_target")
        sheet_mode = config.get("sheet_mode", "sheet_as_class")
        interpolation = config.get("interpolation") or {}

        axis_ref: Optional[np.ndarray] = None
        axes: List[np.ndarray] = []
        X_rows: List[np.ndarray] = []
        sample_ids: List[str] = []
        source_sheets: List[str] = []
        source_items: List[str] = []
        y: List[str] = []
        warnings: List[str] = []
        seen_ids: set[str] = set()
        force_prefix = len(selected_sheets) > 1 and sheet_mode == "sheet_as_class"

        for sheet_name in selected_sheets:
            if sheet_name not in sheets:
                raise DatasetImportError(f"Лист '{sheet_name}' не найден в Excel-файле.")
            df = sheets[sheet_name].dropna(axis=0, how="all").dropna(axis=1, how="all")
            axis_col = axis_column or df.columns[0]
            if axis_col not in df.columns:
                raise DatasetImportError(f"Колонка оси '{axis_col}' не найдена на листе '{sheet_name}'.")
            axis = pd.to_numeric(df[axis_col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(axis).any():
                raise DatasetImportError(f"Ось на листе '{sheet_name}' содержит нечисловые значения.")
            order = np.argsort(axis)
            axis = axis[order]

            sample_columns = config.get("sample_columns")
            if not sample_columns or sample_columns == "all_except_axis":
                sample_columns = [col for col in df.columns if col != axis_col]
            for column in sample_columns:
                numeric_values = pd.to_numeric(df[column], errors="coerce")
                if numeric_values.notna().sum() < len(df):
                    if str(column).lower().startswith("unnamed") or numeric_values.notna().sum() == 0:
                        warnings.append(f"Столбец '{column}' на листе '{sheet_name}' пропущен: не является полным числовым спектром.")
                        continue
                values = numeric_values.to_numpy(dtype=float)[order]
                if np.isnan(values).any():
                    raise DatasetImportError(f"Спектр '{column}' на листе '{sheet_name}' содержит пропуски/текст.")
                if axis_ref is None:
                    axis_ref = axis
                elif axis_ref.shape != axis.shape or not np.allclose(axis_ref, axis):
                    warnings.append("У выбранных листов различается спектральная ось. Спектры приведены к общей сетке.")
                X_rows.append(values)
                axes.append(axis)
                sample_ids.append(_unique_sample_id(str(column), str(sheet_name), seen_ids, force_prefix=force_prefix))
                source_sheets.append(str(sheet_name))
                source_items.append(file_data.filename)
                if target_source == "sheet_name" or sheet_mode == "sheet_as_class":
                    y.append(str(sheet_name))
                elif target_source == "file_name":
                    y.append(Path(file_data.filename).stem)
                elif target_source == "manual":
                    y.append(str(manual_target or "class_1"))

        if axis_ref is None or not X_rows:
            raise DatasetImportError("Не удалось собрать спектры из Excel-таблицы.")
        axis_mismatch = any(axis_ref.shape != axis.shape or not np.allclose(axis_ref, axis) for axis in axes)
        if axis_mismatch:
            mode = interpolation.get("grid_mode", "first_axis")
            if mode == "cancel":
                raise DatasetImportError("У выбранных листов различается спектральная ось. Импорт отменён настройкой пользователя.")
            if mode == "intersection":
                axis_out = _common_grid(axes, mode="intersection", step=interpolation.get("step", "auto"))
            elif mode == "manual":
                axis_out = _common_grid(
                    axes,
                    mode="manual",
                    step=interpolation.get("step", "auto"),
                    manual_min=interpolation.get("grid_min"),
                    manual_max=interpolation.get("grid_max"),
                )
            else:
                axis_out = axis_ref
            X_out = _interpolate_matrix_to_grid(X_rows, axes, axis_out)
        else:
            axis_out = axis_ref
            X_out = np.vstack(X_rows)
        y_out: Optional[List[str]] = y if y else None
        target_name = target_source if y_out else None
        return SpectralDataset(
            X=X_out.astype(float),
            axis=axis_out.astype(float),
            sample_ids=sample_ids,
            y=y_out,
            target_name=target_name,
            class_names=_class_names(y_out),
            metadata={
                "import_config": config,
                "selected_sheets": list(selected_sheets),
                "sheet_mode": sheet_mode,
                "axis_warnings": list(dict.fromkeys(warnings)),
                "measurement_metadata": _measurement_metadata(config),
                "preprocessing_status": "not_applied",
            },
            source_layout="excel_columns",
            source_files=[file_data.filename],
            source_sheets=source_sheets,
            source_items=source_items,
        )


class ExcelRowSpectraParser:
    def parse(self, file_data: UploadedFileData, config: Dict[str, Any]) -> SpectralDataset:
        sheets = _read_excel_sheets(file_data, header=0)
        selected_sheets = config.get("sheets") or [config.get("sheet") or next(iter(sheets.keys()))]
        target_source = config.get("target_source", "column")
        target_role = config.get("target_role", "regression")
        X_parts: List[np.ndarray] = []
        sample_ids: List[str] = []
        source_sheets: List[str] = []
        source_items: List[str] = []
        sample_metadata: List[Dict[str, Any]] = []
        y_parts: List[str] = []
        axis_ref: Optional[np.ndarray] = None
        target_name: Optional[str] = None
        target_type: Optional[str] = None
        alternative_targets: List[Dict[str, str]] = []

        for sheet in selected_sheets:
            if sheet not in sheets:
                raise DatasetImportError(f"Лист '{sheet}' не найден в Excel-файле.")
            df = sheets[sheet].dropna(axis=0, how="all").dropna(axis=1, how="all")
            columns = list(df.columns)

            id_column = config.get("id_column") or next((col for col in columns if _looks_like_id_column(col)), None)
            target_column = None
            if target_source == "column":
                configured_target = config.get("target_column")
                if configured_target:
                    target_column = configured_target
                elif target_role == "classes":
                    target_column = next((col for col in columns if _looks_like_class_target_column(col)), None)
                else:
                    target_column = next(
                        (
                            col for col in columns
                            if _looks_like_numeric_target_column(col)
                            and pd.to_numeric(df[col], errors="coerce").notna().sum() >= max(1, int(len(df) * 0.8))
                        ),
                        None,
                    ) or next((col for col in columns if _looks_like_target_column(col)), None)
            class_target = next((col for col in columns if _looks_like_class_target_column(col) and col != target_column), None)
            if class_target:
                alt = {"name": str(class_target), "target_type": "categorical", "role": "classification"}
                if alt not in alternative_targets:
                    alternative_targets.append(alt)
            feature_columns = config.get("feature_columns")
            if not feature_columns or feature_columns == "numeric_headers":
                excluded = {col for col in columns if str(col).strip().lower() in ROW_METADATA_HINTS}
                excluded.update({col for col in [id_column, target_column, class_target] if col is not None})
                feature_columns = [col for col in columns if col not in excluded and _numeric_header(col)]
            if not feature_columns:
                raise DatasetImportError("Не найдены спектральные признаки. Укажите feature_columns или числовые заголовки.")

            axis = np.array([float(str(col).replace(",", ".")) for col in feature_columns], dtype=float)
            if axis_ref is None:
                axis_ref = axis
            elif axis_ref.shape != axis.shape or not np.allclose(axis_ref, axis):
                raise DatasetImportError("У выбранных листов Excel разные спектральные признаки. Для excel_rows пока нужны одинаковые числовые заголовки.")
            X = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if np.isnan(X).any():
                raise DatasetImportError("Матрица спектров содержит пропуски или нечисловые значения.")

            sheet_metadata_columns = [
                col for col in columns
                if str(col).strip().lower() in ROW_METADATA_HINTS or col in {class_target}
            ]
            for idx, row in df.iterrows():
                row_meta = {str(col): _safe_preview_value(row[col]) for col in sheet_metadata_columns if col in df.columns}
                row_meta["source_sheet"] = str(sheet)
                row_meta["source_file"] = file_data.filename
                sample_metadata.append(row_meta)

            if id_column in df.columns:
                ids = df[id_column].astype(str).tolist()
            else:
                ids = [
                    _auto_sample_id(file_data.filename, str(sheet), row, idx)
                    for idx, row in df.reset_index(drop=True).iterrows()
                ]
            sample_ids.extend(ids if len(selected_sheets) == 1 else [f"{sheet}_{item}" for item in ids])
            source_sheets.extend([str(sheet)] * len(df))
            source_items.extend([file_data.filename] * len(df))
            X_parts.append(X.astype(float))
            if target_source == "column" and target_column in df.columns:
                y_parts.extend(df[target_column].astype(str).tolist())
                target_name = str(target_column)
                numeric_target = pd.to_numeric(df[target_column], errors="coerce").notna().sum() == len(df)
                target_type = "categorical" if target_role == "classes" or _looks_like_class_target_column(target_column) else ("numeric" if numeric_target else "categorical")
            elif target_source == "sheet_name":
                y_parts.extend([str(sheet)] * len(df))
                target_name = "sheet_name"
                target_type = "categorical"
            elif target_source == "manual":
                y_parts.extend([str(config.get("manual_target") or "class_1")] * len(df))
                target_name = "manual"
                target_type = "categorical"

        if axis_ref is None or not X_parts:
            raise DatasetImportError("Не удалось собрать строки спектров из Excel.")
        y = y_parts if len(y_parts) == sum(part.shape[0] for part in X_parts) else None
        duplicate_ids = [item for item, count in Counter(sample_ids).items() if count > 1]

        final_target_type = target_type or _target_type(y)
        return SpectralDataset(
            X=np.vstack(X_parts).astype(float),
            axis=axis_ref.astype(float),
            sample_ids=sample_ids,
            y=y,
            target_name=target_name,
            class_names=None if final_target_type == "numeric" else _class_names(y),
            metadata={
                "import_config": config,
                "duplicate_ids": duplicate_ids,
                "selected_sheets": list(selected_sheets),
                "sheet_mode": config.get("sheet_mode", "single_sheet"),
                "target_type": final_target_type,
                "alternative_targets": alternative_targets,
                "measurement_metadata": _measurement_metadata(config),
                "preprocessing_status": "not_applied",
            },
            source_layout="excel_rows",
            source_files=[file_data.filename],
            source_sheets=source_sheets,
            source_items=source_items,
            sample_metadata=sample_metadata,
            group_ids=sample_ids if duplicate_ids else None,
        )


def _parse_two_column_text(content: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DatasetImportError(f"Не удалось декодировать файл {filename}.")
    rows: List[Tuple[float, float]] = []
    intensity_only: List[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        normalized = stripped.replace(",", ".").replace(";", " ").replace("\t", " ")
        parts = [part for part in normalized.split() if part]
        if len(parts) < 2:
            if len(parts) == 1:
                try:
                    intensity_only.append(float(parts[0]))
                except ValueError:
                    pass
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(rows) < 2 and len(intensity_only) >= 2:
        intensity = np.asarray(intensity_only, dtype=float)
        axis = np.arange(intensity.size, dtype=float)
        return axis, intensity, False, True
    if len(rows) < 2:
        raise DatasetImportError(f"Файл {filename} не содержит минимум две числовые пары x/intensity.")
    arr = np.asarray(rows, dtype=float)
    axis = arr[:, 0]
    intensity = arr[:, 1]
    reverse_axis = bool(axis[0] > axis[-1])
    order = np.argsort(axis)
    return axis[order], intensity[order], reverse_axis, False


def _target_from_path(name: str, source: str, manual_target: Optional[str], filename_regex: Optional[str] = None) -> Optional[str]:
    path = Path(name)
    if source in {"none", "no_target", "", None}:
        return None
    if source in {"folder_name", "class_from_folder"}:
        parts = path.parts
        return str(parts[-2]) if len(parts) > 1 else None
    if source in {"filename_regex", "class_from_filename_regex"}:
        if not filename_regex:
            return None
        match = re.search(filename_regex, str(name))
        if not match:
            return None
        return str(match.group(1) if match.groups() else match.group(0))
    if source == "file_name":
        return path.stem
    if source == "manual":
        return manual_target or "class_1"
    return None


def _metadata_from_spectrum_filename(name: str) -> Dict[str, Any]:
    filename = Path(name).name
    stem = Path(filename).stem
    text = str(name).lower().replace(",", ".")
    sample_name = re.sub(r"[_\s-]*\d+(?:\.\d+)?\s*mw(?:t)?", "", stem, flags=re.IGNORECASE)
    sample_name = re.sub(r"[_\s-]*\d+(?:\.\d+)?\s*sec", "", sample_name, flags=re.IGNORECASE)
    sample_name = re.sub(r"[_\s-]*x\d+", "", sample_name, flags=re.IGNORECASE)
    sample_name = re.sub(r"[_\s-]*\d+[_\s-]*l[_\s-]*mm", "", sample_name, flags=re.IGNORECASE)
    sample_name = re.sub(r"[_\s-]*slit[_\s-]*\d+(?:\.\d+)?\s*um", "", sample_name, flags=re.IGNORECASE)
    sample_name = re.sub(r"[_\s-]+$", "", sample_name).strip("_ -") or stem
    metadata: Dict[str, Any] = {
        "sample_id": stem,
        "sample_name": sample_name,
        "source_file": name,
    }
    slit = re.search(r"slit[_\s-]*(\d+(?:\.\d+)?)\s*um", text)
    grating = re.search(r"grating[_\s-]*(\d{3,4})", text) or re.search(r"(\d{3,4})[_\s-]*l[/_\s-]*mm", text)
    exposure = re.search(r"(\d+(?:\.\d+)?)\s*sec", text)
    accumulations = re.search(r"x(\d+)", text)
    power = re.search(r"(\d+(?:\.\d+)?)\s*mw(?:t)?", text)
    if slit:
        metadata["slit_width_um"] = float(slit.group(1))
    if grating:
        metadata["grating_lines_mm"] = int(grating.group(1))
    if exposure:
        metadata["exposure_time_s"] = float(exposure.group(1))
    if accumulations:
        metadata["accumulations"] = int(accumulations.group(1))
    if power:
        metadata["power_mw"] = float(power.group(1))
    return metadata


def _unique_metadata_values(sample_metadata: Sequence[Dict[str, Any]], key: str) -> List[Any]:
    values = []
    for item in sample_metadata:
        value = item.get(key)
        if value is not None and value not in values:
            values.append(value)
    return sorted(values)


def _detected_measurement_metadata(sample_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["slit_width_um", "grating_lines_mm", "exposure_time_s", "accumulations", "power_mw", "sample_name"]
    detected: Dict[str, Any] = {}
    for key in keys:
        values = _unique_metadata_values(sample_metadata, key)
        if values:
            detected[f"{key}_values"] = values
            if len(values) == 1:
                detected[key] = values[0]
    return detected


def _common_grid(
    axes: Sequence[np.ndarray],
    mode: str,
    step: str | float | int | None,
    manual_min: Optional[float] = None,
    manual_max: Optional[float] = None,
) -> np.ndarray:
    if mode == "first_axis":
        return np.asarray(axes[0], dtype=float)
    if mode == "manual":
        if manual_min is None or manual_max is None:
            raise DatasetImportError("Для ручной сетки укажите grid_min и grid_max.")
        start, end = float(manual_min), float(manual_max)
    elif mode == "union":
        start = min(float(axis.min()) for axis in axes)
        end = max(float(axis.max()) for axis in axes)
    else:
        start = max(float(axis.min()) for axis in axes)
        end = min(float(axis.max()) for axis in axes)
    if start >= end:
        raise DatasetImportError("Диапазоны спектров не пересекаются; невозможно построить общую сетку.")
    if step in (None, "", "auto"):
        diffs = np.concatenate([np.diff(axis) for axis in axes if axis.size > 1])
        step_value = float(np.median(np.abs(diffs)))
    else:
        step_value = float(step)
    if step_value <= 0:
        raise DatasetImportError("Шаг общей сетки должен быть положительным.")
    return np.arange(start, end + step_value * 0.5, step_value, dtype=float)


class TxtFolderSpectraParser:
    def parse(self, files: Sequence[UploadedFileData], config: Dict[str, Any]) -> SpectralDataset:
        entries: List[Tuple[str, bytes]] = []
        nested_archives: List[str] = []
        if len(files) == 1 and _is_zip(files[0].filename):
            with zipfile.ZipFile(io.BytesIO(files[0].content)) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    ext = _extension(name)
                    if ext in {"txt", "csv", "esp", "asc"}:
                        entries.append((name, archive.read(name)))
                    elif ext in {"zip", "rar", "7z"}:
                        nested_archives.append(name)
        else:
            entries = [(file.filename, file.content) for file in files if _extension(file.filename) in {"txt", "csv", "esp", "asc"}]
        if not entries:
            raise DatasetImportError("Не найдены TXT/CSV/ESP спектры для режима один файл = один спектр.")

        target_source = config.get("target_source", "none")
        manual_target = config.get("manual_target")
        filename_regex = config.get("filename_regex") or config.get("target_regex")
        spectra: List[Tuple[str, Optional[str], np.ndarray, np.ndarray, bool, bool]] = []
        sample_metadata: List[Dict[str, Any]] = []
        for name, content in entries:
            axis, intensity, reversed_axis, intensity_only = _parse_two_column_text(content, name)
            metadata = _metadata_from_spectrum_filename(name)
            if intensity_only:
                metadata["intensity_only"] = True
            target = _target_from_path(name, target_source, manual_target, filename_regex)
            if target is not None:
                metadata["target"] = target
            spectra.append((name, target, axis, intensity, reversed_axis, intensity_only))
            sample_metadata.append(metadata)

        interpolation = config.get("interpolation") or {}
        enabled = bool(interpolation.get("enabled", True))
        if enabled:
            grid = _common_grid(
                [item[2] for item in spectra],
                mode=interpolation.get("grid_mode", "intersection"),
                step=interpolation.get("step", "auto"),
                manual_min=interpolation.get("grid_min"),
                manual_max=interpolation.get("grid_max"),
            )
            X = np.vstack([np.interp(grid, axis, intensity) for _name, _target, axis, intensity, _rev, _one_col in spectra])
            axis_out = grid
        else:
            first_axis = spectra[0][2]
            for name, _target, axis, _intensity, _rev, _one_col in spectra[1:]:
                if first_axis.shape != axis.shape or not np.allclose(first_axis, axis):
                    raise DatasetImportError(
                        f"Файл {name} имеет другую спектральную сетку. Включите interpolation.enabled=true."
                    )
            X = np.vstack([item[3] for item in spectra])
            axis_out = first_axis

        y = [target for _name, target, _axis, _intensity, _rev, _one_col in spectra if target is not None]
        y_out = y if len(y) == len(spectra) else None
        target_warning: Optional[str] = None
        if y_out:
            distribution = Counter(y_out)
            if len(distribution) == len(y_out) and all(count == 1 for count in distribution.values()):
                target_warning = (
                    "Каждый файл образует отдельный класс. Такой target непригоден для классификации. "
                    "Выберите 'Без target' или задайте regex, который группирует файлы в классы."
                )
                y_out = None
        detected_measurements = _detected_measurement_metadata(sample_metadata)
        slit_values = detected_measurements.get("slit_width_um_values", [])
        grating_values = detected_measurements.get("grating_lines_mm_values", [])
        measurement_metadata = {**detected_measurements, **_measurement_metadata(config)}
        return SpectralDataset(
            X=X.astype(float),
            axis=axis_out.astype(float),
            sample_ids=[Path(item[0]).stem for item in spectra],
            y=y_out,
            target_name=target_source if y_out else None,
            class_names=_class_names(y_out),
            metadata={
                "import_config": config,
                "nested_archives": nested_archives,
                "reverse_axis_files": [name for name, _target, _axis, _intensity, rev, _one_col in spectra if rev],
                "single_column_intensity_files": [name for name, _target, _axis, _intensity, _rev, one_col in spectra if one_col],
                "single_column_intensity_only": any(one_col for _name, _target, _axis, _intensity, _rev, one_col in spectra),
                "interpolation": interpolation,
                "slit_width_um_values": slit_values,
                "grating_lines_mm_values": grating_values,
                "exposure_time_s_values": detected_measurements.get("exposure_time_s_values", []),
                "accumulations_values": detected_measurements.get("accumulations_values", []),
                "power_mw_values": detected_measurements.get("power_mw_values", []),
                "sample_name_values": detected_measurements.get("sample_name_values", []),
                "sample_metadata_preview": sample_metadata[:10],
                "measurement_metadata": measurement_metadata,
                "target_warning": target_warning,
                "warnings": [target_warning] if target_warning else [],
                "preprocessing_status": "not_applied",
            },
            source_layout="txt_folder",
            source_files=[item[0] for item in spectra],
            source_items=[item[0] for item in spectra],
            sample_metadata=sample_metadata,
        )


def import_dataset(files: Sequence[UploadedFileData], config: Dict[str, Any]) -> SpectralDataset:
    layout = config.get("layout") or auto_detect_layout(files)["layout"]
    if layout == "long_table":
        raise DatasetImportError("Длинный формат таблицы запланирован, но пока не реализован.")
    if layout == "excel_columns":
        if len(files) != 1 or not _is_excel(files[0].filename):
            raise DatasetImportError("Для excel_columns загрузите один Excel-файл.")
        return ExcelColumnSpectraParser().parse(files[0], config)
    if layout == "excel_rows":
        if len(files) != 1 or not _is_excel(files[0].filename):
            raise DatasetImportError("Для excel_rows загрузите один Excel-файл.")
        return ExcelRowSpectraParser().parse(files[0], config)
    if layout == "txt_folder":
        return TxtFolderSpectraParser().parse(files, config)
    raise DatasetImportError("Не удалось определить layout. Выберите тип структуры вручную.")


def validate_dataset(dataset: SpectralDataset) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    X = np.asarray(dataset.X, dtype=float)
    axis = np.asarray(dataset.axis, dtype=float)

    if X.ndim != 2:
        errors.append("X должен быть двумерным массивом n_samples x n_features.")
    if axis.ndim != 1:
        errors.append("axis должен быть одномерным массивом.")
    if X.ndim == 2 and axis.ndim == 1 and axis.size != X.shape[1]:
        errors.append("Длина axis должна совпадать с числом признаков X.")
    if X.ndim == 2 and len(dataset.sample_ids) != X.shape[0]:
        errors.append("Количество sample_ids должно совпадать с числом строк X.")
    if dataset.y is not None and X.ndim == 2 and len(dataset.y) != X.shape[0]:
        errors.append("Количество target-значений должно совпадать с числом строк X.")
    if X.ndim == 2:
        if np.any(np.all(~np.isfinite(X), axis=1)):
            errors.append("Есть полностью пустые спектры.")
        if np.any(np.all(~np.isfinite(X), axis=0)):
            errors.append("Есть полностью пустые спектральные признаки.")
        missing_fraction = float(np.mean(~np.isfinite(X)))
        if missing_fraction > 0:
            warnings.append(f"Доля пропусков/некорректных чисел в X: {missing_fraction:.2%}.")
        if X.shape[0] < 6:
            warnings.append("Меньше 6 спектров: supervised-валидация будет нестабильной.")

    duplicates = [item for item, count in Counter(dataset.sample_ids).items() if count > 1]
    if duplicates:
        warnings.append("Найдены дубликаты sample_id: " + ", ".join(duplicates[:10]))
    for warning in dataset.metadata.get("axis_warnings", []) if isinstance(dataset.metadata.get("axis_warnings"), list) else []:
        warnings.append(str(warning))
    for warning in dataset.metadata.get("warnings", []) if isinstance(dataset.metadata.get("warnings"), list) else []:
        if warning:
            warnings.append(str(warning))

    distribution = dict(Counter(dataset.y or []))
    if distribution:
        target_type = str(dataset.metadata.get("target_type") or _target_type(dataset.y))
        counts = list(distribution.values())
        if target_type == "numeric":
            if X.ndim == 2 and X.shape[0] < 100:
                warnings.append("Набор данных небольшой. Метрики регрессии могут быть нестабильными. Рекомендуется использовать кросс-валидацию.")
            if len(distribution) <= min(10, max(2, X.shape[0] // 3)):
                warnings.append("Target содержит дискретные уровни концентрации. При случайном train/test split возможна переоценка качества. Рекомендуется KFold или leave-one-concentration-out.")
        else:
            if len(counts) >= 2 and min(counts) / max(counts) < 0.3:
                warnings.append("Классы сильно несбалансированы; используйте stratified split или class_weight.")
            if min(counts) < 2:
                warnings.append("В одном из классов меньше 2 образцов.")

    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": dataset.summary(),
    }


def _normalized_preview_strategy(strategy: str) -> str:
    aliases = {
        "first_n": "first",
        "first": "first",
        "balanced_by_class": "balanced",
        "balanced": "balanced",
        "random": "random",
    }
    return aliases.get(str(strategy or "").strip(), "balanced")


def spectra_preview(dataset: SpectralDataset, limit: int | str = 5, strategy: str = "balanced", version: str = "raw") -> Dict[str, Any]:
    X = dataset.processed_X if version == "processed" and dataset.processed_X is not None else dataset.X
    axis = dataset.processed_axis if version == "processed" and dataset.processed_axis is not None else dataset.axis
    indices = _preview_indices(dataset, limit, strategy=_normalized_preview_strategy(strategy))
    return {
        "dataset_id": dataset.dataset_id,
        "version": "processed" if X is dataset.processed_X else "raw",
        "axis": axis.astype(float).tolist(),
        "spectra": [
            {
                "sample_id": dataset.sample_ids[int(idx)],
                "target": dataset.y[int(idx)] if dataset.y else None,
                "intensity": X[int(idx)].astype(float).tolist(),
                "y": X[int(idx)].astype(float).tolist(),
            }
            for idx in indices
        ],
    }


def _preprocessing_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "none": {
            "baseline": {"method": "off"},
            "smoothing": {"method": "off"},
            "normalization": {"method": "off"},
            "crop": {"enabled": False},
        },
        "raman_standard": {
            "baseline": {"method": "als", "lambda": 100000.0, "p": 0.01, "iterations": 10},
            "smoothing": {"method": "savgol", "window_length": 15, "polyorder": 3},
            "normalization": {"method": "snv"},
            "crop": {"enabled": False},
        },
        "noisy": {
            "baseline": {"method": "als", "lambda": 100000.0, "p": 0.01, "iterations": 10},
            "smoothing": {"method": "gaussian", "sigma": 2.0},
            "normalization": {"method": "max"},
            "crop": {"enabled": False},
        },
        "class_comparison": {
            "baseline": {"method": "als", "lambda": 100000.0, "p": 0.01, "iterations": 10},
            "smoothing": {"method": "savgol", "window_length": 15, "polyorder": 3},
            "normalization": {"method": "snv"},
            "crop": {"enabled": False},
            "mean_by_class": False,
        },
    }


def resolve_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    preset = str(config.get("preset") or "custom")
    if preset != "custom":
        base = _preprocessing_presets().get(preset, _preprocessing_presets()["none"])
        merged = json.loads(json.dumps(base))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            elif key != "preset":
                merged[key] = value
        merged["preset"] = preset
        return merged
    return {
        "preset": "custom",
        "baseline": config.get("baseline") or {"method": "off"},
        "smoothing": config.get("smoothing") or {"method": "off"},
        "normalization": config.get("normalization") or {"method": "off"},
        "crop": config.get("crop") or {"enabled": False},
        "missing": config.get("missing") or {"action": "none"},
    }


def validate_preprocessing_config(dataset: SpectralDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = resolve_preprocessing_config(config)
    errors: List[str] = []
    warnings: List[str] = []
    n_features = int(dataset.X.shape[1])
    axis = dataset.axis

    crop = cfg.get("crop") or {}
    if crop.get("enabled"):
        crop_min = crop.get("min_axis")
        crop_max = crop.get("max_axis")
        if crop_min in (None, "") or crop_max in (None, ""):
            errors.append("Для обрезки диапазона нужно указать min_axis и max_axis.")
        else:
            lo, hi = float(crop_min), float(crop_max)
            if lo >= hi:
                errors.append("crop_min должен быть меньше crop_max.")
            elif int(np.sum((axis >= lo) & (axis <= hi))) < 5:
                errors.append("После обрезки диапазона осталось слишком мало спектральных точек.")

    baseline = cfg.get("baseline") or {}
    if baseline.get("method") == "als":
        lam = float(baseline.get("lambda", 100000.0))
        p = float(baseline.get("p", 0.01))
        iterations = int(baseline.get("iterations", 10))
        if lam <= 0:
            errors.append("lambda для ALS должен быть положительным.")
        if not (0 < p < 1):
            errors.append("p для ALS должен быть между 0 и 1.")
        if iterations < 1:
            errors.append("Число итераций ALS должно быть не меньше 1.")
    elif baseline.get("method") == "polynomial":
        if int(baseline.get("degree", 3)) < 1:
            errors.append("Степень polynomial baseline должна быть не меньше 1.")
    elif baseline.get("method") == "rolling_ball":
        if float(baseline.get("radius", 50.0)) <= 0:
            errors.append("Радиус Rolling Ball должен быть положительным.")

    smoothing = cfg.get("smoothing") or {}
    if smoothing.get("method") == "savgol":
        window = int(smoothing.get("window_length", 15))
        polyorder = int(smoothing.get("polyorder", 3))
        if window > n_features:
            errors.append("Окно сглаживания должно быть меньше количества спектральных точек.")
        if window % 2 == 0:
            errors.append("Для фильтра Савицкого — Голея размер окна должен быть нечётным.")
        if polyorder >= window:
            errors.append("polyorder должен быть меньше window_length.")
    elif smoothing.get("method") == "gaussian":
        if float(smoothing.get("sigma", 2.0)) <= 0:
            errors.append("sigma для Gaussian smoothing должен быть положительным.")
    elif smoothing.get("method") == "moving_average":
        window = int(smoothing.get("window_size", 5))
        if window < 2 or window > n_features:
            errors.append("Окно Moving Average должно быть от 2 до количества спектральных точек.")
    elif smoothing.get("method") == "whittaker":
        if float(smoothing.get("lambda", 1000.0)) <= 0:
            errors.append("lambda для Whittaker smoothing должен быть положительным.")

    if np.any(~np.isfinite(dataset.X)):
        warnings.append("В исходном датасете есть NaN/Inf. Перед обучением лучше удалить или заполнить такие значения.")

    return {"status": "error" if errors else ("warning" if warnings else "ok"), "errors": errors, "warnings": warnings, "config": cfg}


def _moving_average(y: np.ndarray, window_size: int) -> np.ndarray:
    window = max(2, int(window_size))
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize_vector(y: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(y))
    return y if norm == 0 else y / norm


def _normalize_area(axis: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = float(np.trapz(np.abs(y), axis))
    return y if area == 0 else y / area


def preprocess_matrix(dataset: SpectralDataset, config: Dict[str, Any], indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    validation = validate_preprocessing_config(dataset, config)
    if validation["status"] == "error":
        raise DatasetImportError("; ".join(validation["errors"]))
    cfg = validation["config"]
    axis = dataset.axis.astype(float)
    X = dataset.X.astype(float)
    selected_indices = list(indices) if indices is not None else list(range(X.shape[0]))

    crop = cfg.get("crop") or {}
    mask = np.ones(axis.shape, dtype=bool)
    if crop.get("enabled"):
        mask = (axis >= float(crop.get("min_axis"))) & (axis <= float(crop.get("max_axis")))
    axis_out = axis[mask]
    rows: List[np.ndarray] = []
    for idx in selected_indices:
        y = X[int(idx), mask].astype(float)
        baseline = cfg.get("baseline") or {}
        b_method = baseline.get("method", "off")
        if b_method == "als":
            y = y - baseline_als(y, lam=float(baseline.get("lambda", 100000.0)), p=float(baseline.get("p", 0.01)), niter=int(baseline.get("iterations", 10)))
        elif b_method == "polynomial":
            y, _baseline = baseline_polyfit(axis_out, y, degree=int(baseline.get("degree", 3)), n_iter=int(baseline.get("iterations", 10)), asymmetry=float(baseline.get("asymmetry", 0.01)))
        elif b_method == "rolling_ball":
            y, _baseline = baseline_rolling_ball(axis_out, y, radius=float(baseline.get("radius", 50.0)), radius_units=str(baseline.get("units", "x")))

        smoothing = cfg.get("smoothing") or {}
        s_method = smoothing.get("method", "off")
        if s_method == "savgol":
            y = smooth_signal(y, window_length=int(smoothing.get("window_length", 15)), polyorder=int(smoothing.get("polyorder", 3)))
        elif s_method == "gaussian":
            y = gaussian_smooth(y, sigma=float(smoothing.get("sigma", 2.0)))
        elif s_method == "moving_average":
            y = _moving_average(y, int(smoothing.get("window_size", 5)))
        elif s_method == "whittaker":
            y = whittaker_smooth(y, lmbd=float(smoothing.get("lambda", 1000.0)), d=int(smoothing.get("order", 2)))

        normalization = cfg.get("normalization") or {}
        n_method = normalization.get("method", "off")
        if n_method == "snv":
            y = normalize_snv(y)
        elif n_method == "max":
            y = normalize_max(y)
        elif n_method == "vector":
            y = _normalize_vector(y)
        elif n_method == "area":
            y = _normalize_area(axis_out, y)
        rows.append(y.astype(float))

    X_out = np.vstack(rows)
    if np.any(np.all(~np.isfinite(X_out), axis=1)):
        raise DatasetImportError("После предобработки появились полностью пустые спектры.")
    return {"X": X_out, "axis": axis_out, "config": cfg, "warnings": validation["warnings"]}


def _preview_indices(dataset: SpectralDataset, limit: int | str, strategy: str = "balanced") -> List[int]:
    n = dataset.X.shape[0]
    if str(limit) == "all":
        return list(range(n))
    limit = max(1, min(int(limit), n))
    strategy = _normalized_preview_strategy(strategy)
    if strategy == "first":
        return list(range(limit))
    if strategy == "balanced" and dataset.y:
        groups: Dict[str, List[int]] = {}
        for idx, target in enumerate(dataset.y):
            groups.setdefault(str(target), []).append(idx)
        picked: List[int] = []
        while len(picked) < limit and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(picked) < limit:
                    picked.append(groups[key].pop(0))
        return picked
    rng = np.random.default_rng(42)
    return sorted(rng.choice(np.arange(n), size=limit, replace=False).astype(int).tolist())


def preview_preprocessing(dataset: SpectralDataset, config: Dict[str, Any], limit: int | str = 5, strategy: str = "balanced") -> Dict[str, Any]:
    indices = _preview_indices(dataset, limit, strategy=strategy)
    processed = preprocess_matrix(dataset, config, indices=indices)
    raw_axis = dataset.axis.astype(float)
    raw_X = dataset.X[indices]
    quality = preprocessing_quality(raw_X, processed["X"])
    warnings = processed["warnings"] + quality["warnings"]
    return {
        "status": "ok",
        "warnings": warnings,
        "preprocessing_summary": preprocessing_summary(processed["config"], processed["axis"], raw_axis),
        "quality": quality,
        "raw_preview": {"axis": raw_axis.tolist()},
        "processed_preview": {"axis": processed["axis"].astype(float).tolist()},
        "axis_raw": raw_axis.tolist(),
        "axis_processed": processed["axis"].astype(float).tolist(),
        "spectra": [
            {
                "sample_id": dataset.sample_ids[int(idx)],
                "target": dataset.y[int(idx)] if dataset.y else None,
                "raw": raw_X[pos].astype(float).tolist(),
                "processed": processed["X"][pos].astype(float).tolist(),
            }
            for pos, idx in enumerate(indices)
        ],
    }


def preprocessing_summary(config: Dict[str, Any], axis: np.ndarray, raw_axis: Optional[np.ndarray] = None) -> Dict[str, Any]:
    return {
        "baseline": (config.get("baseline") or {}).get("method", "off"),
        "smoothing": (config.get("smoothing") or {}).get("method", "off"),
        "normalization": (config.get("normalization") or {}).get("method", "off"),
        "preset": config.get("preset", "custom"),
        "axis_min": float(np.min(axis)) if axis.size else None,
        "axis_max": float(np.max(axis)) if axis.size else None,
        "n_features": int(axis.size),
        "raw_n_features": int(raw_axis.size) if raw_axis is not None else None,
    }


def preprocessing_quality(raw_X: np.ndarray, processed_X: np.ndarray) -> Dict[str, Any]:
    nan_count = int(np.sum(~np.isfinite(processed_X)))
    missing_fraction = float(np.mean(~np.isfinite(processed_X))) if processed_X.size else 0.0
    amplitudes = np.nanmax(processed_X, axis=1) - np.nanmin(processed_X, axis=1)
    raw_amplitudes = np.nanmax(raw_X, axis=1) - np.nanmin(raw_X, axis=1)
    flat_count = int(np.sum(amplitudes < 1e-9))
    negative_fraction = float(np.mean(processed_X < 0)) if processed_X.size else 0.0
    median_ratio = float(np.nanmedian(amplitudes / np.maximum(raw_amplitudes, 1e-12))) if amplitudes.size else 0.0
    warnings: List[str] = []
    if nan_count:
        warnings.append("После обработки появились NaN или бесконечные значения. Проверьте параметры предобработки.")
    if flat_count:
        warnings.append("После обработки часть спектров имеет очень малую амплитуду. Проверьте параметры нормализации и коррекции базовой линии.")
    if median_ratio < 0.02:
        warnings.append("Сглаживание или коррекция базовой линии выглядят слишком жёсткими: амплитуда спектров сильно уменьшилась.")
    if negative_fraction > 0.25:
        warnings.append("После коррекции базовой линии много отрицательных значений. Это может быть нормально, но проверьте параметры baseline.")
    return {
        "nan_count": nan_count,
        "missing_fraction": missing_fraction,
        "flat_spectra_count": flat_count,
        "negative_fraction": negative_fraction,
        "median_amplitude_ratio": median_ratio,
        "warnings": warnings,
    }


def apply_preprocessing(dataset: SpectralDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    dataset.metadata["preprocessing_status"] = "running"
    processed = preprocess_matrix(dataset, config)
    dataset.processed_X = processed["X"]
    dataset.processed_axis = processed["axis"]
    dataset.preprocessing_config = processed["config"]
    quality = preprocessing_quality(dataset.X, processed["X"])
    dataset.preprocessing_warnings = processed["warnings"] + quality["warnings"]
    dataset.metadata["preprocessing_status"] = "applied"
    return {
        "status": "ok",
        "warnings": dataset.preprocessing_warnings,
        "quality": quality,
        "preprocessing_summary": preprocessing_summary(processed["config"], processed["axis"], dataset.axis),
        "processed_summary": dataset.summary(version="processed"),
    }


def preprocessing_status(dataset: SpectralDataset) -> Dict[str, Any]:
    return {
        "status": "ok",
        "preprocessing_status": dataset.metadata.get("preprocessing_status", "not_applied"),
        "has_processed": dataset.processed_X is not None,
        "warnings": dataset.preprocessing_warnings,
        "preprocessing_config": dataset.preprocessing_config or {},
    }


async def upload_files_to_memory(files: Sequence[Any]) -> List[UploadedFileData]:
    items: List[UploadedFileData] = []
    for file in files:
        items.append(UploadedFileData(filename=file.filename or "unknown", content=await file.read()))
    return items
