from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


SUPPORTED_TABULAR_FORMATS = {"csv", "txt", "esp", "xlsx", "xls"}
UNSUPPORTED_BINARY_FORMATS = {
    "wdf": "Формат WDF требует отдельного бинарного парсера Renishaw/WiRE и сейчас не реализован.",
    "spc": "Формат SPC требует отдельного бинарного парсера Galactic/SPC и сейчас не реализован.",
}


class SpectrumValidationError(ValueError):
    """Raised when input spectral file is invalid."""


@dataclass
class SpectrumDataset:
    x: np.ndarray
    spectral_axis: Optional[np.ndarray]
    sample_names: List[str]
    source_format: str
    filename: str = ""
    file_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.x.shape[1])

    def to_spectrum_records(self) -> List[Dict[str, Any]]:
        axis = self.spectral_axis if self.spectral_axis is not None else np.arange(self.feature_count, dtype=float)
        records: List[Dict[str, Any]] = []
        for idx, row in enumerate(self.x):
            records.append(
                {
                    "x": axis.astype(float).tolist(),
                    "y": np.asarray(row, dtype=float).tolist(),
                    "filename": self.filename,
                    "format": self.file_format or self.source_format,
                    "sample_name": self.sample_names[idx] if idx < len(self.sample_names) else f"spectrum_{idx + 1}",
                    "metadata": dict(self.metadata),
                }
            )
        return records


def detect_format(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if not suffix:
        raise SpectrumValidationError("Не удалось определить формат файла: отсутствует расширение.")
    return suffix


def _decode_bytes(raw: bytes) -> str:
    if not raw:
        raise SpectrumValidationError("Файл пустой.")
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SpectrumValidationError("Не удалось декодировать файл. Поддерживаются UTF-8/CP1251/LATIN-1.")


def _text_to_numeric_rows(content: str) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", ";")):
            continue

        normalized = stripped.replace(",", ".")
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", normalized)
        if len(values) < 2:
            continue
        try:
            rows.append([float(value) for value in values])
        except ValueError:
            continue

    return rows


def _read_text_table(raw: bytes, extension: str) -> pd.DataFrame:
    content = _decode_bytes(raw)
    parse_attempts = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": "\t", "decimal": "."},
        {"sep": r"\s+", "decimal": "."},
    ]
    for options in parse_attempts:
        try:
            parsed = pd.read_csv(
                io.StringIO(content),
                header=None,
                comment="#",
                engine="python",
                **options,
            )
        except Exception:
            continue
        numeric_df = parsed.apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if numeric_df.shape[0] >= 2 and numeric_df.shape[1] >= 2 and not numeric_df.isna().any().any():
            return numeric_df.astype(float)

    rows = _text_to_numeric_rows(content)
    if not rows:
        raise SpectrumValidationError(
            f"Не найдено числовых строк в {extension.upper()} файле. "
            "Ожидаются пары x y или матрица интенсивностей."
        )

    max_len = max(len(row) for row in rows)
    if max_len < 2:
        raise SpectrumValidationError("Файл должен содержать минимум два числовых столбца.")

    normalized_rows = [row + [np.nan] * (max_len - len(row)) for row in rows]
    df = pd.DataFrame(normalized_rows).dropna(axis=1, how="all")
    if df.isna().any().any():
        raise SpectrumValidationError("В числовой матрице обнаружены строки с разным количеством столбцов.")
    return df.astype(float)


def _read_excel_table(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(raw), header=None)
    except ImportError as exc:
        raise SpectrumValidationError("Для чтения XLSX нужен пакет openpyxl. Добавьте его в окружение.") from exc
    except Exception as exc:
        raise SpectrumValidationError(f"Не удалось прочитать XLSX файл: {exc}") from exc

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise SpectrumValidationError("XLSX файл пустой или не содержит табличных данных.")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric_df.empty:
        raise SpectrumValidationError("XLSX файл не содержит числовой матрицы.")
    if numeric_df.isna().any().any():
        raise SpectrumValidationError("XLSX содержит нечисловые значения внутри числовой матрицы.")
    return numeric_df.astype(float)


def _is_axis_like(values: np.ndarray) -> bool:
    if values.ndim != 1 or values.size < 3:
        return False
    diffs = np.diff(values.astype(float))
    return bool(np.all(np.isfinite(values)) and np.all(diffs != 0) and (np.all(diffs > 0) or np.all(diffs < 0)))


def _validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise SpectrumValidationError("Ожидалась двумерная числовая матрица.")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise SpectrumValidationError("Недостаточно данных: требуется минимум 2 строки и 1 столбец.")
    if not np.all(np.isfinite(matrix)):
        raise SpectrumValidationError("Матрица содержит NaN или бесконечные значения.")


def _from_axis_plus_intensities(
    matrix: np.ndarray,
    *,
    filename: str,
    file_format: str,
    metadata: Dict[str, Any],
) -> SpectrumDataset:
    axis = matrix[:, 0].astype(float)
    intensity = matrix[:, 1:].astype(float)
    x = intensity.T
    names = [f"spectrum_{idx + 1}" for idx in range(x.shape[0])]
    return SpectrumDataset(
        x=x,
        spectral_axis=axis,
        sample_names=names,
        source_format="axis+intensity",
        filename=filename,
        file_format=file_format,
        metadata=metadata,
    )


def _from_pure_matrix(
    matrix: np.ndarray,
    *,
    filename: str,
    file_format: str,
    metadata: Dict[str, Any],
) -> SpectrumDataset:
    x = matrix.astype(float)
    names = [f"spectrum_{idx + 1}" for idx in range(x.shape[0])]
    return SpectrumDataset(
        x=x,
        spectral_axis=np.arange(x.shape[1], dtype=float),
        sample_names=names,
        source_format="intensity_matrix",
        filename=filename,
        file_format=file_format,
        metadata=metadata,
    )


def parse_spectrum_file(raw: bytes, filename: str) -> SpectrumDataset:
    extension = detect_format(filename)
    if extension in UNSUPPORTED_BINARY_FORMATS:
        raise SpectrumValidationError(UNSUPPORTED_BINARY_FORMATS[extension])
    if extension not in SUPPORTED_TABULAR_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TABULAR_FORMATS | set(UNSUPPORTED_BINARY_FORMATS)))
        raise SpectrumValidationError(f"Формат .{extension} не поддерживается. Известные форматы: {supported}.")

    if extension in {"xlsx", "xls"}:
        df = _read_excel_table(raw)
    else:
        df = _read_text_table(raw, extension)

    matrix = df.to_numpy(dtype=float)
    _validate_matrix(matrix)

    metadata = {
        "original_filename": filename,
        "extension": extension,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
    }

    if matrix.shape[1] > 1 and _is_axis_like(matrix[:, 0]):
        dataset = _from_axis_plus_intensities(
            matrix,
            filename=filename,
            file_format=extension,
            metadata=metadata,
        )
    else:
        dataset = _from_pure_matrix(
            matrix,
            filename=filename,
            file_format=extension,
            metadata=metadata,
        )

    if dataset.sample_count < 1 or dataset.feature_count < 2:
        raise SpectrumValidationError("Матрица должна содержать минимум 1 спектр и 2 признака.")

    return dataset


def parse_target_values(raw_targets: Optional[str], expected_len: int) -> np.ndarray:
    if raw_targets is None:
        raise SpectrumValidationError("Для выбранной модели требуются целевые значения.")

    values = [v.strip() for v in re.split(r"[\n,;]+", raw_targets) if v.strip()]
    if len(values) != expected_len:
        raise SpectrumValidationError(
            f"Количество target-значений ({len(values)}) не совпадает с числом спектров ({expected_len})."
        )

    return np.array(values, dtype=object)
