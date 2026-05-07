from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


class SpectrumValidationError(ValueError):
    """Raised when input spectral file is invalid."""


@dataclass
class SpectrumDataset:
    x: np.ndarray
    spectral_axis: Optional[np.ndarray]
    sample_names: List[str]
    source_format: str

    @property
    def sample_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.x.shape[1])


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SpectrumValidationError("Не удалось декодировать файл. Поддерживаются UTF-8/CP1251/LATIN-1.")


def _read_table(content: str, extension: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(content), sep=None, engine="python", header=None)
    except Exception as exc:
        raise SpectrumValidationError(f"Не удалось распарсить {extension.upper()} файл: {exc}") from exc

    if df.empty:
        raise SpectrumValidationError("Файл пустой.")

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise SpectrumValidationError("Файл не содержит числовых данных.")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().all().all():
        raise SpectrumValidationError("Файл не содержит валидной числовой матрицы.")

    if numeric_df.isna().any().any():
        raise SpectrumValidationError("Обнаружены нечисловые значения внутри матрицы спектров.")

    return numeric_df


def _is_axis_like(values: np.ndarray) -> bool:
    if values.ndim != 1 or values.size < 3:
        return False
    diffs = np.diff(values)
    return bool(np.all(diffs != 0) and (np.all(diffs > 0) or np.all(diffs < 0)))


def _from_axis_plus_intensities(matrix: np.ndarray) -> SpectrumDataset:
    axis = matrix[:, 0]
    intensity = matrix[:, 1:]
    x = intensity.T
    names = [f"spectrum_{idx + 1}" for idx in range(x.shape[0])]
    return SpectrumDataset(
        x=x.astype(float),
        spectral_axis=axis.astype(float),
        sample_names=names,
        source_format="axis+intensity",
    )


def _from_pure_matrix(matrix: np.ndarray) -> SpectrumDataset:
    x = matrix.astype(float)
    if x.ndim != 2:
        raise SpectrumValidationError("Ожидалась двумерная матрица интенсивностей.")
    names = [f"spectrum_{idx + 1}" for idx in range(x.shape[0])]
    return SpectrumDataset(
        x=x,
        spectral_axis=np.arange(x.shape[1], dtype=float),
        sample_names=names,
        source_format="intensity_matrix",
    )


def parse_spectrum_file(raw: bytes, filename: str) -> SpectrumDataset:
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"csv", "txt"}:
        raise SpectrumValidationError("Поддерживаются только файлы .csv и .txt")

    content = _decode_bytes(raw)
    df = _read_table(content, extension)
    matrix = df.to_numpy(dtype=float)

    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise SpectrumValidationError("Недостаточно данных: требуется минимум 2 строки и 1 столбец.")

    if matrix.shape[1] > 1 and _is_axis_like(matrix[:, 0]):
        dataset = _from_axis_plus_intensities(matrix)
    else:
        dataset = _from_pure_matrix(matrix)

    if dataset.sample_count < 1 or dataset.feature_count < 2:
        raise SpectrumValidationError("Матрица должна содержать минимум 1 спектр и 2 признака.")

    return dataset


def parse_target_values(raw_targets: Optional[str], expected_len: int) -> np.ndarray:
    if raw_targets is None:
        raise SpectrumValidationError("Для выбранной модели требуются целевые значения.")

    values = [v.strip() for v in raw_targets.replace("\n", ",").split(",") if v.strip()]
    if len(values) != expected_len:
        raise SpectrumValidationError(
            f"Количество target-значений ({len(values)}) не совпадает с числом спектров ({expected_len})."
        )

    return np.array(values, dtype=object)
