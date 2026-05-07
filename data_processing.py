from __future__ import annotations

import re
from typing import List, Tuple, Sequence, Union, Dict, Any

import numpy as np
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
from scipy.sparse.linalg import spsolve
from scipy import sparse

ArrayLike = Union[np.ndarray, Sequence[float]]


def calculate_boxplot_stats(amplitudes_list: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    """
    Рассчитывает статистики для box-plot по наборам амплитуд.
    :param amplitudes_list: последовательность массивов амплитуд
    :return: список словарей с q1, median, q3, нижней/верхней границей и выбросами
    """
    if not amplitudes_list:
        raise ValueError("amplitudes_list must contain at least one series")

    boxplot_stats: List[Dict[str, Any]] = []
    for amplitudes in amplitudes_list:
        array = np.asarray(amplitudes, dtype=float)
        if array.size == 0:
            continue

        q1 = np.percentile(array, 25)
        median = np.percentile(array, 50)
        q3 = np.percentile(array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = array[(array < lower_bound) | (array > upper_bound)]

        boxplot_stats.append({
            'q1': float(q1),
            'median': float(median),
            'q3': float(q3),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outliers': outliers.tolist()
        })

    if not boxplot_stats:
        raise ValueError("No valid amplitude data for boxplot statistics")

    return boxplot_stats


def parse_txt_file(content: str) -> Tuple[List[float], List[float]]:
    """
    Парсит текстовый .txt с парами значений: частота амплитуда.
    Допускаются пробелы в качестве разделителей, запятая/точка как десятичный разделитель.
    """
    frequencies: List[float] = []
    amplitudes: List[float] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = stripped.replace(',', '.')
        parts = cleaned.split()
        if len(parts) < 2:
            continue
        try:
            freq, ampl = map(float, parts[:2])
        except ValueError:
            continue
        frequencies.append(freq)
        amplitudes.append(ampl)

    if not frequencies:
        raise ValueError("Не найдено ни одной валидной пары в .txt файле")

    return frequencies, amplitudes


def parse_csv_file(content: str) -> Tuple[List[float], List[float]]:
    """
    Парсит .csv со строками "freq,ampl" или "freq;ampl". Десятичная запятая поддерживается.
    """
    frequencies: List[float] = []
    amplitudes: List[float] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        delimiter = ',' if ',' in stripped else (';' if ';' in stripped else None)
        if delimiter is None:
            continue
        parts = [part.replace(',', '.') for part in stripped.split(delimiter)]
        if len(parts) < 2:
            continue
        try:
            freq, ampl = map(float, parts[:2])
        except ValueError:
            continue
        frequencies.append(freq)
        amplitudes.append(ampl)

    if not frequencies:
        raise ValueError("Не найдено ни одной валидной пары в .csv файле")

    return frequencies, amplitudes


def parse_esp_file(file_content: str) -> Tuple[List[float], List[float]]:
    """
    Парсит .esp: строки из двух чисел (частота амплитуда) через пробел, строки с # игнорируются.
    :return: кортеж (frequencies, amplitudes)
    """
    frequencies: List[float] = []
    amplitudes: List[float] = []

    for line in file_content.splitlines():
        if line.startswith("#"):
            continue
        try:
            freq, ampl = map(float, line.split())
            frequencies.append(freq)
            amplitudes.append(ampl)
        except ValueError:
            continue

    return frequencies, amplitudes


def _parse_numeric_pairs_generic(content: str) -> Tuple[List[float], List[float]]:
    """Запасной парсер: находит первые два числа в строке (частота и амплитуда)."""
    frequencies: List[float] = []
    amplitudes: List[float] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        normalized = stripped.replace(',', '.')
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
        if len(numbers) < 2:
            continue
        try:
            freq = float(numbers[0])
            ampl = float(numbers[1])
        except ValueError:
            continue
        frequencies.append(freq)
        amplitudes.append(ampl)

    if not frequencies:
        raise ValueError("Не найдено числовых пар (частота амплитуда) в содержимом")

    return frequencies, amplitudes


def parse_any_spectral_file(content: str) -> Tuple[List[float], List[float]]:
    parsers = (parse_csv_file, parse_txt_file, parse_esp_file)
    last_error: Exception | None = None

    for parser in parsers:
        try:
            frequencies, amplitudes = parser(content)
            if frequencies and amplitudes:
                return frequencies, amplitudes
        except ValueError as exc:
            last_error = exc

    try:
        return _parse_numeric_pairs_generic(content)
    except ValueError as exc:
        last_error = exc

    raise ValueError(str(last_error) if last_error else 'Не удалось распарсить спектральный файл')


def baseline_als(amplitudes: ArrayLike, lam: float, p: float, niter: int = 10) -> np.ndarray:
    """
    Оценка базовой линии методом ALS.
    - lam: параметр сглаживания (>0)
    - p: асимметрия (0 < p < 1)
    - niter: число итераций
    """
    if lam <= 0:
        raise ValueError("Параметр lam должен быть положительным")
    if not (0 < p < 1):
        raise ValueError("p должен быть в пределах (0, 1)")

    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.array(amplitudes)

    if amplitudes.size == 0:
        raise ValueError("Массив амплитуд пуст")

    L = len(amplitudes)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * amplitudes)
        w = p * (amplitudes > z) + (1 - p) * (amplitudes < z)
    return z  # type: ignore[name-defined]

def baseline_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    n_iter: int = 10,
    asymmetry: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Полиномиальная аппроксимация базовой линии (robust reweighted polyfit).

    Возвращает:
    - y_corrected: y - baseline
    - baseline: оценка базовой линии
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("baseline_polyfit: x и y должны быть 1D массивами одинаковой длины")

    if degree < 1:
        raise ValueError("baseline_polyfit: degree должен быть >= 1")

    if n_iter < 1:
        n_iter = 1

    if not (0 < asymmetry <= 1):
        raise ValueError("baseline_polyfit: asymmetry должен быть в (0; 1]")

    # Нормируем x в [-1; 1] для устойчивости polyfit
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        baseline = np.full_like(y, y.mean())
        return y - baseline, baseline

    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    w = np.ones_like(y, dtype=float)

    for _ in range(int(n_iter)):
        coeffs = np.polyfit(x_norm, y, deg=degree, w=w)
        baseline = np.polyval(coeffs, x_norm)
        r = y - baseline

        # Точки выше baseline считаем "пиками" и сильно занижаем их вклад
        w = np.where(r > 0, asymmetry, 1.0)

    coeffs = np.polyfit(x_norm, y, deg=degree, w=w)
    baseline = np.polyval(coeffs, x_norm)
    y_corrected = y - baseline
    return y_corrected, baseline




def _rolling_min_1d(y: np.ndarray, r: int) -> np.ndarray:
    """Наивный rolling minimum (окно 2r+1). O(n*r). Для n~2-4k работает быстро."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty_like(y)
    for i in range(n):
        a = max(0, i - r)
        b = min(n, i + r + 1)
        out[i] = np.min(y[a:b])
    return out

def _rolling_max_1d(y: np.ndarray, r: int) -> np.ndarray:
    """Наивный rolling maximum (окно 2r+1). O(n*r)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty_like(y)
    for i in range(n):
        a = max(0, i - r)
        b = min(n, i + r + 1)
        out[i] = np.max(y[a:b])
    return out

def baseline_rolling_ball(
    x: np.ndarray,
    y: np.ndarray,
    radius: float = 50.0,
    radius_units: str = "x",  # "x" (в единицах оси, см^-1) или "points"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling Ball baseline (морфологический метод) для 1D спектров.

    Практическая реализация:
    - В 1D Rolling Ball обычно приближают морфологическим "открытием":
      erosion (rolling min) → dilation (rolling max)
      baseline = max(min(y, r), r)

    Параметры:
    - radius: радиус "шара" (по сути ширина структурного элемента).
      Чем больше — тем более сглаженный (медленный) фон.
    - radius_units:
        "x"      — radius задан в единицах оси x (например, см^-1)
        "points" — radius задан в точках массива (индексах)

    Возвращает:
    - y_corrected = y - baseline
    - baseline
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("baseline_rolling_ball: x и y должны быть 1D массивами одинаковой длины")

    if radius <= 0:
        baseline = np.full_like(y, np.min(y))
        return y - baseline, baseline

    # Перевод радиуса в точки
    if radius_units == "points":
        r_pts = int(round(radius))
    else:
        # radius задан в единицах x (см^-1). Берём медианный шаг.
        dx = np.median(np.diff(x))
        if dx <= 0 or not np.isfinite(dx):
            # fallback: считаем, что x равномерный по индексам
            r_pts = int(round(radius))
        else:
            r_pts = int(round(radius / dx))

    r_pts = max(1, r_pts)

    # Морфологическое открытие: erosion → dilation
    eroded = _rolling_min_1d(y, r_pts)
    baseline = _rolling_max_1d(eroded, r_pts)

    y_corrected = y - baseline
    return y_corrected, baseline


def smooth_signal(amplitudes: ArrayLike, window_length: int, polyorder: int) -> np.ndarray:
    """
    Сглаживание по Савицкому-Голею.
    - window_length: длина окна фильтра
    - polyorder: порядок полинома
    """
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.array(amplitudes)

    if len(amplitudes) < window_length:
        raise ValueError("Длина сигнала меньше длины окна фильтра")
    if polyorder >= window_length:
        raise ValueError("Порядок полинома должен быть меньше длины окна")

    return savgol_filter(amplitudes, window_length, polyorder)


def whittaker_smooth(y: np.ndarray, lmbd: float = 1000.0, d: int = 2) -> np.ndarray:
    """
    Whittaker smoother (Eilers): решает задачу
    min_z ||y - z||^2 + λ ||Δ^d z||^2

    y     — сигнал
    λ     — жёсткость сглаживания (больше => сильнее сглаживание)
    d     — порядок разностного оператора (обычно 2)
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    if d < 1:
        d = 1

    # Чтобы не писать велосипед с плотными матрицами — используем scipy.sparse
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception as e:
        raise ImportError("Для Whittaker smoothing нужен scipy.sparse") from e

    E = sparse.eye(n, format="csc")
    # разностная матрица Δ (d раз)
    D = E[1:] - E[:-1]
    for _ in range(d - 1):
        D = D[1:] - D[:-1]

    A = E + (lmbd * (D.T @ D))
    z = spsolve(A, y)
    return np.asarray(z)

def gaussian_smooth(y: np.ndarray, sigma: float = 2.0, mode: str = "nearest") -> np.ndarray:
    """
    Гауссово сглаживание 1D (свертка с гауссовым ядром).
    sigma — стандартное отклонение (в точках массива).
    mode  — обработка краёв: 'nearest', 'reflect', 'mirror', 'constant', 'wrap'
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    sigma = float(sigma)
    if sigma <= 0:
        return y.copy()

    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma, mode=mode)




def normalize_snv(amplitudes: ArrayLike) -> np.ndarray:
    """Нормализация SNV (Standard Normal Variate)."""
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.array(amplitudes)

    if amplitudes.size == 0:
        raise ValueError("Массив амплитуд пуст")

    mean = np.mean(amplitudes)
    std = np.std(amplitudes)
    if std == 0:
        raise ValueError("Стандартное отклонение равно нулю — нормализация невозможна")
    return (amplitudes - mean) / std

def normalize_max(amplitudes: ArrayLike) -> np.ndarray:
    """
    Нормировка на максимум: делим спектр на его максимальную амплитуду.

    Для случаев после baseline, когда могут быть отрицательные значения:
    - если max(y) <= 0, используем max(|y|), чтобы избежать деления на <=0.
    """
    y = np.asarray(amplitudes, dtype=float)
    if y.size == 0:
        return y

    max_val = float(np.max(y))
    if max_val <= 0:
        max_val = float(np.max(np.abs(y)))

    if max_val == 0:
        return y

    return y / max_val

def normalize_minmax(amplitudes):
    """
    Min–Max нормализация спектра в диапазон [0, 1]
    """
    y = np.asarray(amplitudes, dtype=float)
    if y.size == 0:
        return y

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if y_max == y_min:
        return np.zeros_like(y)

    return (y - y_min) / (y_max - y_min)


def find_signal_peaks(y, width=1, prominence=1, height=None, distance=None):
    kwargs = {
        "width": width,
        "prominence": prominence,
    }
    if height is not None:
        kwargs["height"] = height
    if distance is not None:
        kwargs["distance"] = distance

    peaks, props = find_peaks(y, **kwargs)
    return peaks, props


def find_peaks_findpeaks(
    amplitudes: ArrayLike,
    width: float = 1,
    prominence: float = 1,
    height: float | None = None,
    distance: int | None = None,
    threshold: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(amplitudes, dtype=float)
    if y.size == 0:
        return np.array([]), {}

    peaks, props = find_peaks(
        y,
        width=width,
        prominence=prominence,
        height=height,
        distance=distance,
        threshold=threshold,
    )
    return peaks, props


def find_peaks_cwt_method(y: np.ndarray, widths: list[int]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    widths = [int(w) for w in widths if int(w) > 0]
    if len(widths) == 0:
        widths = [3, 5, 7, 9, 11]
    peaks = find_peaks_cwt(y, widths)
    return np.asarray(peaks, dtype=int)


def find_peaks_derivative_method(
    amplitudes: ArrayLike,
    min_prominence: float = 1.0,
    min_distance: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Производная: кандидат — точка, где dy меняет знак + -> -
    Потом фильтруем по "выраженности" (минимальная prominence).
    """
    y = np.asarray(amplitudes, dtype=float)
    if y.size < 3:
        return np.array([]), {}

    dy = np.diff(y)
    # пик: dy[i-1] > 0 и dy[i] <= 0 (переход через максимум)
    cand = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if cand.size == 0:
        return np.array([]), {}

    # Фильтрация по расстоянию (простейшая)
    cand_sorted = cand[np.argsort(y[cand])[::-1]]  # по высоте
    picked = []
    for idx in cand_sorted:
        if all(abs(idx - p) >= max(1, int(min_distance)) for p in picked):
            picked.append(int(idx))
    picked = np.array(sorted(picked), dtype=int)

    # Фильтрация по выраженности через find_peaks (используем как оценщик prominence)
    peaks, props = find_peaks(y, prominence=min_prominence, distance=max(1, int(min_distance)))
    # оставляем пересечение
    final = np.intersect1d(picked, peaks)

    return final, props






from scipy.signal import savgol_filter, find_peaks

def find_peaks_second_derivative_method(
    amplitudes: ArrayLike,
    *,
    min_prominence: float = 1.0,
    min_distance: int = 1,
    sg_window: int = 21,
    sg_poly: int = 3,
    min_curvature: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(amplitudes, dtype=float)
    n = y.size
    if n < 7:
        return np.array([], dtype=int), {}

    # окно SG: нечётное, > poly
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window <= sg_poly:
        sg_window = sg_poly + 3
        if sg_window % 2 == 0:
            sg_window += 1
    if sg_window >= n:
        sg_window = n - (1 if n % 2 == 0 else 0)
        if sg_window < 5:
            return np.array([], dtype=int), {}

    y_smooth = savgol_filter(y, window_length=sg_window, polyorder=sg_poly, mode="interp")

    # 1-я производная: кандидаты на вершины (+ -> -)
    dy = np.diff(y_smooth)
    cand = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if cand.size == 0:
        return np.array([], dtype=int), {}

    # 2-я производная: должна быть < 0 (вершина)
    d2y = np.diff(y_smooth, n=2)  # d2y[j] соответствует точке j+1
    cand = cand[(cand >= 1) & (cand <= n - 2)]
    cand = cand[d2y[cand - 1] < 0]
    if cand.size == 0:
        return np.array([], dtype=int), {}

    # Порог кривизны (если не задан — авто)
    if min_curvature is None:
        min_d2 = np.min(d2y[cand - 1])  # отрицательный
        min_curvature = 0.05 * abs(min_d2) if min_d2 < 0 else 0.0

    if min_curvature > 0:
        cand = cand[np.abs(d2y[cand - 1]) >= float(min_curvature)]
        if cand.size == 0:
            return np.array([], dtype=int), {}

    # Финальная фильтрация prominence/distance на сглаженном сигнале
    peaks, props = find_peaks(
        y_smooth,
        prominence=float(min_prominence),
        distance=max(1, int(min_distance)),
    )

    final = np.intersect1d(cand, peaks).astype(int)
    props = dict(props)
    props["sg_window_used"] = sg_window
    props["sg_poly_used"] = sg_poly
    props["min_curvature_used"] = float(min_curvature)
    return final, props





from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
from scipy.signal import find_peaks_cwt

Array1D = Union[Sequence[float], np.ndarray]

def find_peaks_cwt_method(
    amplitudes: Array1D,
    widths: Sequence[int] = (3, 5, 7, 9, 11),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = np.asarray(amplitudes, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int), {"widths": list(widths)}

    w = [int(v) for v in widths if int(v) > 0]
    if not w:
        w = [3, 5, 7, 9, 11]

    peaks = np.asarray(find_peaks_cwt(y, w), dtype=int)
    return peaks, {"widths": w}







def filter_frequency_range(
    frequencies: ArrayLike,
    amplitudes: ArrayLike,
    min_freq: float,
    max_freq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Фильтрация диапазона частот [min_freq, max_freq].
    Возвращает отфильтрованные массивы той же длины.
    """
    if not isinstance(frequencies, np.ndarray):
        frequencies = np.array(frequencies)
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.array(amplitudes)

    if frequencies.size == 0 or amplitudes.size == 0:
        raise ValueError("Пустые массивы частот или амплитуд")

    if min_freq > max_freq:
        raise ValueError("min_freq не может быть больше max_freq")

    mask = (frequencies >= min_freq) & (frequencies <= max_freq)

    if not np.any(mask):
        raise ValueError(f"Нет точек в диапазоне от {min_freq} до {max_freq}")

    filtered_frequencies = frequencies[mask]
    filtered_amplitudes = amplitudes[mask]

    return filtered_frequencies, filtered_amplitudes


def calculate_mean_std(amplitudes_list: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Среднее и стандартное отклонение по наборам амплитуд (по оси 0)."""
    amplitudes_array = np.array(amplitudes_list)
    mean_amplitude = np.mean(amplitudes_array, axis=0)
    std_amplitude = np.std(amplitudes_array, axis=0)

    return mean_amplitude, std_amplitude


def format_spectral_data(frequencies: Sequence[float], amplitudes: Sequence[float]) -> str:
    """Форматирует спектральные данные в табличный текст (freq\tampl)."""
    if len(frequencies) != len(amplitudes):
        raise ValueError("Длины массивов частот и амплитуд не совпадают")

    lines: List[str] = []
    for freq, ampl in zip(frequencies, amplitudes):
        lines.append(f"{freq:.2f}\t{ampl:.6f}")

    return "\n".join(lines)


def calculate_moving_average(amplitudes: ArrayLike, window_size: int) -> np.ndarray:
    """
    Скользящее среднее по окну указанной длины.
    :param amplitudes: массив амплитуд
    :param window_size: размер окна (целое > 0)
    :return: массив скользящего среднего
    """
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.array(amplitudes)

    if amplitudes.size == 0:
        raise ValueError("Массив амплитуд пуст")

    if window_size <= 0:
        raise ValueError("Размер окна должен быть положительным")

    if window_size > len(amplitudes):
        raise ValueError("Размер окна не может превышать длину сигнала")

    moving_avg = np.convolve(amplitudes, np.ones(window_size) / window_size, mode='same')

    return moving_avg

