from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_processing import (  # noqa: E402
    baseline_als,
    find_peaks_findpeaks,
    gaussian_smooth,
    normalize_snv,
)


def main() -> int:
    x = np.linspace(400, 1800, 600)
    baseline = 0.0004 * (x - 400)
    peak_1 = 2.5 * np.exp(-0.5 * ((x - 735) / 18) ** 2)
    peak_2 = 1.6 * np.exp(-0.5 * ((x - 1320) / 25) ** 2)
    y = baseline + peak_1 + peak_2 + 0.03 * np.sin(x / 25)

    corrected = y - baseline_als(y, lam=1000, p=0.001)
    smoothed = gaussian_smooth(corrected, sigma=2.0)
    normalized = normalize_snv(smoothed)
    peaks, _props = find_peaks_findpeaks(normalized, prominence=1.0, width=1)

    print("Demo preprocessing")
    print("points:", len(x))
    print("detected_peaks:", len(peaks))
    print("peak_positions:", [round(float(x[idx]), 2) for idx in peaks[:10]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
