from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analysis.dataset_importer import UploadedFileData, import_dataset


def _file_data(path: Path) -> UploadedFileData:
    if not path.exists():
        raise AssertionError(f"File not found: {path}")
    return UploadedFileData(filename=path.name, content=path.read_bytes())


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_excel(path: Path) -> None:
    dataset = import_dataset(
        [_file_data(path)],
        {
            "layout": "excel_columns",
            "target_source": "sheet_name",
            "sheet_mode": "sheet_as_class",
            "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
        },
    )
    classes = set(dataset.y or [])
    _assert(dataset.X.shape == (100, 2000), f"Expected X shape (100, 2000), got {dataset.X.shape}")
    _assert(dataset.axis.shape[0] == 2000, f"Expected 2000 axis points, got {dataset.axis.shape[0]}")
    _assert(classes == {"health", "heart disease"}, f"Expected classes health/heart disease, got {sorted(classes)}")
    _assert(dataset.y is not None and len(dataset.y) == 100, "Expected 100 target values")
    _assert(np.all(np.isfinite(dataset.X)), "Excel X contains NaN/Inf")
    _assert(np.all(np.isfinite(dataset.axis)), "Excel axis contains NaN/Inf")
    print(f"[OK] Excel import: samples={dataset.X.shape[0]}, features={dataset.X.shape[1]}, classes={sorted(classes)}")


def check_zip(path: Path) -> None:
    dataset = import_dataset(
        [_file_data(path)],
        {
            "layout": "txt_folder",
            "target_source": "none",
            "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
        },
    )
    _assert(dataset.X.shape[0] == 56, f"Expected 56 spectra, got {dataset.X.shape[0]}")
    _assert(1800 <= dataset.X.shape[1] <= 2200, f"Expected about 2000 features, got {dataset.X.shape[1]}")
    _assert(780 <= float(np.nanmin(dataset.axis)) <= 810, f"Unexpected axis min: {np.nanmin(dataset.axis)}")
    _assert(930 <= float(np.nanmax(dataset.axis)) <= 970, f"Unexpected axis max: {np.nanmax(dataset.axis)}")
    metadata: List[dict[str, Any]] = dataset.sample_metadata or []
    _assert(len(metadata) == 56, f"Expected sample metadata for 56 spectra, got {len(metadata)}")
    slit_values = sorted({item.get("slit_width_um") for item in metadata if item.get("slit_width_um") is not None})
    grating_values = sorted({item.get("grating_lines_mm") for item in metadata if item.get("grating_lines_mm") is not None})
    _assert(grating_values == [600, 1200], f"Expected grating values [600, 1200], got {grating_values}")
    _assert(0 in slit_values and 1000 in slit_values, f"Expected slit values to include 0 and 1000, got {slit_values}")
    print(
        f"[OK] ZIP ASC import: samples={dataset.X.shape[0]}, features={dataset.X.shape[1]}, "
        f"gratings={grating_values}, slit_min={min(slit_values)}, slit_max={max(slit_values)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test existing dataset importers with control Excel and ZIP/ASC files.")
    parser.add_argument("--excel", required=True, help="Path to Raman_krov_SSZ-zdorovye.xlsx")
    parser.add_argument("--zip", required=True, help="Path to Raman_slit_dependence_16_03_2026.zip")
    args = parser.parse_args()

    failed = False
    for label, func, raw_path in [
        ("Excel import", check_excel, args.excel),
        ("ZIP ASC import", check_zip, args.zip),
    ]:
        try:
            func(Path(raw_path))
        except Exception as exc:
            failed = True
            print(f"[FAIL] {label}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
