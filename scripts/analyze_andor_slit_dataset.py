from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from experiment_utils import fail_missing_input, save_csv, simple_line_png


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def iter_asc_files(input_path: Path) -> Iterable[Tuple[str, bytes]]:
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*.asc")):
            yield path.name, path.read_bytes()
    elif input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".asc") and not name.endswith("/"):
                    yield Path(name).name, archive.read(name)
    elif input_path.suffix.lower() == ".asc":
        yield input_path.name, input_path.read_bytes()
    else:
        raise SystemExit("Input must be a directory, ZIP archive or a single .asc file.")


def parse_filename_params(file_name: str) -> Dict[str, Any]:
    text = file_name.lower().replace(",", ".")
    slit = None
    grating = None
    patterns = [
        r"slit[_\s-]*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:um|µm|mkm|micron)",
        r"щель[_\s-]*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            slit = float(match.group(1))
            break
    grating_patterns = [
        r"grating[_\s-]*(\d{3,4})",
        r"(\d{3,4})\s*(?:l/mm|lines|штр|g/mm)",
        r"реш[её]тк[аи][_\s-]*(\d{3,4})",
    ]
    for pattern in grating_patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if value in {600, 1200} or 100 <= value <= 3600:
                grating = value
                break
    return {"slit_width_um": slit, "grating_lines_mm": grating}


def parse_metadata(lines: List[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    text = "\n".join(lines)
    patterns = {
        "exposure_time_s": r"exposure(?:\s*time)?[^0-9+-]*([-+]?\d+(?:[.,]\d+)?)\s*(ms|s|sec|seconds)?",
        "accumulations": r"accumulations?[^0-9+-]*(\d+)",
        "detector": r"detector\s*[:=]\s*(.+)",
        "temperature_c": r"temperature[^0-9+-]*([-+]?\d+(?:[.,]\d+)?)",
        "grating": r"grating\s*[:=]\s*(.+)",
        "wavelength_range": r"wavelength\s*range\s*[:=]\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            metadata[key] = None
            continue
        value = match.group(1).strip()
        if key == "exposure_time_s":
            number = float(value.replace(",", "."))
            unit = (match.group(2) or "s").lower()
            metadata[key] = number / 1000.0 if unit == "ms" else number
        elif key in {"accumulations"}:
            metadata[key] = int(value)
        elif key == "temperature_c":
            metadata[key] = float(value.replace(",", "."))
        else:
            metadata[key] = value
    return metadata


def parse_spectrum(raw: bytes) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], str]:
    lines = decode_text(raw).splitlines()
    metadata = parse_metadata(lines)
    rows: List[List[float]] = []
    for line in lines:
        normalized = line.strip().replace(",", ".")
        if not normalized or normalized.startswith(("#", "//", ";")):
            continue
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", normalized)
        if len(values) >= 2:
            try:
                rows.append([float(values[0]), float(values[1])])
            except ValueError:
                continue
    warning = ""
    if len(rows) < 5:
        return np.array([], dtype=float), np.array([], dtype=float), metadata, "Not enough numeric x/y rows were found."
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    x = data[order, 0]
    y = data[order, 1]
    if not np.all(np.isfinite(y)):
        warning = "Spectrum contains non-finite values."
    return x, y, metadata, warning


def peak_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    warning_parts: List[str] = []
    if x.size < 5 or y.size < 5:
        return {
            "peak_position_nm": None,
            "peak_max_intensity": None,
            "peak_area": None,
            "fwhm_nm": None,
            "noise_std": None,
            "snr": None,
            "warning": "Not enough spectral points for metric calculation.",
        }
    baseline = float(np.nanpercentile(y, 10))
    peak_idx = int(np.nanargmax(y))
    peak_max = float(y[peak_idx])
    height = peak_max - baseline
    peak_position = float(x[peak_idx])
    if height <= 0:
        warning_parts.append("Peak height is non-positive after baseline estimate.")

    half_level = baseline + height / 2.0
    above = np.where(y >= half_level)[0]
    if above.size >= 1:
        left, right = int(above[0]), int(above[-1])
        left_cross = float(x[left])
        right_cross = float(x[right])
        if left > 0 and not np.isclose(y[left], y[left - 1]):
            left_cross = float(x[left - 1] + (half_level - y[left - 1]) * (x[left] - x[left - 1]) / (y[left] - y[left - 1]))
        if right < len(y) - 1 and not np.isclose(y[right], y[right + 1]):
            right_cross = float(x[right] + (half_level - y[right]) * (x[right + 1] - x[right]) / (y[right + 1] - y[right]))
        fwhm = float(right_cross - left_cross) if right_cross >= left_cross else None
        area = float(np.trapezoid(np.maximum(y[max(0, left - 1) : min(len(y), right + 2)] - baseline, 0.0), x[max(0, left - 1) : min(len(y), right + 2)]))
        peak_mask = np.zeros_like(y, dtype=bool)
        peak_mask[max(0, left - 2) : min(len(y), right + 3)] = True
    else:
        fwhm = None
        area = None
        peak_mask = y >= baseline + 0.1 * max(height, 1.0)
        warning_parts.append("FWHM could not be calculated.")

    noise_region = y[~peak_mask]
    if noise_region.size >= 3:
        noise_std = float(np.std(noise_region - np.median(noise_region), ddof=1))
    else:
        noise_std = None
        warning_parts.append("Too few points outside the main peak for noise estimate.")
    snr = float(height / noise_std) if noise_std and noise_std > 0 else None
    if snr is None:
        warning_parts.append("SNR could not be calculated.")

    return {
        "peak_position_nm": peak_position,
        "peak_max_intensity": peak_max,
        "peak_area": area,
        "fwhm_nm": fwhm,
        "noise_std": noise_std,
        "snr": snr,
        "warning": "; ".join(warning_parts),
    }


def enrich_grating_from_metadata(row: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    if row.get("grating_lines_mm") is not None:
        return
    text = str(metadata.get("grating") or "")
    match = re.search(r"(\d{3,4})", text)
    if match:
        row["grating_lines_mm"] = int(match.group(1))


def plot_metric(df: pd.DataFrame, metric: str, path: Path) -> None:
    series = []
    for grating in [600, 1200]:
        subset = df[df["grating_lines_mm"] == grating].dropna(subset=["slit_width_um", metric])
        if not subset.empty:
            grouped = subset.groupby("slit_width_um", as_index=False)[metric].mean(numeric_only=True)
            series.append({"label": str(grating), "x": grouped["slit_width_um"].tolist(), "y": grouped[metric].tolist()})
    simple_line_png(path, series)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Andor .asc spectra by slit width and grating.")
    parser.add_argument("--input", required=True, help="Directory, ZIP archive or single .asc file.")
    parser.add_argument("--output", default="results/andor", help="Output directory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    fail_missing_input(input_path, "Andor .asc spectra in a directory or ZIP archive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    example_series: List[Dict[str, Any]] = []
    for file_name, raw in iter_asc_files(input_path):
        params = parse_filename_params(file_name)
        x, y, metadata, parse_warning = parse_spectrum(raw)
        metrics = peak_metrics(x, y)
        row = {
            "file_name": file_name,
            "slit_width_um": params["slit_width_um"],
            "grating_lines_mm": params["grating_lines_mm"],
            "exposure_time_s": metadata.get("exposure_time_s"),
            "accumulations": metadata.get("accumulations"),
            "detector": metadata.get("detector"),
            "temperature_c": metadata.get("temperature_c"),
            **{key: metrics[key] for key in ["peak_position_nm", "peak_max_intensity", "peak_area", "fwhm_nm", "noise_std", "snr"]},
            "warning": "; ".join(part for part in [parse_warning, metrics.get("warning", "")] if part),
        }
        enrich_grating_from_metadata(row, metadata)
        if row["slit_width_um"] is None:
            row["warning"] = (row["warning"] + "; " if row["warning"] else "") + "Slit width was not parsed from file name."
        if row["grating_lines_mm"] is None:
            row["warning"] = (row["warning"] + "; " if row["warning"] else "") + "Grating lines/mm was not parsed from file name or metadata."
        rows.append(row)
        if x.size and y.size and len(example_series) < 8:
            example_series.append({"label": file_name, "x": x.tolist(), "y": y.tolist()})

    if not rows:
        rows.append(
            {
                "file_name": None,
                "slit_width_um": None,
                "grating_lines_mm": None,
                "exposure_time_s": None,
                "accumulations": None,
                "detector": None,
                "temperature_c": None,
                "peak_position_nm": None,
                "peak_max_intensity": None,
                "peak_area": None,
                "fwhm_nm": None,
                "noise_std": None,
                "snr": None,
                "warning": "No .asc files were found.",
            }
        )

    csv_path = output / "andor_slit_metrics.csv"
    save_csv(rows, csv_path)
    df = pd.DataFrame(rows)
    plot_metric(df, "fwhm_nm", output / "fwhm_vs_slit.png")
    plot_metric(df, "snr", output / "snr_vs_slit.png")
    plot_metric(df, "peak_max_intensity", output / "intensity_vs_slit.png")
    plot_metric(df, "peak_position_nm", output / "peak_position_vs_slit.png")
    simple_line_png(output / "example_spectra.png", example_series)
    print(f"Saved: {csv_path}")
    print(f"Saved plots to: {output}")


if __name__ == "__main__":
    main()
