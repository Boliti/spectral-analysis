from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analysis.analysis_service import AnalysisService
from services.analysis.dataset_importer import UploadedFileData, import_dataset
from services.analysis.spectrum_loader import SpectrumDataset


def _file_data(path: Path) -> UploadedFileData:
    return UploadedFileData(filename=path.name, content=path.read_bytes())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "class"


def _slice_dataset(dataset: SpectrumDataset, index: int, filename: str) -> Dict[str, Any]:
    return {
        "filename": filename,
        "dataset": SpectrumDataset(
            x=dataset.x[index : index + 1],
            spectral_axis=dataset.spectral_axis,
            sample_names=[dataset.sample_names[index] if index < len(dataset.sample_names) else f"sample_{index + 1}"],
            source_format=dataset.source_format,
            filename=filename,
            file_format=dataset.file_format,
            metadata=dict(dataset.metadata or {}),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test saved PLS-DA model inference on source Excel spectra.")
    parser.add_argument("--excel", required=True, help="Path to Raman_krov_SSZ-zdorovye.xlsx")
    parser.add_argument("--per-class", type=int, default=10, help="How many spectra per class to export as TXT for inference.")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"[FAIL] Excel file not found: {excel_path}")
        return 1

    dataset = import_dataset(
        [_file_data(excel_path)],
        {
            "layout": "excel_columns",
            "target_source": "sheet_name",
            "sheet_mode": "sheet_as_class",
            "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
        },
    )
    if not dataset.y or len(set(dataset.y)) < 2:
        print("[FAIL] Imported dataset must contain at least two classes.")
        return 1

    selected: List[int] = []
    by_class: Dict[str, List[int]] = {}
    for idx, label in enumerate(dataset.y):
        by_class.setdefault(str(label), []).append(idx)
    for label in sorted(by_class):
        selected.extend(by_class[label][: max(1, int(args.per_class))])

    with tempfile.TemporaryDirectory(prefix="spectral_infer_smoke_") as tmp:
        tmp_path = Path(tmp)
        service = AnalysisService(model_root=tmp_path / "models")
        trained = service.train_and_save(
            model_type="plsda",
            dataset=dataset.to_analysis_dataset(version="raw"),
            raw_targets="\n".join(dataset.y),
            n_components=None,
            model_name="smoke_plsda",
            do_validation=False,
            random_state=42,
        )
        saved = trained["saved_model"]

        txt_files: List[Path] = []
        true_by_file: Dict[str, str] = {}
        for out_idx, row_idx in enumerate(selected, start=1):
            label = str(dataset.y[row_idx])
            path = tmp_path / f"{out_idx:02d}_{_safe_name(label)}.txt"
            matrix = np.column_stack([dataset.axis.astype(float), dataset.X[row_idx].astype(float)])
            np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")
            txt_files.append(path)
            true_by_file[path.name] = label

        imported_txt = import_dataset(
            [_file_data(path) for path in txt_files],
            {
                "layout": "txt_folder",
                "target_source": "none",
                "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
            },
        )
        infer_dataset = imported_txt.to_analysis_dataset(version="raw")
        items = [_slice_dataset(infer_dataset, idx, txt_files[idx].name) for idx in range(infer_dataset.sample_count)]
        inferred = service.infer_many(model_type=saved["model_type"], model_id=saved["model_id"], datasets=items)
        rows = inferred.get("predictions") or []
        predicted = [str(row.get("predicted")) for row in rows]
        distribution = dict(Counter(predicted))
        correct = 0
        table_rows = []
        warnings_by_file = {item.get("filename"): "; ".join(item.get("warnings") or []) for item in inferred.get("inference_reports") or []}
        for row in rows:
            filename = str(row.get("source_file") or "")
            true_class = true_by_file.get(filename, "")
            pred = str(row.get("predicted"))
            if true_class and pred == true_class:
                correct += 1
            table_rows.append((filename, true_class, pred, row.get("confidence"), warnings_by_file.get(filename, "")))
        accuracy = correct / len(rows) if rows else 0.0
        print(f"[OK] Saved model: {saved['model_type']} / {saved['model_id']}")
        print(f"[OK] Classes saved: {saved.get('classes')}")
        print(f"[OK] Prediction distribution: {distribution}")
        print("[OK] Predictions:")
        print("file\ttrue_class\tpredicted_class\tscore/probability\twarnings")
        for filename, true_class, pred, confidence, warning in table_rows:
            print(f"{filename}\t{true_class}\t{pred}\t{confidence}\t{warning}")
        print(f"[OK] Accuracy: {correct}/{len(rows)} = {accuracy:.3f}")

        if len(distribution) < 2:
            print("[FAIL] All selected source spectra were predicted as one class.")
            return 1
        if accuracy < 0.5:
            print("[FAIL] Accuracy is below 0.5 on source spectra smoke-test.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
