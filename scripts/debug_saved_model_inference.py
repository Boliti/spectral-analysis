from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _find_excel(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"Excel file not found: {path}")
    candidates: List[Path] = []
    for root in [ROOT, ROOT.parent, Path.home() / "Desktop"]:
        if root.exists():
            candidates.extend(root.rglob("Raman_krov_SSZ-zdorovye.xlsx"))
    if not candidates:
        raise FileNotFoundError("Raman_krov_SSZ-zdorovye.xlsx not found. Pass --excel <path>.")
    return candidates[0]


def _label_from_filename(path: Path) -> Optional[str]:
    text = path.stem.lower().replace("_", " ").replace("-", " ")
    if "heart disease" in text or ("heart" in text and "disease" in text) or "ssz" in text:
        return "heart disease"
    if "healthy" in text or "health" in text or "zdorov" in text:
        return "health"
    return None


def _slice_dataset(dataset: SpectrumDataset, index: int, filename: str) -> Dict[str, Any]:
    metadata = dict(dataset.metadata or {})
    sample_metadata = metadata.get("sample_metadata") or []
    if isinstance(sample_metadata, list) and index < len(sample_metadata) and isinstance(sample_metadata[index], dict):
        metadata["sample_metadata"] = [sample_metadata[index]]
        metadata["single_column_intensity_only"] = bool(sample_metadata[index].get("intensity_only"))
    return {
        "filename": filename,
        "dataset": SpectrumDataset(
            x=dataset.x[index : index + 1],
            spectral_axis=dataset.spectral_axis,
            sample_names=[dataset.sample_names[index] if index < len(dataset.sample_names) else f"sample_{index + 1}"],
            source_format=dataset.source_format,
            filename=filename,
            file_format=dataset.file_format,
            metadata=metadata,
        ),
    }


def _import_training_excel(excel_path: Path):
    return import_dataset(
        [_file_data(excel_path)],
        {
            "layout": "excel_columns",
            "target_source": "sheet_name",
            "sheet_mode": "sheet_as_class",
            "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
        },
    )


def _select_excel_indices(labels: Iterable[str], per_class: int) -> List[int]:
    by_class: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(str(label), []).append(idx)
    required = ["health", "heart disease"]
    missing = [label for label in required if label not in by_class]
    if missing:
        raise ValueError(f"Training Excel does not contain required classes: {missing}. Found: {sorted(by_class)}")
    selected: List[int] = []
    for label in required:
        selected.extend(by_class[label][:per_class])
    return selected


def _write_txt_from_excel(dataset, selected: List[int], out_dir: Path) -> Tuple[List[Path], Dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    true_by_file: Dict[str, str] = {}
    for idx, row_idx in enumerate(selected, start=1):
        label = str(dataset.y[row_idx])
        path = out_dir / f"{idx:02d}_{_safe_name(label)}.txt"
        matrix = np.column_stack([dataset.axis.astype(float), dataset.X[row_idx].astype(float)])
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")
        files.append(path)
        true_by_file[path.name] = label
    return files, true_by_file


def _select_txt_files(txt_dir: Path, per_class: int, pattern: str) -> Tuple[List[Path], Dict[str, str]]:
    by_class: Dict[str, List[Path]] = {"health": [], "heart disease": []}
    for path in sorted(txt_dir.glob(pattern)):
        if not path.is_file():
            continue
        label = _label_from_filename(path)
        if label in by_class:
            by_class[label].append(path)
    missing = [label for label, files in by_class.items() if len(files) < per_class]
    if missing:
        details = {label: len(files) for label, files in by_class.items()}
        raise ValueError(f"Not enough labelled TXT files for {missing}. Need {per_class} each, found {details}.")
    files = by_class["health"][:per_class] + by_class["heart disease"][:per_class]
    return files, {path.name: _label_from_filename(path) or "" for path in files}


def _score_for_file(batch_item: Dict[str, Any]) -> Any:
    result = batch_item.get("result") or {}
    scores = result.get("decision_score")
    if isinstance(scores, list) and scores:
        return scores[0]
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions:
        return predictions[0]
    return ""


def _prediction_for_file(batch_item: Dict[str, Any]) -> str:
    result = batch_item.get("result") or {}
    values = result.get("predicted_classes") or result.get("predictions") or []
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


def _confusion_matrix(true_labels: List[str], predicted: List[str], labels: List[str]) -> List[List[int]]:
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: idx for idx, label in enumerate(labels)}
    for true, pred in zip(true_labels, predicted):
        if true in index and pred in index:
            matrix[index[true]][index[pred]] += 1
    return matrix


def _print_table(rows: List[Dict[str, Any]]) -> None:
    columns = [
        "file_name",
        "true_label_from_filename",
        "predicted_label",
        "raw_feature_count",
        "prepared_feature_count",
        "expected_feature_count",
        "axis_min_input",
        "axis_max_input",
        "axis_min_model",
        "axis_max_model",
        "interpolated",
        "preprocessing_applied",
        "decision_score",
        "warning",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug saved PLS-DA inference on TXT spectra using the web backend path.")
    parser.add_argument("--excel", help="Path to Raman_krov_SSZ-zdorovye.xlsx. If omitted, the script searches common project/Desktop locations.")
    parser.add_argument("--txt-dir", help="Optional directory with real TXT files. If omitted, 10+10 TXT files are exported from the Excel dataset.")
    parser.add_argument("--txt-glob", default="*.txt", help="Glob for TXT files inside --txt-dir.")
    parser.add_argument("--per-class", type=int, default=10, help="Number of health and heart disease spectra to test.")
    parser.add_argument("--n-components", type=int, default=None, help="Optional PLS-DA n_components.")
    args = parser.parse_args()

    excel_path = _find_excel(args.excel)
    training_dataset = _import_training_excel(excel_path)
    if not training_dataset.y:
        raise ValueError("Training dataset has no class labels.")

    with tempfile.TemporaryDirectory(prefix="spectral_debug_infer_") as tmp:
        tmp_path = Path(tmp)
        service = AnalysisService(model_root=tmp_path / "saved_models")
        trained = service.train_and_save(
            model_type="plsda",
            dataset=training_dataset.to_analysis_dataset(version="raw"),
            raw_targets="\n".join(str(value) for value in training_dataset.y),
            n_components=args.n_components,
            model_name="debug_plsda",
            do_validation=True,
            random_state=42,
        )
        saved = trained["saved_model"]
        print(f"saved_model: {saved.get('model_type')} / {saved.get('model_id')}")
        print(f"classes: {saved.get('classes') or saved.get('class_labels')}")
        print(f"expected_feature_count: {saved.get('expected_feature_count')}")
        print(f"axis_model: {saved.get('axis_min')}..{saved.get('axis_max')}")

        if args.txt_dir:
            txt_files, true_by_file = _select_txt_files(Path(args.txt_dir), args.per_class, args.txt_glob)
        else:
            selected = _select_excel_indices(training_dataset.y, args.per_class)
            txt_files, true_by_file = _write_txt_from_excel(training_dataset, selected, tmp_path / "txt")

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

        reports = {report.get("filename"): report for report in inferred.get("inference_reports") or []}
        batch = {item.get("filename"): item for item in inferred.get("batch_results") or []}
        rows: List[Dict[str, Any]] = []
        true_labels: List[str] = []
        predicted_labels: List[str] = []
        for path in txt_files:
            filename = path.name
            report = reports.get(filename, {})
            item = batch.get(filename, {})
            true_label = true_by_file.get(filename, "")
            predicted = _prediction_for_file(item)
            true_labels.append(true_label)
            predicted_labels.append(predicted)
            rows.append(
                {
                    "file_name": filename,
                    "true_label_from_filename": true_label,
                    "predicted_label": predicted,
                    "raw_feature_count": report.get("raw_feature_count") or report.get("input_feature_count"),
                    "prepared_feature_count": report.get("prepared_feature_count"),
                    "expected_feature_count": report.get("expected_feature_count"),
                    "axis_min_input": report.get("axis_min_input") or report.get("input_axis_min"),
                    "axis_max_input": report.get("axis_max_input") or report.get("input_axis_max"),
                    "axis_min_model": report.get("axis_min_model") or report.get("model_axis_min"),
                    "axis_max_model": report.get("axis_max_model") or report.get("model_axis_max"),
                    "interpolated": report.get("interpolated_to_model_axis"),
                    "preprocessing_applied": report.get("preprocessing_applied"),
                    "decision_score": _score_for_file(item),
                    "warning": "; ".join(report.get("warnings") or []),
                }
            )

        _print_table(rows)
        correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true and true == pred)
        accuracy = correct / len(predicted_labels) if predicted_labels else 0.0
        distribution = dict(Counter(predicted_labels))
        labels = ["health", "heart disease"]
        print(f"accuracy: {correct}/{len(predicted_labels)} = {accuracy:.3f}")
        print(f"prediction_distribution: {distribution}")
        print(f"confusion_matrix labels={labels}: {_confusion_matrix(true_labels, predicted_labels, labels)}")
        if len(set(predicted_labels)) == 1:
            print("WARNING: All files were assigned to one class. Check spectral axis, preprocessing and model applicability.")
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
