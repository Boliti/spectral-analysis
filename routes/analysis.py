from __future__ import annotations

import asyncio
import json
import logging
import os
import io
import time
import zipfile
import csv
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from services.analysis.analysis_service import AnalysisService
from services.analysis.dataset_importer import (
    DatasetImportError,
    SpectralDataset,
    UploadedFileData,
    apply_preprocessing,
    dataset_store,
    import_dataset,
    preview_preprocessing,
    preprocessing_status,
    preview_files,
    spectra_preview,
    upload_files_to_memory,
    validate_dataset,
    _read_config,
    _safe_preview_value,
)
from services.analysis.run_manager import AnalysisRunManager
from services.analysis.spectrum_loader import SpectrumValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = Path(os.getenv("SAVED_RUNS_DIR", str(BASE_DIR / "saved_runs")))
SAVED_DATASETS_DIR = Path(os.getenv("SAVED_DATASETS_DIR", str(BASE_DIR / "saved_datasets")))
CONFIGS_DIR = RESULTS_DIR / "configs"
REPORTS_DIR = RESULTS_DIR / "reports"
USER_DATA_DIR = BASE_DIR / os.getenv("USER_DATA_DIR", "user_data")

router = APIRouter(prefix="/analysis", tags=["analysis"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
analysis_service = AnalysisService(model_root=MODELS_DIR)
run_manager = AnalysisRunManager(RUNS_DIR)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def _safe_user_segment(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "unknown"))


def _safe_filename(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "unknown")).strip("_-")[:128]


def _build_run_csv_filename(run: Dict[str, Any]) -> str:
    name = _safe_filename(run.get("dataset_name") or run.get("dataset_id") or "dataset")
    version = _safe_filename(run.get("dataset_version") or "unknown")
    run_type = _safe_filename(run.get("run_type") or "run")
    run_id = _safe_filename(run.get("run_id") or "run")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{name}_{version}_{run_type}_{run_id}_{timestamp}.csv"


def dataset_slice(dataset: Any, index: int):
    metadata = dict(dataset.metadata or {})
    sample_metadata = metadata.get("sample_metadata") or []
    if isinstance(sample_metadata, list) and index < len(sample_metadata) and isinstance(sample_metadata[index], dict):
        metadata["sample_metadata"] = [sample_metadata[index]]
        metadata["single_column_intensity_only"] = bool(sample_metadata[index].get("intensity_only"))
    return type(dataset)(
        x=dataset.x[index : index + 1],
        spectral_axis=dataset.spectral_axis,
        sample_names=[dataset.sample_names[index] if index < len(dataset.sample_names) else f"sample_{index + 1}"],
        source_format=dataset.source_format,
        filename=dataset.filename,
        file_format=dataset.file_format,
        metadata=metadata,
    )


def current_analysis_user(request: Request) -> Dict[str, Any]:
    username = request.session.get("user")
    if not username:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "user_message": "Для работы с анализом необходимо войти в аккаунт.",
            },
        )
    user_id = request.session.get("user_id") or username
    return {"user_id": str(user_id), "username": str(username)}


def user_root(user: Dict[str, Any]) -> Path:
    root = USER_DATA_DIR / f"user_{_safe_user_segment(user['user_id'])}"
    for name in ["datasets", "models", "runs", "uploads", "exports", "configs", "reports"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def user_analysis_service(user: Dict[str, Any]) -> AnalysisService:
    return AnalysisService(model_root=user_root(user) / "models")


def user_run_manager(user: Dict[str, Any]) -> AnalysisRunManager:
    return AnalysisRunManager(user_root(user) / "runs")


def _attach_owner(dataset: SpectralDataset, user: Dict[str, Any]) -> None:
    dataset.metadata["user_id"] = user["user_id"]
    dataset.metadata["username"] = user["username"]


def _assert_owner(metadata: Dict[str, Any], user: Dict[str, Any]) -> None:
    owner = metadata.get("user_id")
    if owner is not None and str(owner) != str(user["user_id"]):
        raise HTTPException(status_code=403, detail="Нет доступа к этому объекту.")


def get_owned_dataset(dataset_id: str, user: Dict[str, Any]) -> SpectralDataset:
    try:
        dataset = dataset_store.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Датасет не найден") from exc
    _assert_owner(dataset.metadata, user)
    return dataset


def persist_user_dataset(dataset: SpectralDataset, user: Dict[str, Any]) -> None:
    dataset_id = dataset.dataset_id
    if not dataset_id:
        return
    dataset_dir = user_root(user) / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_export_frame(version="raw").to_csv(dataset_dir / "raw_dataset.csv", index=False)
    if dataset.processed_X is not None:
        dataset.to_export_frame(version="processed").to_csv(dataset_dir / "processed_dataset.csv", index=False)
    (dataset_dir / "metadata.json").write_text(
        json.dumps(dataset.metadata_export(version="processed" if dataset.processed_X is not None else "raw"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "import_config.json").write_text(
        json.dumps(dataset.import_config(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "preprocessing_config.json").write_text(
        json.dumps(dataset.preprocessing_config or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def saved_datasets_root(user: Dict[str, Any]) -> Path:
    root = SAVED_DATASETS_DIR / _safe_user_segment(user["user_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(value: Any, fallback: str = "dataset") -> str:
    text = str(value or "").strip()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text[:80] or fallback


def _preprocessing_suffix(dataset: SpectralDataset) -> str:
    cfg = dataset.preprocessing_config or {}
    parts: List[str] = []
    baseline = (cfg.get("baseline") or {}).get("method")
    smoothing = (cfg.get("smoothing") or {}).get("method")
    normalization = (cfg.get("normalization") or {}).get("method")
    if baseline and baseline != "off":
        parts.append(str(baseline).upper())
    if smoothing and smoothing != "off":
        parts.append("SG" if str(smoothing).lower() == "savgol" else str(smoothing).upper())
    if normalization and normalization != "off":
        parts.append(str(normalization).upper())
    return "_".join(parts)


def _dataset_short_name(dataset: SpectralDataset, version: str = "raw") -> str:
    source_text = " ".join(dataset.source_files[:3]).lower()
    if "raman_krov" in source_text or "krov" in source_text or "ssz" in source_text:
        base = "Raman_krov"
    elif "slit" in source_text or "andor" in source_text or dataset.metadata.get("grating_lines_mm_values"):
        gratings = dataset.metadata.get("grating_lines_mm_values") or []
        grating_part = "_" + "_".join(str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v) for v in gratings[:3]) if gratings else ""
        base = f"Andor_slit{grating_part}"
    else:
        first_source = Path(dataset.source_files[0]).stem if dataset.source_files else "dataset"
        base = _slug(first_source, "dataset")
    suffix = f"_{_preprocessing_suffix(dataset)}" if version == "processed" and _preprocessing_suffix(dataset) else ""
    return f"{base}_{version}{suffix}"


def _default_saved_dataset_name(dataset: SpectralDataset, version: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{_dataset_short_name(dataset, version)}_{date}"


def _saved_dataset_metadata(dataset: SpectralDataset, saved_id: str, dataset_name: str, version: str, data_path: Path) -> Dict[str, Any]:
    summary = dataset.summary(version=version)
    preprocessing_summary = summary.get("preprocessing") or {}
    if version == "processed" and dataset.processed_axis is not None:
        preprocessing_summary = {
            "n_features": int(dataset.processed_axis.size),
            "axis_min": float(np.min(dataset.processed_axis)) if dataset.processed_axis.size else None,
            "axis_max": float(np.max(dataset.processed_axis)) if dataset.processed_axis.size else None,
            "config": dataset.preprocessing_config or {},
        }
    return {
        "dataset_id": saved_id,
        "dataset_name": dataset_name,
        "source_file_name": ";".join(dataset.source_files[:3]),
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": summary.get("n_samples"),
        "n_features": summary.get("n_features"),
        "target_column": dataset.target_name,
        "target_type": summary.get("target_type"),
        "classes": summary.get("class_names") or list((summary.get("class_distribution") or {}).keys()),
        "axis_min": summary.get("axis_min"),
        "axis_max": summary.get("axis_max"),
        "preprocessing_config": dataset.preprocessing_config if version == "processed" else {},
        "preprocessing_summary": preprocessing_summary,
        "data_path": str(data_path),
        "metadata": dataset.metadata_export(version=version),
        "raw_metadata": dataset.metadata,
        "source_layout": dataset.source_layout,
        "source_files": dataset.source_files,
        "sample_metadata": dataset.sample_metadata or [],
    }


def _saved_dataset_dir(user: Dict[str, Any], saved_id: str) -> Path:
    safe_id = _slug(saved_id, "saved_dataset")
    target = saved_datasets_root(user) / safe_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="Сохранённый датасет не найден")
    return target


def _read_saved_dataset_meta(path: Path) -> Dict[str, Any]:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _optional_str_array(values: Optional[List[Any]]) -> np.ndarray:
    if values is None:
        return np.array([], dtype=str)
    return np.asarray([str(value) for value in values], dtype=str)


def _write_saved_dataset_npz(dataset: SpectralDataset, target_dir: Path) -> None:
    processed_X = dataset.processed_X if dataset.processed_X is not None else np.empty((0, 0), dtype=float)
    processed_axis = dataset.processed_axis if dataset.processed_axis is not None else np.array([], dtype=float)
    np.savez_compressed(
        target_dir / "dataset.npz",
        X=np.asarray(dataset.X, dtype=float),
        spectral_axis=np.asarray(dataset.axis, dtype=float),
        processed_X=np.asarray(processed_X, dtype=float),
        processed_axis=np.asarray(processed_axis, dtype=float),
        sample_ids=np.asarray(dataset.sample_ids, dtype=str),
        y=_optional_str_array(dataset.y),
        class_names=_optional_str_array(dataset.class_names),
        group_ids=_optional_str_array(dataset.group_ids),
        source_sheets=_optional_str_array(dataset.source_sheets),
        source_items=_optional_str_array(dataset.source_items),
    )


def _load_saved_dataset(target_dir: Path) -> SpectralDataset:
    metadata = _read_saved_dataset_meta(target_dir)
    npz_path = target_dir / "dataset.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            processed_X = np.asarray(data["processed_X"], dtype=float) if "processed_X" in data else np.empty((0, 0), dtype=float)
            processed_axis = np.asarray(data["processed_axis"], dtype=float) if "processed_axis" in data else np.array([], dtype=float)
            y_arr = data["y"].astype(str).tolist() if "y" in data and data["y"].size else None
            class_arr = data["class_names"].astype(str).tolist() if "class_names" in data and data["class_names"].size else None
            group_arr = data["group_ids"].astype(str).tolist() if "group_ids" in data and data["group_ids"].size else None
            sheets_arr = data["source_sheets"].astype(str).tolist() if "source_sheets" in data and data["source_sheets"].size else None
            items_arr = data["source_items"].astype(str).tolist() if "source_items" in data and data["source_items"].size else None
            exported_meta = metadata.get("metadata") or {}
            raw_metadata = metadata.get("raw_metadata") if isinstance(metadata.get("raw_metadata"), dict) else {}
            exported_source_files = exported_meta.get("source_files")
            if isinstance(exported_source_files, list):
                source_files = [str(item) for item in exported_source_files]
            else:
                raw_source_files = metadata.get("source_files")
                source_files = [str(item) for item in raw_source_files] if isinstance(raw_source_files, list) else str(metadata.get("source_file_name") or metadata.get("dataset_name") or "saved_dataset").split(";")
            dataset = SpectralDataset(
                X=np.asarray(data["X"], dtype=float),
                axis=np.asarray(data["spectral_axis"], dtype=float),
                sample_ids=data["sample_ids"].astype(str).tolist(),
                y=y_arr,
                target_name=metadata.get("target_column") or exported_meta.get("target_name"),
                class_names=class_arr or metadata.get("classes"),
                metadata=dict(raw_metadata or exported_meta.get("metadata") or exported_meta or {}),
                source_layout=str(metadata.get("source_layout") or exported_meta.get("source_layout") or "saved_dataset"),
                source_files=source_files,
                group_ids=group_arr,
                source_sheets=sheets_arr,
                source_items=items_arr,
                sample_metadata=(metadata.get("sample_metadata") if isinstance(metadata.get("sample_metadata"), list) else None),
                processed_X=processed_X if processed_X.size else None,
                processed_axis=processed_axis if processed_axis.size else None,
                preprocessing_config=metadata.get("preprocessing_config") or None,
            )
            dataset.metadata.setdefault("target_type", metadata.get("target_type"))
            dataset.metadata.setdefault("dataset_name", metadata.get("dataset_name"))
            return dataset
    with (target_dir / "dataset.pkl").open("rb") as handle:
        return pickle.load(handle)


def _write_saved_dataset_files(dataset: SpectralDataset, target_dir: Path, version: str, metadata: Dict[str, Any]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_saved_dataset_npz(dataset, target_dir)
    with (target_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump(dataset, handle)
    dataset.to_export_frame(version=version).to_csv(target_dir / f"{version}.csv", index=False)
    try:
        dataset.to_export_frame(version=version).to_excel(target_dir / f"{version}.xlsx", index=False)
    except Exception:
        pass
    (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_user_model(user: Dict[str, Any], model_id: str, model_type: Optional[str] = None) -> Dict[str, str]:
    manager = user_analysis_service(user).model_manager
    if model_type:
        manager.get_metadata(model_type, model_id)
        return {"model_type": model_type, "model_id": model_id}
    matches = [meta for meta in manager.list_models() if meta.get("model_id") == model_id]
    if not matches:
        raise HTTPException(status_code=404, detail="Модель не найдена")
    if len(matches) > 1:
        raise HTTPException(status_code=400, detail="Найдено несколько моделей с таким id. Укажите model_type.")
    return {"model_type": matches[0]["model_type"], "model_id": model_id}


def error_payload(error_code: str, user_message: str, details: str = "", suggestions: Optional[List[str]] = None, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_code": error_code,
            "user_message": user_message,
            "details": details,
            "suggestions": suggestions or [
                "Убедитесь, что выбран правильный лист Excel.",
                "Проверьте, что выбран верный тип структуры данных.",
                "Проверьте, что спектральная ось и интенсивности содержат числовые значения.",
            ],
        },
    )


def import_error_response(exc: Exception, code: str = "IMPORT_FAILED") -> JSONResponse:
    details = str(exc)
    suggestions = [
        "Убедитесь, что выбран правильный лист Excel.",
        "Проверьте, что колонка спектральной оси содержит числовые значения.",
        "Проверьте, что выбранные листы имеют одинаковую спектральную ось или включена общая сетка.",
        "Проверьте, что target-колонка существует, если выбран target из таблицы.",
    ]
    if "Лист" in details:
        suggestions.insert(0, "Выберите существующий лист Excel или используйте кнопку «Выбрать все».")
    if "ось" in details or "axis" in details:
        suggestions.insert(0, "Проверьте колонку спектральной оси: она должна содержать только числа.")
    if "target" in details.lower():
        suggestions.insert(0, "Выберите target-колонку или режим «Без target».")
    return error_payload(
        code,
        "Не удалось импортировать датасет. Проверьте выбранный тип структуры и сопоставление колонок.",
        details=details,
        suggestions=suggestions,
        status_code=400,
    )


class ModelTrainRequest(BaseModel):
    dataset_id: str
    use_processed: bool = False
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
    model_type: str
    model_name: Optional[str] = None
    n_components: Optional[int] = None
    model_params: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    target_column: Optional[str] = None
    hyperparameter_search: bool = False
    refit_on_full_data: bool = True


class SaveDatasetRequest(BaseModel):
    dataset_id: str
    dataset_name: Optional[str] = None
    version: str = "raw"


class RenameSavedDatasetRequest(BaseModel):
    dataset_name: str


@router.get("")
async def analysis_page(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login?next=/analysis", status_code=303)
    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "request": request,
            "user": request.session.get("user"),
            "user_id": request.session.get("user_id"),
            "is_authenticated": bool(request.session.get("user")),
        },
    )


@router.post("/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        dataset = analysis_service.parse_file(raw=raw, filename=file.filename or "")
        records = dataset.to_spectrum_records()
        return {
            "status": "ok",
            "filename": file.filename,
            "sample_count": dataset.sample_count,
            "feature_count": dataset.feature_count,
            "source_format": dataset.source_format,
            "file_format": dataset.file_format,
            "has_axis": dataset.spectral_axis is not None,
            "spectra": records[:3],
            "metadata": dataset.metadata,
        }
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload-preview error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка проверки файла") from exc


@router.post("/dataset/preview")
async def dataset_preview(files: List[UploadFile] = File(...)):
    try:
        memory_files = await upload_files_to_memory(files)
        return preview_files(memory_files)
    except DatasetImportError as exc:
        return import_error_response(exc, "PREVIEW_FAILED")
    except Exception as exc:
        logger.exception("Unexpected dataset preview error")
        return error_payload("PREVIEW_FAILED", "Не удалось выполнить предпросмотр файла.", details=str(exc), status_code=500)


@router.post("/dataset/validate")
async def dataset_validate(
    files: List[UploadFile] = File(...),
    import_config: str = Form("{}"),
):
    try:
        memory_files = await upload_files_to_memory(files)
        config = _read_config(import_config)
        dataset = import_dataset(memory_files, config)
        validation = validate_dataset(dataset)
        return {**validation, "preview": spectra_preview(dataset, limit=10, strategy="balanced_by_class")}
    except DatasetImportError as exc:
        return import_error_response(exc, "VALIDATION_FAILED")
    except Exception as exc:
        logger.exception("Unexpected dataset validation error")
        return error_payload("VALIDATION_FAILED", "Не удалось проверить настройки импорта.", details=str(exc), status_code=500)


@router.post("/dataset/import")
async def dataset_import(
    request: Request,
    files: List[UploadFile] = File(...),
    import_config: str = Form("{}"),
):
    try:
        user = current_analysis_user(request)
        memory_files = await upload_files_to_memory(files)
        config = _read_config(import_config)
        import_start = time.perf_counter()
        dataset = import_dataset(memory_files, config)
        dataset.metadata["import_time_sec"] = round(time.perf_counter() - import_start, 6)
        _attach_owner(dataset, user)
        dataset.metadata["dataset_name"] = _dataset_short_name(dataset, "raw")
        validation = validate_dataset(dataset)
        if validation["status"] == "error":
            return {"status": "error", "validation": validation}
        dataset_id = dataset_store.put(dataset)
        persist_user_dataset(dataset, user)
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "summary": dataset.summary(),
            "validation": validation,
            "preview": spectra_preview(dataset, limit=10, strategy="balanced_by_class"),
        }
    except DatasetImportError as exc:
        return import_error_response(exc)
    except Exception as exc:
        logger.exception("Unexpected dataset import error")
        return error_payload("IMPORT_FAILED", "Не удалось импортировать датасет. Проверьте файл и выбранные настройки.", details=str(exc), status_code=500)


@router.post("/dataset/import-standard")
async def dataset_import_standard(request: Request, file: UploadFile = File(...)):
    try:
        user = current_analysis_user(request)
        raw = await file.read()
        dataset = _parse_standard_dataset(raw, file.filename or "standard_dataset")
        _attach_owner(dataset, user)
        dataset.metadata["dataset_name"] = _dataset_short_name(dataset, "raw")
        validation = validate_dataset(dataset)
        if validation["status"] == "error":
            return {"status": "error", "validation": validation}
        dataset_id = dataset_store.put(dataset)
        persist_user_dataset(dataset, user)
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "summary": dataset.summary(),
            "validation": validation,
            "preview": spectra_preview(dataset, limit=10, strategy="balanced_by_class"),
        }
    except DatasetImportError as exc:
        return import_error_response(exc, "STANDARD_IMPORT_FAILED")
    except Exception as exc:
        logger.exception("Unexpected standard dataset import error")
        return error_payload(
            "STANDARD_IMPORT_FAILED",
            "Не удалось загрузить правильный датасет.",
            details=str(exc),
            suggestions=[
                "Проверьте, что первые колонки называются sample_id, target, source_sheet/source_file.",
                "Проверьте, что спектральные признаки имеют числовые заголовки.",
                "Если target отсутствует, supervised-модели будут недоступны.",
            ],
            status_code=500,
        )


@router.get("/datasets")
async def list_user_datasets(request: Request):
    user = current_analysis_user(request)
    datasets_dir = user_root(user) / "datasets"
    items = []
    for meta_path in sorted(datasets_dir.glob("*/metadata.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dataset_id = meta.get("dataset_id") or meta_path.parent.name
        has_processed = (meta_path.parent / "processed_dataset.csv").exists()
        items.append({
            "dataset_id": dataset_id,
            "created_at": meta.get("created_at"),
            "n_samples": meta.get("n_samples"),
            "n_features": meta.get("n_features"),
            "target_name": meta.get("target_name"),
            "target_type": meta.get("target_type"),
            "class_distribution": meta.get("class_distribution"),
            "source_layout": meta.get("source_layout"),
            "source_files": meta.get("source_files"),
            "has_processed": has_processed,
        })
    return {"status": "ok", "datasets": items}


def _parse_standard_dataset(raw: bytes, filename: str) -> SpectralDataset:
    ext = Path(filename).suffix.lower()
    try:
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(raw))
        else:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1251")
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise DatasetImportError(f"Не удалось прочитать правильный датасет: {exc}") from exc

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise DatasetImportError("Файл правильного датасета пуст.")
    columns = [str(col).strip() for col in df.columns]
    df.columns = columns
    lower_map = {col.lower(): col for col in columns}
    sample_col = lower_map.get("sample_id") or lower_map.get("sample") or lower_map.get("id")
    target_col = (
        lower_map.get("target")
        or lower_map.get("conc (y)")
        or lower_map.get("total_protein (y)")
        or lower_map.get("conc_total_protein (y)")
        or lower_map.get("concentration")
        or lower_map.get("y")
        or lower_map.get("group")
        or lower_map.get("class")
        or lower_map.get("label")
    )
    source_sheet_col = lower_map.get("source_sheet")
    source_file_col = lower_map.get("source_file")
    metadata_cols = {
        col for key, col in lower_map.items()
        if key in {"date", "seria", "cuvetta"} or col in {source_sheet_col, source_file_col}
    }

    meta_cols = {col for col in [sample_col, target_col, source_sheet_col, source_file_col] if col} | metadata_cols
    feature_cols: List[str] = []
    axis_values: List[float] = []
    for col in columns:
        if col in meta_cols:
            continue
        try:
            axis_values.append(float(str(col).replace(",", ".")))
            feature_cols.append(col)
        except ValueError:
            continue
    if len(feature_cols) < 2:
        raise DatasetImportError("Не найдены спектральные признаки: нужны числовые заголовки колонок оси.")

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_df.isna().any().any():
        raise DatasetImportError("В спектральных признаках есть нечисловые или пустые значения.")
    sample_ids = df[sample_col].astype(str).tolist() if sample_col else [f"sample_{idx + 1}" for idx in range(len(df))]
    y = df[target_col].astype(str).tolist() if target_col else None
    target_lower = str(target_col or "").strip().lower()
    target_type = (
        "categorical"
        if target_lower in {"class", "group", "label", "conc (class)"}
        else ("numeric" if target_col and pd.to_numeric(df[target_col], errors="coerce").notna().sum() == len(df) else ("categorical" if target_col else "none"))
    )
    source_sheets = df[source_sheet_col].astype(str).tolist() if source_sheet_col else None
    source_items = df[source_file_col].astype(str).tolist() if source_file_col else None
    sample_metadata = [
        {str(col): _safe_preview_value(row[col]) for col in metadata_cols if col in df.columns}
        for _, row in df.iterrows()
    ] if metadata_cols else None
    return SpectralDataset(
        X=X_df.to_numpy(dtype=float),
        axis=np.asarray(axis_values, dtype=float),
        sample_ids=sample_ids,
        y=y,
        target_name=target_col,
        class_names=None if target_type == "numeric" else (sorted(set(y)) if y else None),
        metadata={
            "import_config": {"layout": "standard_dataset", "filename": filename},
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "target_type": target_type,
            "measurement_metadata": {},
            "preprocessing_status": "not_applied",
        },
        source_layout="standard_dataset",
        source_files=[filename],
        source_sheets=source_sheets,
        source_items=source_items,
        sample_metadata=sample_metadata,
        )


@router.get("/saved-datasets")
async def list_saved_datasets(request: Request):
    user = current_analysis_user(request)
    root = saved_datasets_root(user)
    items: List[Dict[str, Any]] = []
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            items.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"status": "ok", "datasets": items}


@router.post("/saved-datasets/save")
async def save_current_dataset(request: Request, payload: SaveDatasetRequest):
    user = current_analysis_user(request)
    dataset = get_owned_dataset(payload.dataset_id, user)
    version = "processed" if payload.version == "processed" else "raw"
    if version == "processed" and dataset.processed_X is None:
        raise HTTPException(status_code=400, detail="Processed-версия датасета ещё не создана")
    name = _slug(payload.dataset_name or _default_saved_dataset_name(dataset, version), "dataset")
    saved_id = f"{name}_{uuid.uuid4().hex[:8]}"
    target_dir = saved_datasets_root(user) / saved_id
    data_path = target_dir / f"{version}.csv"
    metadata = _saved_dataset_metadata(dataset, saved_id=saved_id, dataset_name=name, version=version, data_path=data_path)
    metadata["source_runtime_dataset_id"] = payload.dataset_id
    _write_saved_dataset_files(dataset, target_dir, version, metadata)
    return {"status": "ok", "saved_dataset": metadata}


@router.post("/saved-datasets/{saved_id}/use")
async def use_saved_dataset(request: Request, saved_id: str):
    user = current_analysis_user(request)
    target_dir = _saved_dataset_dir(user, saved_id)
    metadata = _read_saved_dataset_meta(target_dir)
    dataset = _load_saved_dataset(target_dir)
    _attach_owner(dataset, user)
    dataset.metadata["saved_dataset_id"] = saved_id
    dataset.metadata["dataset_name"] = metadata.get("dataset_name")
    dataset_id = dataset_store.put(dataset)
    persist_user_dataset(dataset, user)
    version = "processed" if metadata.get("version") == "processed" and dataset.processed_X is not None else "raw"
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "saved_dataset_id": saved_id,
        "version": version,
        "summary": dataset.summary(version=version),
        "metadata": metadata,
        "preview": spectra_preview(dataset, limit=10, strategy="balanced_by_class", version=version),
    }


@router.patch("/saved-datasets/{saved_id}")
async def rename_saved_dataset(request: Request, saved_id: str, payload: RenameSavedDatasetRequest):
    user = current_analysis_user(request)
    target_dir = _saved_dataset_dir(user, saved_id)
    metadata = _read_saved_dataset_meta(target_dir)
    new_name = _slug(payload.dataset_name, "dataset")
    metadata["dataset_name"] = new_name
    metadata["renamed_at"] = datetime.now(timezone.utc).isoformat()
    (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "saved_dataset": metadata}


@router.delete("/saved-datasets/{saved_id}")
async def delete_saved_dataset(request: Request, saved_id: str):
    user = current_analysis_user(request)
    target_dir = _saved_dataset_dir(user, saved_id)
    for child in target_dir.iterdir():
        if child.is_file():
            child.unlink()
    target_dir.rmdir()
    return {"status": "ok", "deleted": saved_id}


async def _stream_saved_dataset_export(request: Request, saved_id: str, format: str = "csv"):
    user = current_analysis_user(request)
    target_dir = _saved_dataset_dir(user, saved_id)
    metadata = _read_saved_dataset_meta(target_dir)
    version = metadata.get("version") or "raw"
    fmt = format.lower()
    if fmt == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(target_dir / "metadata.json", "metadata.json")
            if (target_dir / "dataset.npz").exists():
                archive.write(target_dir / "dataset.npz", "dataset.npz")
            if (target_dir / f"{version}.csv").exists():
                archive.write(target_dir / f"{version}.csv", f"{version}.csv")
            if (target_dir / f"{version}.xlsx").exists():
                archive.write(target_dir / f"{version}.xlsx", f"{version}.xlsx")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={saved_id}.zip"})
    if fmt == "json":
        dataset = _load_saved_dataset(target_dir)
        X, axis = dataset._matrix_for_version(version)
        payload = {
            "metadata": metadata,
            "dataset": {
                **dataset.to_jsonable(),
                "version": version,
                "X": X.tolist(),
                "spectral_axis": axis.tolist(),
                "sample_ids": dataset.sample_ids,
                "y": dataset.y,
            },
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return StreamingResponse(io.BytesIO(content), media_type="application/json", headers={"Content-Disposition": f"attachment; filename={saved_id}.json"})
    file_path = target_dir / f"{version}.{fmt}"
    if not file_path.exists() and fmt == "xlsx":
        dataset = _load_saved_dataset(target_dir)
        dataset.to_export_frame(version=version).to_excel(file_path, index=False)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Экспорт сохранённого датасета не найден")
    media = "text/csv" if fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(io.BytesIO(file_path.read_bytes()), media_type=media, headers={"Content-Disposition": f"attachment; filename={saved_id}_{version}.{fmt}"})


@router.get("/saved-datasets/{saved_id}/download")
async def download_saved_dataset(request: Request, saved_id: str, format: str = "csv"):
    return await _stream_saved_dataset_export(request, saved_id, format)


@router.get("/saved-datasets/{saved_id}/export")
async def export_saved_dataset(request: Request, saved_id: str, format: str = "csv"):
    return await _stream_saved_dataset_export(request, saved_id, format)


@router.get("/dataset/{dataset_id}/summary")
async def dataset_summary(request: Request, dataset_id: str, version: str = "raw"):
    dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
    return {"status": "ok", "summary": dataset.summary(version=version)}


@router.get("/dataset/{dataset_id}/spectra-preview")
async def dataset_spectra_preview(request: Request, dataset_id: str, limit: str = "5", strategy: str = "balanced_by_class", version: str = "raw"):
    dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
    if strategy in {"balanced", "balanced_by_class"} and dataset.y is None:
        strategy = "first_n"
    return {"status": "ok", **spectra_preview(dataset, limit=limit, strategy=strategy, version=version)}


@router.get("/dataset/{dataset_id}/config")
async def dataset_config(request: Request, dataset_id: str):
    dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
    return {"status": "ok", "dataset_id": dataset_id, "import_config": dataset.import_config()}


@router.get("/dataset/{dataset_id}/export")
async def dataset_export(request: Request, dataset_id: str, format: str = "csv", version: str = "raw"):
    try:
        dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
        if version == "processed" and dataset.processed_X is None:
            raise HTTPException(status_code=400, detail="Предобработанная версия датасета ещё не создана")
        frame = dataset.to_export_frame(version=version)
    except HTTPException:
        raise

    normalized_format = format.strip().lower()
    if normalized_format == "csv":
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        payload = io.BytesIO(buffer.getvalue().encode("utf-8-sig"))
        media_type = "text/csv"
        filename = f"correct_dataset_{version}.csv"
    elif normalized_format == "xlsx":
        payload = io.BytesIO()
        with pd.ExcelWriter(payload, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Dataset")
            pd.DataFrame([dataset.metadata_export(version=version)]).to_excel(writer, index=False, sheet_name="Metadata")
            pd.DataFrame([dataset.import_config()]).to_excel(writer, index=False, sheet_name="ImportConfig")
            if dataset.preprocessing_config:
                pd.DataFrame([dataset.preprocessing_config]).to_excel(writer, index=False, sheet_name="PreprocessingConfig")
        payload.seek(0)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"correct_dataset_{version}.xlsx"
    elif normalized_format == "zip":
        payload = io.BytesIO()
        csv_buffer = io.StringIO()
        frame.to_csv(csv_buffer, index=False)
        metadata = dataset.metadata_export(version=version)
        readme = (
            "Unified spectral dataset package\n"
            "correct_dataset.csv: one row = one spectrum; metadata columns first; spectral features after them.\n"
            "metadata.json: dataset summary and spectrometer parameters.\n"
            "import_config.json: settings used to import source files.\n"
            "preprocessing_config.json: preprocessing settings, if applied.\n"
        )
        with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("correct_dataset.csv", csv_buffer.getvalue().encode("utf-8-sig"))
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            archive.writestr("import_config.json", json.dumps(dataset.import_config(), ensure_ascii=False, indent=2))
            archive.writestr("preprocessing_config.json", json.dumps(dataset.preprocessing_config or {}, ensure_ascii=False, indent=2))
            archive.writestr("README.txt", readme)
        payload.seek(0)
        media_type = "application/zip"
        filename = f"correct_dataset_package_{version}.zip"
    else:
        raise HTTPException(status_code=400, detail="format должен быть csv, xlsx или zip")

    return StreamingResponse(
        payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/dataset/{dataset_id}/preprocess/preview")
async def dataset_preprocess_preview(request: Request, dataset_id: str, limit: str = "5", strategy: str = "balanced", preprocessing_config: Dict[str, Any] = Body(...)):
    try:
        dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
        return preview_preprocessing(dataset, preprocessing_config, limit=limit, strategy=strategy)
    except HTTPException:
        raise
    except DatasetImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected preprocessing preview error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка предпросмотра предобработки") from exc


@router.post("/dataset/{dataset_id}/preprocess/apply")
async def dataset_preprocess_apply(request: Request, dataset_id: str, preprocessing_config: Dict[str, Any] = Body(...)):
    try:
        user = current_analysis_user(request)
        dataset = get_owned_dataset(dataset_id, user)
        preprocess_start = time.perf_counter()
        result = apply_preprocessing(dataset, preprocessing_config)
        dataset.metadata["preprocessing_time_sec"] = round(time.perf_counter() - preprocess_start, 6)
        dataset.metadata["dataset_name"] = _dataset_short_name(dataset, "processed")
        persist_user_dataset(dataset, user)
        manager = user_run_manager(user)
        run_id = manager.new_run_id("processing")
        result_for_run = {key: value for key, value in result.items() if key != "run"}
        run = manager.save(
            {
                "run_id": run_id,
                "run_type": "processing",
                "dataset_id": dataset.metadata.get("saved_dataset_id") or dataset_id,
                "runtime_dataset_id": dataset_id,
                "dataset_name": dataset.metadata.get("dataset_name") or _dataset_short_name(dataset, "processed"),
                "used_processed_data": True,
                "target_name": dataset.target_name,
                "target_type": dataset.summary().get("target_type"),
                "class_distribution": dataset.summary().get("class_distribution"),
                "methods": [
                    result.get("preprocessing_summary", {}).get("baseline"),
                    result.get("preprocessing_summary", {}).get("smoothing"),
                    result.get("preprocessing_summary", {}).get("normalization"),
                ],
                "best_method": None,
                "status": "warning" if result.get("warnings") else "success",
                "summary": "Предобработка применена и обработанная версия датасета сохранена.",
                "results": result_for_run,
                "warnings": result.get("warnings") or [],
                "import_config": dataset.import_config(),
                "preprocessing_config": dataset.preprocessing_config or preprocessing_config,
                "spectrometer_parameters": dataset.metadata.get("measurement_metadata") or {},
                "user_id": user["user_id"],
                "username": user["username"],
            }
        )
        result["run_id"] = run_id
        result["run"] = {key: value for key, value in run.items() if key != "results"}
        return result
    except HTTPException:
        raise
    except DatasetImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected preprocessing apply error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка применения предобработки") from exc


@router.get("/dataset/{dataset_id}/preprocessing-config")
async def dataset_preprocessing_config(request: Request, dataset_id: str):
    dataset = get_owned_dataset(dataset_id, current_analysis_user(request))
    return {"status": "ok", "dataset_id": dataset_id, "preprocessing_config": dataset.preprocessing_config or {}}


@router.get("/dataset/{dataset_id}/preprocess/status")
async def dataset_preprocess_status(request: Request, dataset_id: str):
    return preprocessing_status(get_owned_dataset(dataset_id, current_analysis_user(request)))


@router.post("/dataset/{dataset_id}/train")
async def train_model_from_dataset(
    request: Request,
    dataset_id: str,
    model_type: str = Form(...),
    n_components: Optional[int] = Form(None),
    model_name: Optional[str] = Form(None),
    do_validation: bool = Form(True),
    test_size: float = Form(0.3),
    random_state: int = Form(42),
    use_processed: bool = Form(False),
    hyperparameter_search: bool = Form(False),
    refit_on_full_data: bool = Form(True),
):
    try:
        user = current_analysis_user(request)
        imported = get_owned_dataset(dataset_id, user)
        version = "processed" if use_processed and imported.processed_X is not None else "raw"
        raw_targets = "\n".join(imported.y) if imported.y else None
        payload = user_analysis_service(user).train_and_save(
            model_type=model_type,
            dataset=imported.to_analysis_dataset(version=version),
            raw_targets=raw_targets,
            n_components=n_components,
            model_name=model_name,
            do_validation=do_validation,
            test_size=test_size,
            random_state=random_state,
            hyperparameter_search=hyperparameter_search,
            refit_on_full_data=refit_on_full_data,
            metadata_extra={
                "dataset_id": dataset_id,
                "dataset_name": imported.metadata.get("dataset_name") or _dataset_short_name(imported, version),
                "source_dataset_name": ";".join(imported.source_files[:3]),
                "dataset_version": version,
                "used_processed_data": version == "processed",
                "target_name": imported.target_name,
                "target_type": imported.summary(version=version).get("target_type"),
                "class_distribution": imported.summary(version=version).get("class_distribution"),
                "axis_range": [imported.summary(version=version).get("axis_min"), imported.summary(version=version).get("axis_max")],
                "axis_min": imported.summary(version=version).get("axis_min"),
                "axis_max": imported.summary(version=version).get("axis_max"),
                "raw_spectral_axis": imported.axis.astype(float).tolist(),
                "import_config": imported.import_config(),
                "preprocessing_config": imported.preprocessing_config if version == "processed" else {},
                "preprocessing_summary": imported.summary(version=version).get("preprocessing") or {},
                "measurement_metadata": imported.metadata.get("measurement_metadata") or {},
                "spectrometer_parameters": imported.metadata.get("measurement_metadata") or {},
                "sample_metadata_preview": imported.metadata.get("sample_metadata_preview") or [],
                "slit_width_um_values": imported.metadata.get("slit_width_um_values") or [],
                "grating_lines_mm_values": imported.metadata.get("grating_lines_mm_values") or [],
                "exposure_time_s_values": imported.metadata.get("exposure_time_s_values") or [],
                "accumulations_values": imported.metadata.get("accumulations_values") or [],
                "power_mw_values": imported.metadata.get("power_mw_values") or [],
                "preprocessing": imported.preprocessing_config if version == "processed" else {},
                "user_id": user["user_id"],
                "username": user["username"],
            },
        )
        return {"status": "ok", "dataset_id": dataset_id, "dataset_version": version, **payload}
    except HTTPException:
        raise
    except (SpectrumValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected dataset train error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка обучения по импортированному датасету") from exc


def _validation_kwargs(
    validation: Dict[str, Any],
    hyperparameter_search_default: bool,
    do_validation_default: bool = True,
    refit_on_full_data_default: bool = True,
    quick_mode_default: bool = True,
) -> Dict[str, Any]:
    """Extracts do_validation/hyperparameter_search/kfold/bootstrap/permutation/n_*/quick_mode
    overrides from the request's `validation` dict, falling back to the given per-call defaults
    (e.g. ВКР-mode defaults for comparison runs) for anything the caller didn't explicitly specify.
    permutation_test is left as None (auto: plsda/pls only) unless the caller explicitly passes
    true/false.

    quick_mode is the new DEFAULT (True): n_bootstrap=100, n_permutations=50, and a reduced
    GridSearchCV grid for random_forest/svm/svr (see hyperparameter_search.py). The frontend's
    "Подробный расчёт" checkbox sends validation.quick_mode=false to switch to the full ВКР
    settings (n_bootstrap=1000, n_permutations=200, full grid) - explicit n_bootstrap/n_permutations
    in the request always win over either default."""
    permutation_raw = validation.get("permutation_test")
    quick_mode = bool(validation.get("quick_mode", quick_mode_default))
    default_n_bootstrap = 100 if quick_mode else 1000
    default_n_permutations = 50 if quick_mode else 200
    return {
        "do_validation": bool(validation.get("enabled", do_validation_default)),
        "test_size": float(validation.get("test_size", 0.3)),
        "random_state": int(validation.get("random_state", 42)),
        "hyperparameter_search": bool(validation.get("hyperparameter_search", hyperparameter_search_default)),
        "refit_on_full_data": bool(validation.get("refit_on_full_data", refit_on_full_data_default)),
        "kfold_enabled": bool(validation.get("kfold", True)),
        "bootstrap_enabled": bool(validation.get("bootstrap", True)),
        "permutation_enabled": bool(permutation_raw) if permutation_raw is not None else None,
        "n_splits": int(validation.get("n_splits", 5)),
        "n_bootstrap": int(validation.get("n_bootstrap", default_n_bootstrap)),
        "n_permutations": int(validation.get("n_permutations", default_n_permutations)),
        "quick_mode": quick_mode,
    }


_background_tasks: set = set()


def _launch_background(coro) -> None:
    """Fire-and-forget a coroutine on the running event loop, keeping a reference so it isn't
    garbage-collected mid-flight. The route that creates the task has already returned a run_id
    to the client by the time this runs; any exception is caught inside the coroutine itself and
    recorded via AnalysisRunManager.update_run_status (status=error) - it must never escape here."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_single_training_background(payload: ModelTrainRequest, imported: Any, user: Dict[str, Any], run_id: str) -> None:
    manager = user_run_manager(user)

    def on_stage(stage: str) -> None:
        manager.update_run_status(run_id, stage=stage)

    try:
        await asyncio.to_thread(_train_single_sync, payload, imported, user, run_id, manager, on_stage)
        manager.update_run_status(run_id, status="success", stage="done")
    except Exception as exc:
        logger.exception("Background single-model training failed: %s", run_id)
        manager.update_run_status(run_id, status="error", stage="error", error=str(exc))


def _train_single_sync(
    payload: ModelTrainRequest,
    imported: Any,
    user: Dict[str, Any],
    run_id: str,
    manager: AnalysisRunManager,
    on_stage,
) -> None:
    validation = payload.validation or {}
    version = "processed" if payload.use_processed else "raw"
    summary = imported.summary(version=version)
    trained = user_analysis_service(user).train_and_save(
        model_type=payload.model_type,
        dataset=imported.to_analysis_dataset(version=version),
        raw_targets="\n".join(imported.y) if imported.y else None,
        n_components=payload.n_components,
        model_name=payload.model_name,
        model_params=payload.model_params,
        on_stage=on_stage,
        **_validation_kwargs(validation, hyperparameter_search_default=payload.hyperparameter_search, refit_on_full_data_default=payload.refit_on_full_data),
        metadata_extra={
            "dataset_id": imported.metadata.get("saved_dataset_id") or payload.dataset_id,
            "runtime_dataset_id": payload.dataset_id,
            "dataset_name": imported.metadata.get("dataset_name") or _dataset_short_name(imported, version),
            "source_dataset_name": ";".join(imported.source_files[:3]),
            "dataset_version": version,
            "used_processed_data": version == "processed",
            "target_name": imported.target_name,
            "target_type": summary.get("target_type"),
            "class_distribution": summary.get("class_distribution"),
            "axis_range": [summary.get("axis_min"), summary.get("axis_max")],
            "axis_min": summary.get("axis_min"),
            "axis_max": summary.get("axis_max"),
            "raw_spectral_axis": imported.axis.astype(float).tolist(),
            "n_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "import_config": imported.import_config(),
            "preprocessing_config": imported.preprocessing_config if version == "processed" else {},
            "preprocessing_summary": summary.get("preprocessing") or {},
            "measurement_metadata": imported.metadata.get("measurement_metadata") or {},
            "spectrometer_parameters": imported.metadata.get("measurement_metadata") or {},
            "sample_metadata_preview": imported.metadata.get("sample_metadata_preview") or [],
            "slit_width_um_values": imported.metadata.get("slit_width_um_values") or [],
            "grating_lines_mm_values": imported.metadata.get("grating_lines_mm_values") or [],
            "exposure_time_s_values": imported.metadata.get("exposure_time_s_values") or [],
            "accumulations_values": imported.metadata.get("accumulations_values") or [],
            "power_mw_values": imported.metadata.get("power_mw_values") or [],
            "preprocessing": imported.preprocessing_config if version == "processed" else {},
            "run_id": run_id,
            "user_id": user["user_id"],
            "username": user["username"],
        },
    )
    _save_single_model_run(
        manager=manager,
        run_id=run_id,
        payload=payload,
        imported=imported,
        summary=summary,
        version=version,
        trained=trained,
    )


async def _run_comparison_background(payload: ModelTrainRequest, imported: Any, normalized_type: str, user: Dict[str, Any], run_id: str) -> None:
    manager = user_run_manager(user)

    def on_stage(stage: str, **extra: Any) -> None:
        progress = {key: extra[key] for key in ("method", "method_index", "methods_total") if key in extra}
        manager.update_run_status(run_id, stage=stage, progress=progress)

    try:
        await asyncio.to_thread(_train_comparison, payload, imported, normalized_type, user, run_id, on_stage)
        manager.update_run_status(run_id, status="success", stage="done")
    except Exception as exc:
        logger.exception("Background comparison training failed: %s", run_id)
        manager.update_run_status(run_id, status="error", stage="error", error=str(exc))


@router.post("/model/train")
async def train_model_json(request: Request, payload: ModelTrainRequest):
    try:
        user = current_analysis_user(request)
        imported = get_owned_dataset(payload.dataset_id, user)
        if payload.use_processed and imported.processed_X is None:
            raise ValueError("Предобработанная версия датасета ещё не создана.")
        normalized_type = payload.model_type.lower().replace("-", "_")
        manager = user_run_manager(user)
        if normalized_type in {"compare_classification", "compare_regression", "compare_clustering"}:
            run_id = manager.new_run_id("comparison")
            manager.start_run(run_id, "comparison", extra={"dataset_id": payload.dataset_id, "model_type": normalized_type})
            _launch_background(_run_comparison_background(payload, imported, normalized_type, user, run_id))
            return {"status": "running", "run_id": run_id, "run_type": "comparison", "model_type": normalized_type}
        supervised_types = {"pls", "pls_regression", "plsda", "pls_da", "svm", "random_forest", "decision_tree", "svr"}
        if normalized_type in supervised_types and not imported.y:
            raise ValueError("Для выбранного supervised-метода нужен target. Выберите target-колонку при импорте датасета.")

        run_id = manager.new_run_id("single_model")
        manager.start_run(run_id, "single_model", extra={"dataset_id": payload.dataset_id, "model_type": payload.model_type})
        _launch_background(_run_single_training_background(payload, imported, user, run_id))
        return {"status": "running", "run_id": run_id, "run_type": "single_model", "model_type": payload.model_type}
    except HTTPException:
        raise
    except (SpectrumValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected JSON model train error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка обучения модели") from exc


@router.post("/train/compare")
async def train_model_compare(request: Request, payload: ModelTrainRequest):
    """Explicit comparison-mode entry point (same body shape as /model/train); model_type must be
    one of compare_classification/compare_regression/compare_clustering. Kept alongside
    /model/train (which also dispatches to the same background comparison job for compare_*
    model_type) so the frontend's 'Сравнить методы' action has a self-describing route to call."""
    normalized_type = payload.model_type.lower().replace("-", "_")
    if normalized_type not in {"compare_classification", "compare_regression", "compare_clustering"}:
        raise HTTPException(status_code=400, detail="/train/compare ожидает model_type из compare_classification/compare_regression/compare_clustering")
    return await train_model_json(request, payload)


def _train_comparison(payload: ModelTrainRequest, imported: Any, normalized_type: str, user: Dict[str, Any], run_id: str, on_stage=None) -> Dict[str, Any]:
    comparison_start = time.perf_counter()
    version = "processed" if payload.use_processed else "raw"
    summary = imported.summary(version=version)
    raw_targets = "\n".join(imported.y) if imported.y else None
    if normalized_type in {"compare_classification", "compare_regression"} and not raw_targets:
        raise ValueError("Для сравнения supervised-методов нужен target из импортированного датасета.")
    if normalized_type == "compare_classification" and summary.get("target_type") != "categorical":
        raise ValueError("Для сравнения классификаторов нужен категориальный target.")
    if normalized_type == "compare_regression" and summary.get("target_type") != "numeric":
        raise ValueError("Для сравнения регрессионных методов нужен числовой target.")

    methods = {
        "compare_classification": ["plsda", "svm", "random_forest", "decision_tree"],
        "compare_regression": ["pls", "svr"],
        "compare_clustering": ["kmeans", "hca"],
    }[normalized_type]
    validation = payload.validation or {}
    # ВКР mode: for supervised comparisons, an honest holdout split and GridSearchCV are ON by
    # default (the user must explicitly pass validation.enabled=false / hyperparameter_search=false
    # to opt out). Clustering has no supervised holdout/search, so it keeps its prior defaults.
    is_supervised_comparison = normalized_type in {"compare_classification", "compare_regression"}
    do_validation_default = is_supervised_comparison
    hyperparameter_search_default = is_supervised_comparison
    manager = user_run_manager(user)
    service = user_analysis_service(user)
    rows: List[Dict[str, Any]] = []
    saved_models: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for method_index, method in enumerate(methods):
        def method_stage(stage: str, _method: str = method, _index: int = method_index) -> None:
            if on_stage is not None:
                on_stage(stage, method=_method, method_index=_index + 1, methods_total=len(methods))
        try:
            trained = service.train_and_save(
                model_type=method,
                dataset=imported.to_analysis_dataset(version=version),
                raw_targets=raw_targets,
                n_components=payload.n_components,
                model_name=f"{payload.model_name or 'comparison'}_{method}",
                model_params=payload.model_params,
                on_stage=method_stage,
                **_validation_kwargs(
                    validation,
                    hyperparameter_search_default=hyperparameter_search_default,
                    do_validation_default=do_validation_default,
                    refit_on_full_data_default=payload.refit_on_full_data,
                ),
                model_id_override=f"{run_id}_{method}",
                metadata_extra={
                    "dataset_id": imported.metadata.get("saved_dataset_id") or payload.dataset_id,
                    "runtime_dataset_id": payload.dataset_id,
                    "dataset_version": version,
                    "used_processed_data": version == "processed",
                    "target_name": imported.target_name,
                    "target_type": summary.get("target_type"),
                    "class_distribution": summary.get("class_distribution"),
                    "axis_range": [summary.get("axis_min"), summary.get("axis_max")],
                    "axis_min": summary.get("axis_min"),
                    "axis_max": summary.get("axis_max"),
                    "raw_spectral_axis": imported.axis.astype(float).tolist(),
                    "n_samples": summary.get("n_samples"),
                    "n_features": summary.get("n_features"),
                    "import_config": imported.import_config(),
                    "preprocessing_config": imported.preprocessing_config if version == "processed" else {},
                    "measurement_metadata": imported.metadata.get("measurement_metadata") or {},
                    "spectrometer_parameters": imported.metadata.get("measurement_metadata") or {},
                    "sample_metadata_preview": imported.metadata.get("sample_metadata_preview") or [],
                    "slit_width_um_values": imported.metadata.get("slit_width_um_values") or [],
                    "grating_lines_mm_values": imported.metadata.get("grating_lines_mm_values") or [],
                    "exposure_time_s_values": imported.metadata.get("exposure_time_s_values") or [],
                    "accumulations_values": imported.metadata.get("accumulations_values") or [],
                    "power_mw_values": imported.metadata.get("power_mw_values") or [],
                    "preprocessing": imported.preprocessing_config if version == "processed" else {},
                    "comparison_group": normalized_type,
                    "run_id": run_id,
                    "user_id": user["user_id"],
                    "username": user["username"],
                },
            )
            metrics = trained.get("metrics") or {}
            saved_model = trained.get("saved_model") or {}
            full_dataset_outputs = trained.get("full_dataset_outputs") or {}
            row = {
                "method": method,
                "status": "success",
                "model_id": saved_model.get("model_id"),
                "model_type": saved_model.get("model_type"),
                "metrics": metrics,
                # metrics.confusion_matrix is the TEST (holdout) confusion matrix when validation
                # succeeded, or null when it didn't run - it must NEVER silently fall back to the
                # full-dataset matrix (that one lives separately in full_dataset_confusion_matrix).
                "confusion_matrix": metrics.get("confusion_matrix"),
                "full_dataset_confusion_matrix": full_dataset_outputs.get("confusion_matrix"),
                "params": saved_model.get("model_params") or saved_model.get("params") or {},
                "best_params": trained.get("best_params"),
                "hyperparameter_search": trained.get("hyperparameter_search") or {},
                "performance": trained.get("performance") or {},
                "training_policy": trained.get("training_policy") or {},
                "validation": trained.get("validation") or {},
                "outputs": trained.get("outputs") or {},
                "test_outputs": trained.get("test_outputs") or {},
                "full_dataset_outputs": full_dataset_outputs,
                "warnings": trained.get("warnings") or [],
                "plot": trained.get("plot") or {},
                **metrics,
            }
            rows.append(row)
            saved_models.append(saved_model)
        except Exception as exc:
            logger.exception("Comparison method failed: %s", method)
            errors.append({"method": method, "error": str(exc)})
    if not rows:
        raise ValueError("Не удалось выполнить ни один метод сравнения.")
    model_ids = [row.get("model_id") for row in rows if row.get("model_id")]
    if len(model_ids) != len(set(model_ids)):
        logger.error("Duplicate model_id detected within comparison run %s: %s", run_id, model_ids)
    best_method_selection = _select_best_method(normalized_type, rows)
    best_method = best_method_selection.get("selected_method")
    comparison_plots = [
        {"plot_id": f"{row.get('method')}_plot", "title": row.get("method"), **row.get("plot")}
        for row in rows
        if row.get("plot")
    ]
    run_status = "warning" if errors else "success"
    comparison_time_sec = time.perf_counter() - comparison_start
    per_method_peak_memory = [row.get("performance", {}).get("peak_memory_mb") for row in rows if isinstance(row.get("performance", {}).get("peak_memory_mb"), (int, float))]
    pipeline_performance = {
        "import_time_sec": imported.metadata.get("import_time_sec"),
        "import_time_source": "measured_at_import" if imported.metadata.get("import_time_sec") is not None else "not_measured_in_current_run",
        "preprocessing_time_sec": imported.metadata.get("preprocessing_time_sec"),
        "preprocessing_time_source": "measured_at_preprocessing" if imported.metadata.get("preprocessing_time_sec") is not None else "not_measured_in_current_run",
        "comparison_time_sec": round(comparison_time_sec, 6),
        "export_time_sec": None,
        "export_time_source": "not_measured_in_current_run",
        "peak_memory_mb": max(per_method_peak_memory) if per_method_peak_memory else None,
        "peak_memory_source": "max(per_method performance.peak_memory_mb)" if per_method_peak_memory else "not_measured_in_current_run",
    }
    comparison_performance = {
        "total_time_sec": round(comparison_time_sec, 6),
        "peak_memory_mb": max(per_method_peak_memory) if per_method_peak_memory else None,
        "peak_memory_source": "max(per_method performance.peak_memory_mb)" if per_method_peak_memory else "not_measured_in_current_run",
        "methods_count": len(methods),
        "successful_methods": len(rows),
        "failed_methods": len(errors),
    }
    vkr_tables = _build_vkr_tables(rows, dataset_name=imported.metadata.get("dataset_name") or _dataset_short_name(imported, version), n_samples=summary.get("n_samples"), n_features=summary.get("n_features"))
    run = manager.save(
        {
            "run_id": run_id,
            "run_type": "comparison",
            "dataset_id": payload.dataset_id,
            "dataset_name": imported.metadata.get("dataset_name") or _dataset_short_name(imported, version),
            "used_processed_data": version == "processed",
            "target_name": imported.target_name,
            "target_type": summary.get("target_type"),
            "class_distribution": summary.get("class_distribution"),
            "n_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "methods": methods,
            "best_method": best_method,
            "best_method_selection": best_method_selection,
            "status": run_status,
            "summary": f"Сравнение методов завершено. Лучший метод: {best_method or 'н/д'}.",
            "results": {"method_results": rows, "errors": errors},
            "plots": comparison_plots,
            "warnings": [f"{item['method']}: {item['error']}" for item in errors],
            "import_config": imported.import_config(),
            "preprocessing_config": imported.preprocessing_config if version == "processed" else {},
            "spectrometer_parameters": imported.metadata.get("measurement_metadata") or {},
            "pipeline_performance": pipeline_performance,
            "comparison_performance": comparison_performance,
            "vkr_tables": vkr_tables,
            "user_id": user["user_id"],
            "username": user["username"],
        }
    )
    return {
        "status": "warning" if errors else "success",
        "run_type": "comparison",
        "model_type": normalized_type,
        "task_type": "comparison",
        "run_id": run_id,
        "run": run,
        "dataset_id": payload.dataset_id,
        "dataset_version": version,
        "summary": summary,
        "best_method": best_method,
        "best_method_selection": best_method_selection,
        "comparison_table": rows,
        "metrics": {"comparison": rows},
        "result": {"comparison": rows, "method_results": rows, "errors": errors},
        "plots": comparison_plots,
        "saved_models": saved_models,
        "pipeline_performance": pipeline_performance,
        "comparison_performance": comparison_performance,
        "vkr_tables": vkr_tables,
        "warnings": [f"{item['method']}: {item['error']}" for item in errors],
    }


def _select_best_method(run_type: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Picks the best method with an explicit, documented tie-breaker chain (citable in a thesis).

    Each tie-breaker is applied in order only to break ties left by the previous one; this makes
    the choice reproducible and auditable instead of an opaque max() over a single metric.
    """
    candidates = [row for row in rows if row.get("status") == "success"]
    if not candidates:
        return {"primary_metric": None, "tie_breakers": [], "selected_method": None, "reason": "Нет успешно выполненных методов для сравнения."}

    def metric_value(row: Dict[str, Any], key: str) -> Optional[float]:
        value = (row.get("metrics") or {}).get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def kfold_value(row: Dict[str, Any], key: str) -> Optional[float]:
        value = ((row.get("validation") or {}).get("kfold") or {}).get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def perf_value(row: Dict[str, Any], key: str) -> Optional[float]:
        value = (row.get("performance") or {}).get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    if run_type == "compare_classification":
        primary_metric = "f1_macro"
        simplicity_order = ["decision_tree", "plsda", "svm", "random_forest"]
        tie_breakers = [
            "f1_macro (максимум)",
            "kfold.mean f1_macro (максимум)",
            "kfold.std f1_macro (минимум)",
            "performance.predict_time_one_sample_ms (минимум)",
            "performance.fit_time_sec (минимум)",
            f"порядок простоты: {', '.join(simplicity_order)}",
        ]
        sort_key = lambda row: (  # noqa: E731
            -(metric_value(row, "f1_macro") if metric_value(row, "f1_macro") is not None else -1.0),
            -(kfold_value(row, "mean") if kfold_value(row, "mean") is not None else -1.0),
            kfold_value(row, "std") if kfold_value(row, "std") is not None else 1.0,
            perf_value(row, "predict_time_one_sample_ms") if perf_value(row, "predict_time_one_sample_ms") is not None else float("inf"),
            perf_value(row, "fit_time_sec") if perf_value(row, "fit_time_sec") is not None else float("inf"),
            simplicity_order.index(row.get("method")) if row.get("method") in simplicity_order else 99,
        )
    elif run_type == "compare_regression":
        primary_metric = "rmse"
        simplicity_order = ["pls", "svr"]
        tie_breakers = [
            "rmse (минимум)",
            "r2 (максимум)",
            "kfold.rmse_mean (минимум)",
            "kfold.rmse_std (минимум)",
            "performance.predict_time_one_sample_ms (минимум)",
            "performance.fit_time_sec (минимум)",
            f"порядок простоты: {', '.join(simplicity_order)}",
        ]
        sort_key = lambda row: (  # noqa: E731
            metric_value(row, "rmse") if metric_value(row, "rmse") is not None else float("inf"),
            -(metric_value(row, "r2") if metric_value(row, "r2") is not None else -float("inf")),
            kfold_value(row, "rmse_mean") if kfold_value(row, "rmse_mean") is not None else float("inf"),
            kfold_value(row, "rmse_std") if kfold_value(row, "rmse_std") is not None else float("inf"),
            perf_value(row, "predict_time_one_sample_ms") if perf_value(row, "predict_time_one_sample_ms") is not None else float("inf"),
            perf_value(row, "fit_time_sec") if perf_value(row, "fit_time_sec") is not None else float("inf"),
            simplicity_order.index(row.get("method")) if row.get("method") in simplicity_order else 99,
        )
    elif run_type == "compare_clustering":
        primary_metric = "silhouette_score"
        tie_breakers = ["silhouette_score (максимум)", "adjusted_rand_score (максимум)", "normalized_mutual_info_score (максимум)"]
        sort_key = lambda row: (  # noqa: E731
            -(metric_value(row, "silhouette_score") if metric_value(row, "silhouette_score") is not None else -1.0),
            -(metric_value(row, "adjusted_rand_score") if metric_value(row, "adjusted_rand_score") is not None else -1.0),
            -(metric_value(row, "normalized_mutual_info_score") if metric_value(row, "normalized_mutual_info_score") is not None else -1.0),
        )
    else:
        return {"primary_metric": None, "tie_breakers": [], "selected_method": candidates[0].get("method"), "reason": "Тип сравнения не поддерживает формализованный выбор; взят первый успешный метод."}

    ranked = sorted(candidates, key=sort_key)
    winner = ranked[0]
    reason = (
        f"Метод «{winner.get('method')}» выбран как лучший по метрике {primary_metric}="
        f"{metric_value(winner, primary_metric) if run_type != 'compare_clustering' else metric_value(winner, 'silhouette_score')}; "
        f"порядок tie-breakers: {' -> '.join(tie_breakers)}."
    )
    return {
        "primary_metric": primary_metric,
        "tie_breakers": tie_breakers,
        "selected_method": winner.get("method"),
        "ranked_methods": [row.get("method") for row in ranked],
        "reason": reason,
    }


def _build_vkr_tables(rows: List[Dict[str, Any]], dataset_name: Optional[str], n_samples: Optional[int], n_features: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    """vkr_tables block for comparison runs: classification_comparison/regression_comparison +
    performance_summary, ready for JSON/CSV/XLSX export. Reuses build_thesis_tables's row-shaping
    logic (single source of truth with the /runs/{run_id}/thesis-tables route) and only renames keys
    to the vkr_tables naming used by this comparison-readiness spec."""
    shim_run = {"dataset_name": dataset_name, "n_samples": n_samples, "n_features": n_features, "results": {"method_results": rows}}
    tables = build_thesis_tables(shim_run)
    return {
        "classification_comparison": tables.get("classification") or [],
        "regression_comparison": tables.get("regression") or [],
        "performance_summary": tables.get("performance") or [],
    }


def _save_single_model_run(
    *,
    manager: AnalysisRunManager,
    run_id: str,
    payload: ModelTrainRequest,
    imported: Any,
    summary: Dict[str, Any],
    version: str,
    trained: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = trained.get("metrics") or {}
    saved = trained.get("saved_model") or {}
    return manager.save(
        {
            "run_id": run_id,
            "run_type": "single_model",
            "dataset_id": payload.dataset_id,
            "dataset_name": imported.metadata.get("dataset_name") or _dataset_short_name(imported, version),
            "used_processed_data": version == "processed",
            "target_name": imported.target_name,
            "target_type": summary.get("target_type"),
            "class_distribution": summary.get("class_distribution"),
            "n_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "methods": [trained.get("model_type") or payload.model_type],
            "best_method": trained.get("model_type") or payload.model_type,
            "status": "success",
            "summary": f"Метод {trained.get('model_type') or payload.model_type} выполнен и модель сохранена.",
            "results": {
                "metrics": metrics,
                "saved_model": saved,
                "result": trained.get("result"),
                "performance": trained.get("performance") or {},
                "best_params": trained.get("best_params"),
                "hyperparameter_search": trained.get("hyperparameter_search") or {},
                "training_policy": trained.get("training_policy") or {},
                "reproducibility": trained.get("reproducibility") or {},
                "outputs": trained.get("outputs") or {},
                "test_outputs": trained.get("test_outputs") or {},
                "full_dataset_outputs": trained.get("full_dataset_outputs") or {},
                "validation": trained.get("validation") or {},
            },
            "plots": [{"plot_id": f"{trained.get('model_type') or payload.model_type}_plot", "title": trained.get("model_type") or payload.model_type, **(trained.get("plot") or {})}] if trained.get("plot") else [],
            "warnings": trained.get("warnings") or [],
            "import_config": imported.import_config(),
            "preprocessing_config": imported.preprocessing_config if version == "processed" else {},
            "spectrometer_parameters": imported.metadata.get("measurement_metadata") or {},
            "user_id": trained.get("model_metadata", {}).get("user_id"),
            "username": trained.get("model_metadata", {}).get("username"),
        }
    )


@router.post("/train")
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_values: Optional[str] = Form(None),
    n_components: Optional[int] = Form(None),
    model_name: Optional[str] = Form(None),
    do_validation: bool = Form(True),
    test_size: float = Form(0.3),
    random_state: int = Form(42),
    hyperparameter_search: bool = Form(False),
    refit_on_full_data: bool = Form(True),
):
    try:
        user = current_analysis_user(request)
        raw = await file.read()
        service = user_analysis_service(user)
        dataset = service.parse_file(raw=raw, filename=file.filename or "")
        payload = service.train_and_save(
            model_type=model_type,
            dataset=dataset,
            raw_targets=target_values,
            n_components=n_components,
            model_name=model_name,
            do_validation=do_validation,
            test_size=test_size,
            random_state=random_state,
            hyperparameter_search=hyperparameter_search,
            refit_on_full_data=refit_on_full_data,
            metadata_extra={"user_id": user["user_id"], "username": user["username"]},
        )
        return {"status": "ok", **payload}
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected train error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка обучения модели") from exc


@router.get("/runs")
async def list_runs(request: Request):
    return {"status": "ok", "runs": user_run_manager(current_analysis_user(request)).list()}


@router.get("/experiments")
async def list_experiments(request: Request):
    return {"status": "ok", "experiments": user_run_manager(current_analysis_user(request)).list()}


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    try:
        return {"status": "ok", "run": user_run_manager(current_analysis_user(request)).get(run_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments/{run_id}")
async def get_experiment(request: Request, run_id: str):
    return await get_run(request, run_id)


@router.get("/runs/{run_id}/status")
async def get_run_status(request: Request, run_id: str):
    """Polled by the frontend every 1-2s while a training/comparison job runs in the
    background. Returns the current stage and overall status (running/success/error);
    once status is success, the frontend follows up with GET /runs/{run_id} to fetch and
    render the full saved result."""
    try:
        status = user_run_manager(current_analysis_user(request)).get_run_status(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "run_status": status}


@router.get("/runs/{run_id}/json")
async def get_run_json(request: Request, run_id: str):
    """Same payload as GET /runs/{run_id}, served as a downloadable .json attachment for the
    'Скачать полный JSON' button - the frontend reads the saved run, never recomputes it."""
    try:
        run = user_run_manager(current_analysis_user(request)).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    return StreamingResponse(
        io.BytesIO(json.dumps(run, ensure_ascii=False, indent=2).encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


@router.get("/runs/{run_id}/performance")
async def get_run_performance(request: Request, run_id: str):
    """Performance-only view of a saved run: per-method performance plus the run-level
    pipeline_performance/comparison_performance blocks, for the 'Производительность' UI section."""
    try:
        run = user_run_manager(current_analysis_user(request)).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    results = run.get("results") or {}
    method_rows = results.get("method_results") or results.get("comparison") or []
    if method_rows:
        methods = [
            {
                "method": row.get("method") or row.get("model_type"),
                "performance": row.get("performance") or {},
                "hyperparameter_search": row.get("hyperparameter_search") or {},
            }
            for row in method_rows
        ]
    else:
        methods = [
            {
                "method": run.get("best_method"),
                "performance": results.get("performance") or {},
                "hyperparameter_search": results.get("hyperparameter_search") or {},
            }
        ]
    return {
        "status": "ok",
        "run_id": run_id,
        "methods": methods,
        "pipeline_performance": run.get("pipeline_performance") or {},
        "comparison_performance": run.get("comparison_performance") or {},
    }


@router.delete("/runs/{run_id}")
async def delete_run(request: Request, run_id: str):
    try:
        return {"status": "ok", **user_run_manager(current_analysis_user(request)).delete(run_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/experiments/{run_id}")
async def delete_experiment(request: Request, run_id: str):
    return await delete_run(request, run_id)


@router.get("/runs/{run_id}/export")
async def export_run(request: Request, run_id: str):
    try:
        payload = user_run_manager(current_analysis_user(request)).export_zip(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'},
    )


@router.get("/runs/{run_id}/csv")
async def export_run_csv(request: Request, run_id: str):
    try:
        csv_payload = user_run_manager(current_analysis_user(request)).csv_bytes(run_id)
        run = user_run_manager(current_analysis_user(request)).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    return StreamingResponse(
        io.BytesIO(csv_payload),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{_build_run_csv_filename(run)}"'},
    )


@router.get("/runs/{run_id}/xlsx")
async def export_run_xlsx(request: Request, run_id: str):
    """Same comparison/single-model row data as /runs/{run_id}/csv, as an .xlsx workbook."""
    try:
        rows = user_run_manager(current_analysis_user(request)).csv_rows(run_id)
        run = user_run_manager(current_analysis_user(request)).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data = rows if rows else [{"message": "Нет данных для экспорта"}]
        pd.DataFrame(data).to_excel(writer, sheet_name="comparison_table", index=False)
    buffer.seek(0)
    filename = _build_run_csv_filename(run).rsplit(".", 1)[0] + ".xlsx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _format_ci(ci: Any) -> Optional[str]:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    try:
        return f"[{float(ci[0]):.4g}, {float(ci[1]):.4g}]"
    except (TypeError, ValueError):
        return None


def _permutation_summary(permutation: Dict[str, Any]) -> Any:
    if not isinstance(permutation, dict):
        return "skipped"
    if permutation.get("status") == "ok":
        return permutation.get("p_value")
    return permutation.get("status") or "skipped"


def _thesis_method_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = run.get("results") or {}
    rows = results.get("method_results") or results.get("comparison") or []
    if rows:
        return rows
    saved_model = results.get("saved_model") or {}
    return [
        {
            "method": saved_model.get("model_type") or run.get("best_method"),
            "model_type": saved_model.get("model_type") or run.get("best_method"),
            "metrics": results.get("metrics") or {},
            "best_params": results.get("best_params"),
            "performance": results.get("performance") or {},
            "validation": results.get("validation") or {},
        }
    ]


def build_thesis_tables(run: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Builds the three VKR-ready tables described in the audit follow-up (classification,
    regression, performance) from a saved run's method_results/results. Used by both the
    JSON/CSV/XLSX export route and (indirectly) by the frontend preview."""
    classification_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    dataset_label = run.get("dataset_name") or run.get("dataset_id") or "dataset"
    data_size = f"{run.get('n_samples')}x{run.get('n_features')}" if run.get("n_samples") and run.get("n_features") else None

    for row in _thesis_method_rows(run):
        metrics = row.get("metrics") or {}
        validation = row.get("validation") or {}
        performance = row.get("performance") or {}
        kfold = validation.get("kfold") or {}
        bootstrap = validation.get("bootstrap") or validation.get("bootstrap_ci") or {}
        permutation = validation.get("permutation_test") or {}
        method_name = row.get("method") or row.get("model_type") or "unknown"
        best_params = row.get("best_params") or {}

        if "accuracy" in metrics or "f1_macro" in metrics:
            classification_rows.append(
                {
                    "method": method_name,
                    "best_params": json.dumps(best_params, ensure_ascii=False) if best_params else "",
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "kfold_f1_mean": kfold.get("mean"),
                    "kfold_f1_std": kfold.get("std"),
                    "bootstrap_95ci": _format_ci(bootstrap.get("ci_95")),
                    "permutation_p_value_or_status": _permutation_summary(permutation),
                    "fit_time_sec": performance.get("fit_time_sec"),
                    "inference_time_ms": performance.get("predict_time_one_sample_ms"),
                    "peak_memory_mb": performance.get("peak_memory_mb"),
                }
            )
        elif "rmse" in metrics or "r2" in metrics:
            regression_rows.append(
                {
                    "method": method_name,
                    "best_params": json.dumps(best_params, ensure_ascii=False) if best_params else "",
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "kfold_rmse_mean": kfold.get("rmse_mean"),
                    "kfold_rmse_std": kfold.get("rmse_std"),
                    "kfold_r2_mean": kfold.get("r2_mean"),
                    "kfold_r2_std": kfold.get("r2_std"),
                    "bootstrap_95ci": _format_ci(bootstrap.get("ci_95")),
                    "permutation_p_value_or_status": _permutation_summary(permutation),
                    "fit_time_sec": performance.get("fit_time_sec"),
                    "inference_time_ms": performance.get("predict_time_one_sample_ms"),
                    "peak_memory_mb": performance.get("peak_memory_mb"),
                }
            )

        performance_rows.append(
            {
                "operation": f"train+validate ({method_name})",
                "dataset": dataset_label,
                "data_size": data_size,
                "execution_time_sec": performance.get("total_time_sec"),
                "peak_memory_mb": performance.get("peak_memory_mb"),
            }
        )

    return {"classification": classification_rows, "regression": regression_rows, "performance": performance_rows}


@router.get("/runs/{run_id}/thesis-tables")
async def export_run_thesis_tables(request: Request, run_id: str, format: str = Query("json", pattern="^(json|csv|xlsx)$")):
    """Flat, VKR-ready tables (classification/regression/performance) built from a saved run.

    format=json returns all three tables in one object; format=csv returns a ZIP with one CSV
    per non-empty table; format=xlsx returns a single workbook with one sheet per table.
    """
    try:
        run = user_run_manager(current_analysis_user(request)).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    tables = build_thesis_tables(run)

    if format == "json":
        return {"status": "ok", "run_id": run_id, "tables": tables}

    if format == "csv":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, rows in tables.items():
                if not rows:
                    continue
                csv_buffer = io.StringIO()
                pd.DataFrame(rows).to_csv(csv_buffer, index=False)
                archive.writestr(f"{name}.csv", csv_buffer.getvalue())
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_thesis_tables.zip"'},
        )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        any_sheet = False
        for name, rows in tables.items():
            if not rows:
                continue
            pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31], index=False)
            any_sheet = True
        if not any_sheet:
            pd.DataFrame([{"message": "Нет данных для таблиц ВКР в этом запуске"}]).to_excel(writer, sheet_name="info", index=False)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_thesis_tables.xlsx"'},
    )


@router.get("/experiments/{run_id}/export")
async def export_experiment(request: Request, run_id: str):
    return await export_run(request, run_id)


@router.get("/experiments/{run_id}/csv")
async def export_experiment_csv(request: Request, run_id: str):
    return await export_run_csv(request, run_id)


@router.post("/configs")
async def save_analysis_config(request: Request, config: Dict[str, Any] = Body(...)):
    user = current_analysis_user(request)
    configs_dir = user_root(user) / "configs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = configs_dir / f"config_{timestamp}.json"
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "user_id": user["user_id"], "username": user["username"], "config": config}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "filename": path.name, "path": str(path.relative_to(BASE_DIR)), **payload}


@router.get("/configs")
async def list_analysis_configs(request: Request):
    configs_dir = user_root(current_analysis_user(request)) / "configs"
    items = []
    for path in sorted(configs_dir.glob("config_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        items.append({"filename": path.name, "created_at": payload.get("created_at"), "config": payload.get("config")})
    return {"status": "ok", "configs": items}


@router.get("/configs/{filename}")
async def get_analysis_config(request: Request, filename: str):
    if not filename.startswith("config_") or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Некорректное имя config-файла")
    path = user_root(current_analysis_user(request)) / "configs" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Конфигурация не найдена")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.get("/reports/from-run/{run_id}")
async def create_report_from_run(request: Request, run_id: str, format: str = "html"):
    user = current_analysis_user(request)
    reports_dir = user_root(user) / "reports"
    try:
        run = user_run_manager(user).get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Эксперимент не найден") from exc
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if format.lower() == "json":
        path = reports_dir / f"report_{run_id}_{timestamp}.json"
        payload = {
            "report_type": "technical_analysis_report",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run": run,
            "technical_conclusion": "Результаты сформированы автоматически и требуют инженерной проверки параметров обработки.",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "path": str(path.relative_to(BASE_DIR)), "report": payload}
    html = _run_report_html(run)
    path = reports_dir / f"report_{run_id}_{timestamp}.html"
    path.write_text(html, encoding="utf-8")
    return StreamingResponse(
        io.BytesIO(html.encode("utf-8")),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


def _run_report_html(run: Dict[str, Any]) -> str:
    results = run.get("results") or {}
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Технический отчёт {run.get('run_id')}</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;max-width:980px;margin:32px auto;color:#172033}} pre{{background:#f5f7fb;padding:12px;border-radius:8px;overflow:auto}}</style></head>
<body>
<h1>Технический отчёт по анализу спектральных данных</h1>
<p><strong>run_id:</strong> {run.get('run_id')}</p>
<p><strong>Дата:</strong> {run.get('created_at')}</p>
<p><strong>Тип запуска:</strong> {run.get('run_type')}</p>
<p><strong>Исходный датасет:</strong> {run.get('dataset_name') or run.get('dataset_id')}</p>
<p><strong>Размерность:</strong> см. metadata запуска</p>
<p><strong>Использованные методы:</strong> {', '.join([str(v) for v in run.get('methods') or [] if v])}</p>
<p><strong>Краткий результат:</strong> {run.get('summary') or ''}</p>
<p><strong>Лучшая модель/метод:</strong> {run.get('best_method') or 'н/д'}</p>
<h2>Параметры и результаты</h2>
<pre>{json.dumps(results, ensure_ascii=False, indent=2)}</pre>
<h2>Технический вывод</h2>
<p>Отчёт содержит вычислительные результаты обработки спектральных данных. Интерпретация должна выполняться в контексте методики измерений и параметров обработки.</p>
</body></html>"""


@router.get("/models")
async def list_models(request: Request, model_type: Optional[str] = None):
    return user_analysis_service(current_analysis_user(request)).model_manager.list_models(model_type=model_type)


@router.post("/models/{model_type}/{model_id}/load")
async def load_model(request: Request, model_type: str, model_id: str):
    try:
        metadata = user_analysis_service(current_analysis_user(request)).model_manager.get_metadata(model_type=model_type, model_id=model_id)
        return {
            "status": "ok",
            "message": "loaded",
            "model": metadata,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/{model_id}/load")
async def load_model_by_id(request: Request, model_id: str, model_type: Optional[str] = None):
    user = current_analysis_user(request)
    resolved = resolve_user_model(user, model_id, model_type)
    return await load_model(request, resolved["model_type"], resolved["model_id"])


@router.get("/models/{model_id}/metadata")
async def download_model_metadata(request: Request, model_id: str, model_type: Optional[str] = None):
    user = current_analysis_user(request)
    resolved = resolve_user_model(user, model_id, model_type)
    try:
        payload, filename = user_analysis_service(user).model_manager.metadata_bytes(**resolved)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/models/{model_id}/download")
async def download_model_file(request: Request, model_id: str, model_type: Optional[str] = None):
    user = current_analysis_user(request)
    resolved = resolve_user_model(user, model_id, model_type)
    try:
        payload, filename = user_analysis_service(user).model_manager.model_bytes(**resolved)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/models/{model_id}/download-zip")
async def download_model_zip(request: Request, model_id: str, model_type: Optional[str] = None):
    user = current_analysis_user(request)
    resolved = resolve_user_model(user, model_id, model_type)
    try:
        payload = user_analysis_service(user).model_manager.export_zip(**resolved)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{model_id}.zip"'},
    )


@router.post("/models/upload")
async def upload_model(
    request: Request,
    model_file: UploadFile = File(...),
    meta_file: Optional[UploadFile] = File(None),
    model_type: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
):
    if os.getenv("ALLOW_MODEL_UPLOADS", "false").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Загрузка внешних joblib-моделей отключена. Включайте ALLOW_MODEL_UPLOADS=true только в доверенной среде.",
        )
    try:
        user = current_analysis_user(request)
        model_blob = await model_file.read()
        metadata = {}

        if meta_file is not None:
            raw_meta = await meta_file.read()
            metadata = json.loads(raw_meta.decode("utf-8"))

        resolved_model_type = (
            (model_type or "").strip().lower()
            or str(metadata.get("model_type", "")).strip().lower()
        )
        if not resolved_model_type:
            raise ValueError("Укажите model_type (или передайте его в meta.json)")

        if model_name:
            metadata["model_name"] = model_name.strip()
        metadata["user_id"] = user["user_id"]
        metadata["username"] = user["username"]

        saved = user_analysis_service(user).model_manager.import_uploaded(
            model_type=resolved_model_type,
            model_blob=model_blob,
            metadata=metadata,
        )
        return {"status": "ok", "message": "uploaded", "saved_model": saved}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Невалидный meta.json") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected model upload error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка загрузки модели") from exc


@router.delete("/models/{model_type}/{model_id}")
async def delete_model(request: Request, model_type: str, model_id: str):
    try:
        removed = user_analysis_service(current_analysis_user(request)).model_manager.delete(model_type=model_type, model_id=model_id)
        return {"status": "ok", "message": "deleted", "model_type": model_type, "model_id": model_id, **removed}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/models/{model_id}")
async def delete_model_by_id(request: Request, model_id: str, model_type: Optional[str] = None):
    user = current_analysis_user(request)
    resolved = resolve_user_model(user, model_id, model_type)
    return await delete_model(request, resolved["model_type"], resolved["model_id"])


@router.post("/infer")
async def infer_model(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form(...),
    model_id: str = Form(...),
):
    try:
        service = user_analysis_service(current_analysis_user(request))
        raw = await file.read()
        imported = import_dataset(
            [UploadedFileData(filename=file.filename or "spectrum.txt", content=raw)],
            {
                "layout": "txt_folder",
                "target_source": "none",
                "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
            },
        )
        dataset = imported.to_analysis_dataset(version="raw")
        return {"status": "ok", **service.infer(model_type=model_type, model_id=model_id, dataset=dataset)}
    except (SpectrumValidationError, DatasetImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка применения модели") from exc


@router.post("/model/infer")
async def model_infer(request: Request, file: UploadFile = File(...), model_type: str = Form(...), model_id: str = Form(...)):
    return await infer_model(request, file, model_type, model_id)


@router.post("/infer-batch")
async def infer_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model_type: str = Form(...),
    model_id: str = Form(...),
):
    try:
        service = user_analysis_service(current_analysis_user(request))
        if not files:
            raise HTTPException(status_code=400, detail="Не переданы файлы для инференса")

        uploaded: List[UploadedFileData] = []
        for file in files:
            raw = await file.read()
            uploaded.append(UploadedFileData(filename=file.filename or "unknown.txt", content=raw))
        imported = import_dataset(
            uploaded,
            {
                "layout": "txt_folder",
                "target_source": "none",
                "interpolation": {"enabled": True, "grid_mode": "first_axis", "step": "auto"},
            },
        )
        dataset = imported.to_analysis_dataset(version="raw")
        source_items = imported.source_items or [item.filename for item in uploaded]
        datasets = [
            {"filename": source_items[idx] if idx < len(source_items) else f"sample_{idx + 1}", "dataset": dataset_slice(dataset, idx)}
            for idx in range(dataset.sample_count)
        ]

        return {"status": "ok", **service.infer_many(model_type=model_type, model_id=model_id, datasets=datasets)}
    except (SpectrumValidationError, DatasetImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected batch inference error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка пакетного применения модели") from exc
