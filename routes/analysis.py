from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from services.analysis.analysis_service import AnalysisService
from services.analysis.spectrum_loader import SpectrumValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "saved_models"

router = APIRouter(prefix="/analysis", tags=["analysis"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
analysis_service = AnalysisService(model_root=MODELS_DIR)


@router.get("")
async def analysis_page(request: Request):
    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "request": request,
            "user": request.session.get("user"),
            "is_authenticated": bool(request.session.get("user")),
        },
    )


@router.post("/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        dataset = analysis_service.parse_file(raw=raw, filename=file.filename or "")
        return {
            "filename": file.filename,
            "sample_count": dataset.sample_count,
            "feature_count": dataset.feature_count,
            "source_format": dataset.source_format,
            "has_axis": dataset.spectral_axis is not None,
        }
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_values: Optional[str] = Form(None),
    n_components: Optional[int] = Form(None),
    model_name: Optional[str] = Form(None),
    do_validation: bool = Form(True),
    test_size: float = Form(0.3),
    random_state: int = Form(42),
):
    try:
        raw = await file.read()
        dataset = analysis_service.parse_file(raw=raw, filename=file.filename or "")
        payload = analysis_service.train_and_save(
            model_type=model_type,
            dataset=dataset,
            raw_targets=target_values,
            n_components=n_components,
            model_name=model_name,
            do_validation=do_validation,
            test_size=test_size,
            random_state=random_state,
        )
        return payload
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models")
async def list_models(model_type: Optional[str] = None):
    return analysis_service.model_manager.list_models(model_type=model_type)


@router.post("/models/{model_type}/{model_id}/load")
async def load_model(model_type: str, model_id: str):
    try:
        metadata = analysis_service.model_manager.get_metadata(model_type=model_type, model_id=model_id)
        return {
            "status": "loaded",
            "model": metadata,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    meta_file: Optional[UploadFile] = File(None),
    model_type: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
):
    try:
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

        saved = analysis_service.model_manager.import_uploaded(
            model_type=resolved_model_type,
            model_blob=model_blob,
            metadata=metadata,
        )
        return {"status": "uploaded", "saved_model": saved}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Невалидный meta.json") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{model_type}/{model_id}")
async def delete_model(model_type: str, model_id: str):
    try:
        removed = analysis_service.model_manager.delete(model_type=model_type, model_id=model_id)
        return {"status": "deleted", "model_type": model_type, "model_id": model_id, **removed}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/infer")
async def infer_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    model_id: str = Form(...),
):
    try:
        raw = await file.read()
        dataset = analysis_service.parse_file(raw=raw, filename=file.filename or "")
        return analysis_service.infer(model_type=model_type, model_id=model_id, dataset=dataset)
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/infer-batch")
async def infer_batch(
    files: List[UploadFile] = File(...),
    model_type: str = Form(...),
    model_id: str = Form(...),
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="Не переданы файлы для инференса")

        datasets = []
        for file in files:
            raw = await file.read()
            dataset = analysis_service.parse_file(raw=raw, filename=file.filename or "")
            datasets.append({"filename": file.filename or "unknown", "dataset": dataset})

        return analysis_service.infer_many(model_type=model_type, model_id=model_id, datasets=datasets)
    except SpectrumValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
