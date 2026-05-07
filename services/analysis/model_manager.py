from __future__ import annotations

import json
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib


class ModelManager:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_type: str) -> Path:
        target = self.root_dir / model_type.lower()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _next_model_id(self, model_type: str) -> str:
        model_dir = self._model_dir(model_type)
        pattern = re.compile(r"model_(\d{3})_meta\.json$")
        max_id = 0
        for path in model_dir.glob("model_*_meta.json"):
            match = pattern.search(path.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"model_{max_id + 1:03d}"

    def _paths(self, model_type: str, model_id: str) -> Tuple[Path, Path]:
        model_dir = self._model_dir(model_type)
        return model_dir / f"{model_id}.joblib", model_dir / f"{model_id}_meta.json"

    def save(self, model_type: str, model_obj: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        model_id = self._next_model_id(model_type)
        model_path, meta_path = self._paths(model_type, model_id)

        payload = {
            "model_id": model_id,
            "model_type": model_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }

        joblib.dump(model_obj, model_path)
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def import_uploaded(
        self,
        model_type: str,
        model_blob: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed_meta = dict(metadata or {})
        loaded_obj = joblib.load(BytesIO(model_blob))
        if not hasattr(loaded_obj, "inference_result"):
            raise ValueError("Загруженный объект не поддерживает inference_result и не является совместимой моделью")
        return self.save(model_type=model_type, model_obj=loaded_obj, metadata=parsed_meta)

    def load(self, model_type: str, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        model_path, meta_path = self._paths(model_type, model_id)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Модель не найдена")

        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return model, meta

    def get_metadata(self, model_type: str, model_id: str) -> Dict[str, Any]:
        _, meta_path = self._paths(model_type, model_id)
        if not meta_path.exists():
            raise FileNotFoundError("Метаданные модели не найдены")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        types = [model_type.lower()] if model_type else [p.name for p in self.root_dir.iterdir() if p.is_dir()]
        result: List[Dict[str, Any]] = []

        for m_type in types:
            model_dir = self.root_dir / m_type
            if not model_dir.exists() or not model_dir.is_dir():
                continue
            for meta_file in sorted(model_dir.glob("model_*_meta.json")):
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    result.append(meta)
                except json.JSONDecodeError:
                    continue

        result.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return result

    def delete(self, model_type: str, model_id: str) -> Dict[str, Any]:
        model_path, meta_path = self._paths(model_type, model_id)
        removed = {"model_removed": False, "meta_removed": False}
        if model_path.exists():
            model_path.unlink()
            removed["model_removed"] = True
        if meta_path.exists():
            meta_path.unlink()
            removed["meta_removed"] = True
        if not removed["model_removed"] and not removed["meta_removed"]:
            raise FileNotFoundError("Модель не найдена")
        return removed
