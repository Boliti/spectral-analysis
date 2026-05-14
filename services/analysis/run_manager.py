from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnalysisRunManager:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
            raise ValueError("Некорректный run_id")
        return value

    def new_run_id(self, prefix: str = "run") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{stamp}"
        idx = 1
        run_id = f"{base}_{idx:03d}"
        while (self.root_dir / run_id).exists():
            idx += 1
            run_id = f"{base}_{idx:03d}"
        return run_id

    def _run_dir(self, run_id: str) -> Path:
        return self.root_dir / self._safe_id(run_id)

    def save(self, run: Dict[str, Any]) -> Dict[str, Any]:
        run_id = self._safe_id(str(run.get("run_id") or self.new_run_id()))
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **run,
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "results.json").write_text(
            json.dumps(payload.get("results", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        plots = payload.get("plots")
        if plots is None:
            plots = payload.get("results", {}).get("plots") or []
        (run_dir / "plots.json").write_text(
            json.dumps({"plots": plots}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_comparison_csv(run_dir / "comparison_table.csv", payload)
        return payload

    def _write_comparison_csv(self, path: Path, run: Dict[str, Any]) -> None:
        rows = run.get("results", {}).get("method_results") or run.get("results", {}).get("comparison") or []
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        flat_rows: List[Dict[str, Any]] = []
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            flat = {k: v for k, v in row.items() if k not in {"metrics", "confusion_matrix", "params"}}
            flat.update(metrics)
            flat_rows.append(flat)
        keys = sorted({key for row in flat_rows for key in row.keys()})
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)
        path.write_text(buffer.getvalue(), encoding="utf-8")

    def list(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for meta_path in self.root_dir.glob("*/run_metadata.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            items.append(
                {
                    "run_id": payload.get("run_id"),
                    "run_type": payload.get("run_type"),
                    "created_at": payload.get("created_at"),
                    "dataset_id": payload.get("dataset_id"),
                    "dataset_name": payload.get("dataset_name"),
                    "target_name": payload.get("target_name"),
                    "target_type": payload.get("target_type"),
                    "methods": payload.get("methods"),
                    "best_method": payload.get("best_method"),
                    "status": payload.get("status"),
                    "summary": payload.get("summary"),
                }
            )
        items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return items

    def get(self, run_id: str) -> Dict[str, Any]:
        meta_path = self._run_dir(run_id) / "run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("Эксперимент не найден")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        plots_path = self._run_dir(run_id) / "plots.json"
        if plots_path.exists():
            try:
                payload["plots"] = json.loads(plots_path.read_text(encoding="utf-8")).get("plots", [])
            except json.JSONDecodeError:
                payload["plots"] = []
        else:
            payload.setdefault("plots", [])
        return payload

    def delete(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError("Эксперимент не найден")
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        run_dir.rmdir()
        return {"run_id": run_id, "deleted": True}

    def export_zip(self, run_id: str) -> bytes:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError("Эксперимент не найден")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in run_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir).as_posix())
        buffer.seek(0)
        return buffer.read()
