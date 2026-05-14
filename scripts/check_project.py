from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "app.py",
    "data_processing.py",
    "requirements.txt",
    "routes/analysis.py",
    "services/openrouter.py",
    "services/analysis/spectrum_loader.py",
    "services/analysis/dataset_importer.py",
    "services/analysis/analysis_service.py",
    "services/analysis/model_manager.py",
    "services/analysis/models.py",
    "templates/index.html",
    "templates/analysis.html",
    "static/app.js",
    "static/analysis.js",
]

LOCAL_ONLY_PATHS = [
    "venv",
    ".venv",
    "__pycache__",
    "users.db",
    "uploads",
    "saved_models",
    "DeepSeekAPI.env",
]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    print("Project root:", ROOT)
    print("Required files:", "ok" if not missing else "missing")
    for path in missing:
        print("  missing:", path)

    print("\nLocal-only files/directories present:")
    for path in LOCAL_ONLY_PATHS:
        if (ROOT / path).exists():
            print("  present, keep out of final archive/git:", path)

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
