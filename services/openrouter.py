import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_openrouter_client() -> Any:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if OpenAI is None:
        raise RuntimeError("AI functions are unavailable because the openai package is not installed; core analysis still works.")
    if not api_key:
        raise RuntimeError("AI functions are unavailable because OPENROUTER_API_KEY is not configured; core analysis still works.")

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
