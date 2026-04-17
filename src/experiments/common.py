"""Shared helpers for experiment runners."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm_client import LLMClient, LLMResponse


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path | str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wrap_response(resp: LLMResponse) -> dict[str, Any]:
    return {
        "text": resp.text,
        "thinking": resp.thinking,
        "model": resp.model,
        "provider": resp.provider,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "total_tokens": resp.total_tokens,
        "latency_ms": resp.latency_ms,
        "cached": resp.cached,
    }


def call_and_wrap(client: LLMClient, prompt: str, *, run_idx: int = 0, **kwargs: Any) -> dict[str, Any]:
    resp = client.generate(prompt, run_idx=run_idx, **kwargs)
    return wrap_response(resp)
