"""Disk-based API response cache.

Key = hash(model_name + prompt_text + temperature + run_index + extras).
Value = full API response dict.

CRITICAL: This saves money. A full re-run without cache costs ~$20+.
With cache, re-runs cost $0 because cached responses are returned verbatim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import diskcache


class APICache:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir))

    @staticmethod
    def make_key(
        model: str,
        prompt: str,
        temperature: float,
        run_idx: int = 0,
        **extras: Any,
    ) -> str:
        extras_str = json.dumps(extras, sort_keys=True, default=str) if extras else ""
        blob = f"{model}|{prompt}|{temperature}|{run_idx}|{extras_str}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self.cache.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.cache.set(key, value)

    def get_or_call(
        self,
        model: str,
        prompt: str,
        temperature: float,
        run_idx: int,
        call_fn: Callable[[], dict[str, Any]],
        **extras: Any,
    ) -> tuple[dict[str, Any], bool]:
        """Return (response, was_cached)."""
        key = self.make_key(model, prompt, temperature, run_idx, **extras)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        response = call_fn()
        self.cache.set(key, response)
        return response, False

    def close(self) -> None:
        self.cache.close()
