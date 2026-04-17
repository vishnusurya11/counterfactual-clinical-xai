"""Unified LLM client built on LiteLLM.

Handles OpenRouter (models 1-4) and LM Studio (models 5-6) through a single
interface. Integrates the API cache and rate limiter.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

# Silence Pydantic serializer UserWarnings emitted by LiteLLM internals.
# The warnings clutter stdout but the text content is unaffected.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.api_cache import APICache
from src.utils.rate_limiter import RateLimiter

# Keep LiteLLM quiet; we do our own logging.
litellm.suppress_debug_info = True
litellm.drop_params = True   # silently drop unsupported params per-provider


@dataclass
class LLMResponse:
    text: str
    thinking: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


# Extract <think>...</think> that some open-weight reasoning models embed.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _split_thinking(text: str) -> tuple[str, str]:
    matches = _THINK_PATTERN.findall(text or "")
    if not matches:
        return text or "", ""
    thinking = "\n\n".join(m.strip() for m in matches)
    cleaned = _THINK_PATTERN.sub("", text or "").strip()
    return cleaned, thinking


class LLMClient:
    """Thin wrapper around LiteLLM with disk cache and rate limiting."""

    def __init__(
        self,
        model_cfg: dict[str, Any],
        cache: APICache,
        rate_limiter: RateLimiter | None = None,
    ):
        self.cfg = model_cfg
        self.name = model_cfg["name"]
        self.litellm_model = model_cfg["litellm_model"]
        self.provider = model_cfg["provider"]
        self.default_temperature = float(model_cfg.get("temperature", 0.7))
        self.default_max_tokens = int(model_cfg.get("max_tokens", 1024))
        self.capture_thinking = bool(model_cfg.get("capture_thinking", False))
        self.extra_body = dict(model_cfg.get("extra_body") or {})
        self.api_base = model_cfg.get("api_base")
        self._cache = cache
        self._rate = rate_limiter

        # Resolve API key from env
        api_key_env = model_cfg.get("api_key_env")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if self.provider == "lmstudio" and not self.api_key:
            self.api_key = "lm-studio"  # LM Studio ignores it

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_idx: int = 0,
        system: str | None = None,
    ) -> LLMResponse:
        temp = self.default_temperature if temperature is None else float(temperature)
        mt = self.default_max_tokens if max_tokens is None else int(max_tokens)

        cache_key = self._cache.make_key(
            model=self.litellm_model,
            prompt=(system or "") + "\n---\n" + prompt,
            temperature=temp,
            run_idx=run_idx,
            max_tokens=mt,
            extra_body=self.extra_body,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return LLMResponse(
                text=cached["text"],
                thinking=cached.get("thinking", ""),
                model=self.name,
                provider=self.provider,
                input_tokens=cached.get("input_tokens", 0),
                output_tokens=cached.get("output_tokens", 0),
                total_tokens=cached.get("total_tokens", 0),
                latency_ms=cached.get("latency_ms", 0),
                cached=True,
                raw=cached.get("raw", {}),
            )

        if self._rate is not None:
            self._rate.wait()

        response = self._call_with_retry(prompt=prompt, temperature=temp, max_tokens=mt, system=system)

        # Persist to cache
        self._cache.set(
            cache_key,
            {
                "text": response.text,
                "thinking": response.thinking,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
                "raw": response.raw,
            },
        )
        return response

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=3, min=5, max=120),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_with_retry(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str | None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.litellm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 120,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        start = time.monotonic()
        resp = litellm.completion(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        choice = resp.choices[0]
        message = choice.message
        content = getattr(message, "content", "") or ""

        # Reasoning content comes through in different places depending on provider
        reasoning_content = getattr(message, "reasoning_content", "") or ""
        if not reasoning_content:
            # OpenRouter sometimes returns `reasoning` on the message
            reasoning_content = getattr(message, "reasoning", "") or ""

        # Some open-weight models (R1 distills, QwQ) embed <think>...</think> in content
        clean_text, inline_thinking = _split_thinking(content)
        thinking = reasoning_content or inline_thinking

        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0) if usage else 0

        raw = {}
        try:
            raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        except Exception:
            raw = {}

        return LLMResponse(
            text=clean_text,
            thinking=thinking,
            model=self.name,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cached=False,
            raw=raw,
        )


def build_clients(
    cfg: dict[str, Any],
    cache: APICache,
    rate_limiter: RateLimiter | None = None,
    only: list[str] | None = None,
) -> dict[str, LLMClient]:
    """Build one LLMClient per model in config. `only` restricts by name."""
    clients: dict[str, LLMClient] = {}
    for m in cfg["models"]:
        if only and m["name"] not in only:
            continue
        clients[m["name"]] = LLMClient(m, cache=cache, rate_limiter=rate_limiter)
    return clients
