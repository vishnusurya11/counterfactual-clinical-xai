"""Simple token-bucket rate limiter."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = max(1, requests_per_minute)
        self.min_interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()
