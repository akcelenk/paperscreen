"""A small polite HTTP client: per-host rate limiting, retries with backoff, injectable for tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from urllib.parse import urlparse

import requests

_lock = threading.Lock()
_last_call: dict[str, float] = {}  # host -> monotonic time of the last request (process-wide)


class HttpError(RuntimeError):
    pass


class HttpClient:
    """GET with a minimum interval per host (shared by every client in the process).

    arXiv's terms ask for at most one request per three seconds and one connection, summed over
    all your machines. Keeping a process-wide limiter and a single sequential fetch loop is the
    easiest way to respect that.
    """

    def __init__(
        self,
        min_interval: float = 0.0,
        user_agent: str = "paperscreen/0.1",
        timeout: float = 30.0,
        retries: int = 4,
        max_backoff: float = 60.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.max_backoff = max_backoff
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)
        self._sleep = sleep
        self._clock = clock

    def _wait_turn(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        with _lock:
            now = self._clock()
            last = _last_call.get(host)
            if last is not None and now - last < self.min_interval:
                self._sleep(self.min_interval - (now - last))
            _last_call[host] = self._clock()

    def get(self, url: str, params: dict | None = None, headers: dict | None = None):
        host = urlparse(url).netloc
        last_err: str = ""
        for attempt in range(self.retries + 1):
            self._wait_turn(host)
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:  # network error: retry
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if r.status_code == 200:
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    retry_after = r.headers.get("Retry-After", "")
                    if retry_after.isdigit() and int(retry_after) > self.max_backoff:
                        # Long quotas (e.g. a daily budget) will not clear by waiting here.
                        raise HttpError(f"{url}: HTTP {r.status_code}, Retry-After={retry_after}s")
                    delay = float(retry_after) if retry_after.isdigit() else 2.0**attempt
                    if attempt < self.retries:
                        self._sleep(min(delay, self.max_backoff))
                    continue
                raise HttpError(f"{url}: HTTP {r.status_code}: {r.text[:200]}")
            if attempt < self.retries:
                self._sleep(min(2.0**attempt, self.max_backoff))
        raise HttpError(f"{url}: giving up after {self.retries + 1} attempts ({last_err})")
