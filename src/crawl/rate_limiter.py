"""
Rate Limiter for vnstock API

Gioi han thuc te cua vnstock:
  Guest     : 20 requests/phut  (dung 18 de co buffer)
  Community : 60 requests/phut
  Sponsor   : 180-600 requests/phut
"""

import time
import re
import logging
from collections import deque
from threading import Lock

log = logging.getLogger("stock_crawl")


class RateLimiter:
    """Rate limiter dam bao khong vuot qua gioi han request."""

    def __init__(self, max_per_minute: int = 18, max_per_hour: int = 1000):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_requests: deque = deque()
        self.hour_requests: deque = deque()
        self.lock = Lock()
        self.total_requests = 0
        self.total_wait_time = 0.0

    def _clean_old(self, now: float) -> None:
        cutoff_min = now - 60
        while self.minute_requests and self.minute_requests[0] < cutoff_min:
            self.minute_requests.popleft()
        cutoff_hr = now - 3600
        while self.hour_requests and self.hour_requests[0] < cutoff_hr:
            self.hour_requests.popleft()

    def _wait_needed(self, now: float) -> float:
        waits = []
        if len(self.minute_requests) >= self.max_per_minute:
            w = (self.minute_requests[0] + 60) - now
            if w > 0:
                waits.append(w)
        if len(self.hour_requests) >= self.max_per_hour:
            w = (self.hour_requests[0] + 3600) - now
            if w > 0:
                waits.append(w)
        return max(waits) if waits else 0

    def wait_if_needed(self) -> float:
        with self.lock:
            now = time.time()
            self._clean_old(now)
            w = self._wait_needed(now)
            if w > 0:
                w += 1.0
                log.info("Rate limit reached. Waiting %.1fs...", w)
                time.sleep(w)
                self.total_wait_time += w
                now = time.time()
            self.minute_requests.append(now)
            self.hour_requests.append(now)
            self.total_requests += 1
            return w

    def force_wait(self, seconds: float) -> None:
        log.info("Forced wait: %.1fs due to API rate limit", seconds)
        time.sleep(seconds)
        self.total_wait_time += seconds
        with self.lock:
            self.minute_requests.clear()


def extract_wait_time(error_message: str, default: int = 65) -> int:
    """Trich xuat thoi gian cho tu error message cua vnstock."""
    match = re.search(r"(\d+)\s*gi[aâ]y", str(error_message))
    if match:
        return int(match.group(1)) + 5
    return default


def is_rate_limit_error(error: Exception) -> bool:
    s = str(error).lower()
    return any(kw in s for kw in ["rate limit", "ratelimit", "too many requests", "429"])


def fetch_with_retry(func, rate_limiter: RateLimiter, retries: int = 5, *args, **kwargs):
    """Goi API voi rate limiting va retry logic.

    Handles:
    - vnstock goi sys.exit() khi rate limit -> catch SystemExit
    - HTTP 429 / rate limit errors -> extract wait time
    - Other errors -> exponential backoff
    """
    last_error = None
    for attempt in range(retries):
        try:
            rate_limiter.wait_if_needed()
            return func(*args, **kwargs)
        except SystemExit as e:
            last_error = e
            w = extract_wait_time(str(e))
            log.warning(
                "API rate limit (SystemExit). Waiting %ds... (attempt %d/%d)",
                w, attempt + 1, retries,
            )
            rate_limiter.force_wait(w)
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                w = extract_wait_time(str(e))
                log.warning(
                    "Rate limit error. Waiting %ds... (attempt %d/%d)",
                    w, attempt + 1, retries,
                )
                rate_limiter.force_wait(w)
            else:
                w = 2 ** attempt
                log.warning(
                    "Error: %s. Retry in %ds (attempt %d/%d)",
                    str(e)[:100], w, attempt + 1, retries,
                )
                time.sleep(w)
    log.error("Failed after %d retries. Last error: %s", retries, str(last_error)[:200])
    return None
