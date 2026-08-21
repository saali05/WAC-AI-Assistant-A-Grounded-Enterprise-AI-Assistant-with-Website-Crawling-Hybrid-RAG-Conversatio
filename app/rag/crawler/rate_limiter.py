import asyncio
import time
from typing import Optional
from app.core.config import settings


class RateLimiter:
    """Asynchronous concurrency limiter and crawl-delay request throttler."""

    def __init__(self, concurrency: Optional[int] = None, crawl_delay: Optional[float] = None) -> None:
        self.concurrency = concurrency or settings.RAG_CONCURRENCY
        self.crawl_delay = crawl_delay if crawl_delay is not None else settings.RAG_CRAWL_DELAY
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire() -> None:
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.crawl_delay:
                await asyncio.sleep(self.crawl_delay - elapsed)
            self._last_request_time = time.monotonic()

    async def __aenter__(self):
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.crawl_delay:
                await asyncio.sleep(self.crawl_delay - elapsed)
            self._last_request_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()
