"""Off-path enrichment with a deadline that is enforced, not promised.

Phase 5's hard rule: if the LLM has not answered by the time the entry decision
is made, the decision proceeds without it. That rule is worthless as a
convention — someone will await the coroutine "just this once" — so the only way
to call this layer from anywhere near an entry path is through a function that
cannot block past its deadline.

Gate D is the test: remove the enrichment entirely and entry timing must be
byte-identical. If it is not, the wiring is wrong.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

R = TypeVar("R")


async def with_deadline(
    factory: Callable[[], Awaitable[R]],
    deadline_s: float,
    default: Optional[R] = None,
) -> Optional[R]:
    """Await ``factory()`` for at most ``deadline_s``; return ``default`` otherwise.

    Swallows every exception by design. An advisory feature that can raise into
    the caller is an advisory feature that can stop a trade.
    """
    try:
        return await asyncio.wait_for(factory(), timeout=deadline_s)
    except asyncio.TimeoutError:
        logger.debug("advisory enrichment missed its %.2fs deadline", deadline_s)
        return default
    except Exception:  # noqa: BLE001 - deliberate: advisory must never propagate
        logger.warning("advisory enrichment failed", exc_info=True)
        return default


class AdvisoryEnricher(Generic[R]):
    """Fire-and-forget enrichment that attaches to a record if it lands in time.

    Start the work as early as you like; the entry decision reads
    :meth:`peek`, which never awaits anything. Whatever has not arrived by then
    simply is not there, and the caller's code path is identical either way.
    """

    def __init__(self, deadline_s: float = 5.0) -> None:
        self.deadline_s = deadline_s
        self._tasks: dict[str, asyncio.Task[Optional[R]]] = {}
        self._results: dict[str, R] = {}

    def start(self, key: str, factory: Callable[[], Awaitable[R]]) -> None:
        """Kick off enrichment for ``key``. Returns immediately."""
        if key in self._tasks:
            return

        async def runner() -> Optional[R]:
            value = await with_deadline(factory, self.deadline_s)
            if value is not None:
                self._results[key] = value
            return value

        self._tasks[key] = asyncio.ensure_future(runner())

    def peek(self, key: str) -> Optional[R]:
        """Whatever has arrived so far. Never blocks, never raises."""
        return self._results.get(key)

    async def collect(self, key: str) -> Optional[R]:
        """Wait out the remaining deadline. **Never call this on the entry path.**

        This exists for post-hoc analysis and backfill jobs, where the answer is
        worth waiting for because nothing is racing.
        """
        task = self._tasks.get(key)
        if task is None:
            return None
        try:
            return await task
        except Exception:  # noqa: BLE001
            logger.warning("advisory collect failed for %s", key, exc_info=True)
            return None

    async def drain(self) -> None:
        """Let outstanding work finish (or fail) at shutdown."""
        pending = [t for t in self._tasks.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()


__all__ = ["AdvisoryEnricher", "with_deadline"]
