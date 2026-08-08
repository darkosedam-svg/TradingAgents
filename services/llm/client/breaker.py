"""Circuit breaker guarding the local endpoint.

The failure this exists for is not a crash — it is the GPU being busy, the
container being restarted after a driver upgrade, or a cold start landing in the
middle of a decision. In all three the right answer is the same: stop paying the
timeout, route to the hosted API, and probe again later.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class BreakerState(str, Enum):
    CLOSED = "closed"  # local endpoint in use
    OPEN = "open"  # tripped, everything goes hosted
    HALF_OPEN = "half_open"  # cooldown expired, next call is a probe


class CircuitBreaker:
    """Trips after ``threshold`` consecutive failures; probes after ``cooldown_s``.

    A single success in ``HALF_OPEN`` closes it. A single failure in
    ``HALF_OPEN`` re-opens it and restarts the cooldown, so a flapping endpoint
    is probed at a fixed low rate rather than hammered.
    """

    def __init__(
        self,
        threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = BreakerState.CLOSED

    @property
    def state(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.cooldown_s
        ):
            self._state = BreakerState.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def allows_local(self) -> bool:
        """True when the caller should try the local endpoint."""
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            # The probe failed. Straight back to open, cooldown restarts.
            self._failures += 1
            self._trip()
            return

        self._failures += 1
        if self._failures >= self.threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()

    def reset(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
