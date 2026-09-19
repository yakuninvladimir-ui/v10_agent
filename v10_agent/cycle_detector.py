"""Bounded recovery for exact visible action cycles.

Ported and enhanced from Flash-Next v11.1 loop recovery for ARC-AGI-3 V10.0 runtime.
Detects repeated (before, action, after) closed state transitions across periods 1..8,
allowing early candidate severing and replanning before burning the level action budget.
"""

from __future__ import annotations

import collections
import hashlib
import math
from typing import Any


def _hash_grid(grid: Any) -> str:
    """Compute a fast deterministic hash of a 2D grid structure."""
    if grid is None:
        return "none"
    if isinstance(grid, str):
        return grid
    try:
        # Fast row string encoding
        flat = ";".join("".join(str(c) for c in row) for row in grid)
        return hashlib.md5(flat.encode("ascii")).hexdigest()[:16]
    except Exception:
        return str(hash(str(grid)))


class VisibleCycle:
    """Detect repeated (before, action, after) closed transition blocks."""

    def __init__(
        self,
        min_actions: int = 24,
        max_period: int = 8,
        min_cycles: int = 4,
    ) -> None:
        if not (8 <= min_actions <= 128 and 1 <= max_period <= 16 and 2 <= min_cycles <= 16):
            raise ValueError(f"Invalid VisibleCycle thresholds: min_actions={min_actions}, max_period={max_period}, min_cycles={min_cycles}")
        self.min_actions = min_actions
        self.max_period = max_period
        self.min_cycles = min_cycles
        max_len = max(min_actions + max_period, max_period * min_cycles)
        self.trace: collections.deque[tuple[str, str, str]] = collections.deque(maxlen=max_len)

    def clear(self) -> None:
        """Clear the observation trace."""
        self.trace.clear()

    def observe(self, before: Any, action: str, after: Any) -> dict[str, Any] | None:
        """Observe transition (before, action, after).

        Returns:
            Dictionary with cycle details if a closed repeating pattern is detected,
            or None if no cycle is detected.
        """
        b_hash = _hash_grid(before) if not isinstance(before, str) else before
        a_hash = _hash_grid(after) if not isinstance(after, str) else after
        act_str = str(action).upper()

        if self.trace and self.trace[-1][2] != b_hash:
            # Non-contiguous transition (board reset, level jump, or state discontinuity)
            self.trace.clear()

        self.trace.append((b_hash, act_str, a_hash))
        rows = list(self.trace)

        for period in range(1, self.max_period + 1):
            repetitions = max(self.min_cycles, math.ceil(self.min_actions / period))
            length = period * repetitions
            if len(rows) < length:
                continue
            block = rows[-period:]
            # Must be a closed orbit: starting state of period equals ending state of period
            if block[0][0] != block[-1][2]:
                continue
            if rows[-length:] == block * repetitions:
                return {
                    "period": period,
                    "repetitions": repetitions,
                    "observed_actions": length,
                    "action_pattern": [r[1] for r in block],
                }
        return None
