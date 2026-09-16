"""Structured audit logging of the complete decision and verification chain."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AuditRecord:
    """A single auditable event in the decision pipeline."""
    timestamp: float
    event_type: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": dict(self.data),
        }


class StructuredAuditLogger:
    """Bounded, thread-safe in-memory structured audit logger."""

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.records: deque[AuditRecord] = deque(maxlen=max_records)

    def log(self, event_type: str, **kwargs: Any) -> None:
        """Record an audit event."""
        rec = AuditRecord(
            timestamp=time.time(),
            event_type=event_type,
            data=dict(kwargs),
        )
        self.records.append(rec)

    def get_records(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def dump_json_lines(self) -> str:
        return "\n".join(json.dumps(r.to_dict(), default=str) for r in self.records)
