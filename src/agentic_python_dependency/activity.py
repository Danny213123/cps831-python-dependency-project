from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any


@dataclass(slots=True)
class CaseActivityEvent:
    case_id: str
    attempt: int
    kind: str
    detail: str
    timestamp: str

    @classmethod
    def create(
        cls,
        case_id: str,
        *,
        attempt: int = 0,
        kind: str,
        detail: str,
        timestamp: str | None = None,
    ) -> "CaseActivityEvent":
        return cls(
            case_id=case_id,
            attempt=max(0, int(attempt or 0)),
            kind=kind.strip() or "activity",
            detail=detail.strip() or "activity update",
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def from_payload(cls, payload: object) -> "CaseActivityEvent | None":
        if not isinstance(payload, dict):
            return None
        case_id = str(payload.get("case_id", "") or "").strip()
        if not case_id:
            return None
        return cls.create(
            case_id,
            attempt=int(payload.get("attempt", 0) or 0),
            kind=str(payload.get("kind", "") or "activity"),
            detail=str(payload.get("detail", "") or "activity update"),
            timestamp=str(payload.get("timestamp", "") or datetime.now(timezone.utc).isoformat()),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class CaseActivityTracker:
    def __init__(
        self,
        current_payload: object | None = None,
        recent_payload: object | None = None,
        *,
        max_recent: int = 80,
    ):
        self._lock = RLock()
        self._current: dict[str, CaseActivityEvent] = {}
        self._recent: list[CaseActivityEvent] = []
        self._max_recent = max(1, max_recent)
        self._restore(current_payload=current_payload, recent_payload=recent_payload)

    def _restore(self, *, current_payload: object | None, recent_payload: object | None) -> None:
        if isinstance(current_payload, list):
            for item in current_payload:
                event = CaseActivityEvent.from_payload(item)
                if event is not None:
                    self._current[event.case_id] = event
        if isinstance(recent_payload, list):
            for item in recent_payload:
                event = CaseActivityEvent.from_payload(item)
                if event is not None:
                    self._recent.append(event)
        del self._recent[self._max_recent :]

    def emit(
        self,
        case_id: str,
        *,
        attempt: int = 0,
        kind: str,
        detail: str,
        timestamp: str | None = None,
    ) -> CaseActivityEvent:
        event = CaseActivityEvent.create(
            case_id,
            attempt=attempt,
            kind=kind,
            detail=detail,
            timestamp=timestamp,
        )
        with self._lock:
            self._current[event.case_id] = event
            self._recent.insert(0, event)
            del self._recent[self._max_recent :]
        return event

    def finish_case(self, case_id: str) -> None:
        with self._lock:
            self._current.pop(case_id, None)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            current = [event.to_payload() for event in sorted(self._current.values(), key=lambda item: item.case_id)]
            recent = [event.to_payload() for event in self._recent]
        return {
            "current_case_activity": current,
            "recent_case_activity": recent,
        }
