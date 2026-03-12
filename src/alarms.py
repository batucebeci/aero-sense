from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

ALARM_TRIGGER_RISKS = {"High", "Critical"}
ALARM_BUFFER_SIZE = 200


@dataclass
class Alarm:
    timestamp: str
    system_id: str | None
    fault: str
    risk: str
    confidence: float | None
    recommendation: str
    source: str = "prediction"

    def as_dict(self) -> dict:
        return asdict(self)


class AlarmLog:
    def __init__(self, capacity: int = ALARM_BUFFER_SIZE) -> None:
        self._buffer: deque[Alarm] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._webhook: Callable[[Alarm], None] | None = None

    def register_webhook(self, callback: Callable[[Alarm], None] | None) -> None:
        self._webhook = callback

    def add(self, alarm: Alarm) -> None:
        with self._lock:
            self._buffer.appendleft(alarm)
        if self._webhook is not None:
            try:
                self._webhook(alarm)
            except Exception:
                pass

    def add_from_prediction(
        self,
        fault: str,
        risk: str,
        confidence: float | None,
        recommendation: str,
        system_id: str | None = None,
        source: str = "prediction",
    ) -> Alarm | None:
        if risk not in ALARM_TRIGGER_RISKS:
            return None
        alarm = Alarm(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            system_id=system_id,
            fault=fault,
            risk=risk,
            confidence=confidence,
            recommendation=recommendation,
            source=source,
        )
        self.add(alarm)
        return alarm

    def snapshot(self, limit: int | None = None) -> list[Alarm]:
        with self._lock:
            items = list(self._buffer)
        return items[:limit] if limit else items

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


_ALARM_LOG = AlarmLog()


def get_alarm_log() -> AlarmLog:
    return _ALARM_LOG
