from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class State(str, Enum):
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Observation:
    timestamp_sec: float
    state: State
    confidence: float
    bbox: tuple[int, int, int, int] | None
    bed_overlap: float = 0.0
    speed_px_sec: float = 0.0
    keypoints: list[list[float]] | None = None
    track_id: int | None = None
    reason: str = "unspecified"
    bed_polygon: list[list[int]] | None = None
    bed_relation: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result


@dataclass
class TimelineSegment:
    start_sec: float
    end_sec: float
    state: State
    confidence: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "state": self.state.value,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Event:
    event: str
    start_time_sec: float
    confirmed_time_sec: float
    previous_state: State
    current_state: State
    confidence: float
    decision: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["previous_state"] = self.previous_state.value
        result["current_state"] = self.current_state.value
        return result
