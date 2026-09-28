"""Confirm bed events using spatial evidence over time."""
from ..models import Event, State
from .states import IN_BED


def detect_events(intervals, posture_hold_sec, event_hold_sec, context_gap_sec):
    events, reviews = [], []
    baseline = None
    baseline_state = State.UNKNOWN
    candidate = None
    candidate_start = 0.0
    confidences = []
    unknown_start = None
    identity = None
    for o, stop in intervals:
        if o.track_id is not None and identity is not None and o.track_id != identity:
            baseline, candidate = None, None
            reviews.append({"time_sec": o.timestamp_sec, "reason": "target_identity_changed"})
        if o.track_id is not None:
            identity = o.track_id
        if o.state == State.UNKNOWN:
            candidate = None
            unknown_start = o.timestamp_sec if unknown_start is None else unknown_start
            if stop - unknown_start >= context_gap_sec:
                baseline = None
            continue
        unknown_start = None
        location = "in" if o.state in IN_BED else "away" if o.bed_relation == "away" else None
        if location is None:
            if candidate is not None:
                reviews.append({"time_sec": o.timestamp_sec, "reason": "insufficient_spatial_evidence"})
            candidate = None
            continue
        if location == baseline:
            baseline_state = o.state
            candidate = None
            continue
        if candidate != location:
            candidate, candidate_start, confidences = location, o.timestamp_sec, []
        confidences.append(o.confidence)
        required = posture_hold_sec if baseline is None and location == "in" else event_hold_sec
        if stop-candidate_start + 1e-9 < required:
            continue
        confirmed = candidate_start + required
        if baseline is not None:
            event_name = "bed_exit" if location == "away" else "return_to_bed"
            events.append(Event(event_name, candidate_start, confirmed, baseline_state,
                                o.state, min(confidences), "MONITOR" if location == "away" else "NORMAL"))
            reviews.append({"time_sec": confirmed, "reason": "sustained_" + location,
                            "evidence_start_sec": candidate_start, "event": event_name})
        baseline, baseline_state, candidate = location, o.state, None
    return events, reviews

