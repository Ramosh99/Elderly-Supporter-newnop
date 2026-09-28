from __future__ import annotations

import math
from dataclasses import replace

from .models import Event, Observation, State, TimelineSegment

IN_BED = {State.LYING_IN_BED, State.SITTING_ON_BED}
OUT_OF_BED = {State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED}


class TemporalStateTracker:
    """Offline temporal review with separate activity and bed-event confirmation.

    Each sample represents [timestamp, next timestamp). Unknown evidence is never
    turned into known time. finish() is repeatable and does not mutate history.
    """

    def __init__(self, hold_sec=1.5, alert_after_sec=300.0, posture_hold_sec=0.6,
                 event_hold_sec=1.5, context_gap_sec=2.0, sitting_monitor_sec=120.0,
                 min_confidence=0.35, max_sample_gap_sec=1.0):
        values = (hold_sec, alert_after_sec, posture_hold_sec, event_hold_sec,
                  context_gap_sec, sitting_monitor_sec, max_sample_gap_sec)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Temporal thresholds must be finite and positive")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")
        self.hold_sec = hold_sec
        self.alert_after_sec = alert_after_sec
        self.posture_hold_sec = min(posture_hold_sec, hold_sec)
        self.event_hold_sec = event_hold_sec
        self.context_gap_sec = context_gap_sec
        self.sitting_monitor_sec = sitting_monitor_sec
        self.min_confidence = min_confidence
        self.max_sample_gap_sec = max_sample_gap_sec
        self.observations = []

    def update(self, observation: Observation):
        t = observation.timestamp_sec
        if not math.isfinite(t) or t < 0 or (self.observations and t <= self.observations[-1].timestamp_sec):
            raise ValueError("Observation timestamps must be finite, nonnegative and strictly increasing")
        if not math.isfinite(observation.confidence) or not 0 <= observation.confidence <= 1:
            raise ValueError("Observation confidence must be between zero and one")
        if observation.confidence < self.min_confidence:
            observation = replace(observation, state=State.UNKNOWN, confidence=0.0)
        self.observations.append(observation)

    def _intervals(self, end):
        intervals = []
        if not self.observations:
            return [(Observation(0, State.UNKNOWN, 0, None), end)] if end else []
        if self.observations[0].timestamp_sec > 0:
            intervals.append((Observation(0, State.UNKNOWN, 0, None), self.observations[0].timestamp_sec))
        for i, observation in enumerate(self.observations):
            stop = self.observations[i+1].timestamp_sec if i+1 < len(self.observations) else end
            bounded = min(stop, observation.timestamp_sec + self.max_sample_gap_sec)
            if bounded > observation.timestamp_sec:
                intervals.append((observation, bounded))
            if stop > bounded:
                intervals.append((Observation(bounded, State.UNKNOWN, 0, None, reason="sampling_gap"), stop))
        return intervals

    def _timeline(self, intervals):
        away_ranges = []
        for o, stop in intervals:
            if o.state in OUT_OF_BED and o.bed_relation == "away":
                if away_ranges and away_ranges[-1][1] == o.timestamp_sec and away_ranges[-1][2] == o.track_id:
                    away_ranges[-1][1] = stop
                else:
                    away_ranges.append([o.timestamp_sec, stop, o.track_id])
        runs = []
        for o, stop in intervals:
            if runs and runs[-1][0] == o.state and runs[-1][4] == o.track_id:
                runs[-1][2] = stop
                runs[-1][3].append((o.confidence, stop-o.timestamp_sec))
            else:
                runs.append([o.state, o.timestamp_sec, stop, [(o.confidence, stop-o.timestamp_sec)], o.track_id])
        segments = []
        for i, (state, start, stop, confidence, identity) in enumerate(runs):
            threshold = self.posture_hold_sec if state in IN_BED else self.hold_sec
            if state != State.UNKNOWN and stop-start + 1e-9 < threshold:
                if (0 < i < len(runs)-1 and runs[i-1][0] == runs[i+1][0]
                        and runs[i-1][0] != State.UNKNOWN
                        and runs[i-1][2]-runs[i-1][1] >= (self.posture_hold_sec if runs[i-1][0] in IN_BED else self.hold_sec)
                        and runs[i+1][2]-runs[i+1][1] >= (self.posture_hold_sec if runs[i+1][0] in IN_BED else self.hold_sec)
                        and runs[i-1][4] == identity == runs[i+1][4]):
                    state = runs[i-1][0]
                else:
                    supported_away = state in OUT_OF_BED and any(
                        a <= start and stop <= b and b-a + 1e-9 >= self.hold_sec and target == identity
                        for a,b,target in away_ranges)
                    state = State.OUT_OF_BED if supported_away else State.UNKNOWN
            avg = sum(c*d for c,d in confidence) / (stop-start) if state != State.UNKNOWN else 0.0
            if segments and segments[-1].state == state:
                prev = segments[-1]
                avg = (prev.confidence*prev.duration_sec + avg*(stop-start))/(stop-prev.start_sec)
                prev.end_sec, prev.confidence = stop, avg
            else:
                segments.append(TimelineSegment(start, stop, state, avg))
        return segments

    def _events(self, intervals):
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
                if stop - unknown_start >= self.context_gap_sec:
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
            required = self.posture_hold_sec if baseline is None and location == "in" else self.event_hold_sec
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

    def finish(self, end_sec):
        if not math.isfinite(end_sec) or end_sec < 0 or (self.observations and end_sec < self.observations[-1].timestamp_sec):
            raise ValueError("Video end must be finite and at or after the last observation")
        intervals = self._intervals(end_sec)
        segments = self._timeline(intervals)
        events, reviews = self._events(intervals)
        durations = {s.value.lower(): 0.0 for s in State}
        for segment in segments:
            durations[segment.state.value.lower()] += segment.duration_sec
        longest_out = longest_sitting = running_out = running_sitting = 0.0
        for segment in segments:
            running_out = running_out + segment.duration_sec if segment.state in OUT_OF_BED else 0.0
            running_sitting = running_sitting + segment.duration_sec if segment.state == State.SITTING_ON_BED else 0.0
            longest_out = max(longest_out, running_out)
            longest_sitting = max(longest_sitting, running_sitting)
        longest_away = running_away = 0.0
        away_identity = None
        for o, stop in intervals:
            if o.track_id != away_identity:
                running_away = 0.0
            away_identity = o.track_id
            running_away = running_away + stop-o.timestamp_sec if o.state in OUT_OF_BED and o.bed_relation == "away" else 0.0
            longest_away = max(longest_away, running_away)
        reasons = []
        decision = "NORMAL"
        if durations["unknown"] > 0 or not intervals:
            reasons.append("uncertain_or_missing_observations")
        if longest_sitting >= self.sitting_monitor_sec:
            reasons.append("prolonged_sitting_on_bed")
        if events and events[-1].event == "bed_exit":
            reasons.append("bed_exit_without_confirmed_return")
        if reasons:
            decision = "MONITOR"
        if longest_away >= self.alert_after_sec:
            decision = "ALERT"
            reasons.append("prolonged_confirmed_absence_from_bed")
        return {
            "observation_duration_sec": round(end_sec, 3),
            "activity_duration_sec": {k: round(v, 3) for k,v in durations.items()},
            "bed_exit_count": sum(e.event == "bed_exit" for e in events),
            "bed_return_count": sum(e.event == "return_to_bed" for e in events),
            "total_in_bed_sec": round(sum(durations[s.value.lower()] for s in IN_BED), 3),
            "total_out_of_bed_sec": round(sum(durations[s.value.lower()] for s in OUT_OF_BED), 3),
            "total_unknown_sec": round(durations["unknown"], 3),
            "longest_out_of_bed_period_sec": round(longest_out, 3),
            "longest_confirmed_away_sec": round(longest_away, 3),
            "final_state": segments[-1].state.value if segments else State.UNKNOWN.value,
            "decision": decision, "decision_reasons": reasons,
            "timeline": [s.to_dict() for s in segments],
            "events": [e.to_dict() for e in events], "temporal_reviews": reviews,
        }
