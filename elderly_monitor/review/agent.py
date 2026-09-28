"""Select bounded context windows and request additional visual evidence."""
from bisect import bisect_left

from ..models import State


def box_iou(a, b):
    x = max(a[0], b[0]); y = max(a[1], b[1])
    right = min(a[0]+a[2], b[0]+b[2]); bottom = min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x)*max(0, bottom-y)
    return intersection / max(1, a[2]*a[3]+b[2]*b[3]-intersection)


def identity_anchor(observations, timestamp, max_gap=1.0):
    """Require two nearby detections of the same original target."""
    index = bisect_left([o.timestamp_sec for o in observations], timestamp)
    if index == 0 or index == len(observations):
        return None
    left, right = observations[index-1:index+1]
    if (left.bbox is None or right.bbox is None or left.track_id is None
            or left.track_id != right.track_id or right.timestamp_sec-left.timestamp_sec > max_gap):
        return None
    fraction = (timestamp-left.timestamp_sec)/(right.timestamp_sec-left.timestamp_sec)
    box = tuple(a+(b-a)*fraction for a,b in zip(left.bbox, right.bbox))
    return box, left.track_id


def select_identity(candidates, anchor):
    if anchor is None:
        return None
    matches = [c for c in candidates if box_iou(c['bbox'], anchor[0]) >= 0.3]
    return matches[0] if len(matches) == 1 else None


class ReviewAgent:
    def __init__(self, config):
        self.config = config

    def plan(self, observations, duration):
        requests = []
        for i, o in enumerate(observations):
            reason = None
            if o.state == State.UNKNOWN or o.confidence < self.config['min_state_confidence']:
                reason = 'uncertain_observation'
            elif i and o.bed_relation != observations[i-1].bed_relation:
                reason = 'possible_bed_transition'
            elif i and o.state != observations[i-1].state:
                reason = 'activity_transition'
            if reason:
                requests.append((o.timestamp_sec, reason))
        windows = []
        context = self.config['review_context_sec']
        for timestamp, reason in requests:
            if any(w['start_sec'] <= timestamp < w['end_sec'] for w in windows):
                continue
            if len(windows) >= self.config['review_max_windows']:
                break
            start = max(0.0, timestamp-context)
            if windows:
                start = max(start, windows[-1]['end_sec'])
            end = min(duration, timestamp+context)
            if end > start:
                windows.append(dict(start_sec=start, end_sec=end, trigger_sec=timestamp,
                                    reason=reason, action='inspect_additional_frames'))
        return windows

    def run(self, observations, duration, gather):
        merged = {round(o.timestamp_sec, 9): o for o in observations}
        log = []
        remaining = self.config['review_max_frames']
        for window in self.plan(observations, duration):
            if remaining <= 0:
                break
            extra, attempted, rejected = gather(window, remaining)
            remaining -= attempted
            added = 0
            for o in extra:
                key = round(o.timestamp_sec, 9)
                if key not in merged:
                    merged[key] = o
                    added += 1
            log.append({**window, 'attempted_frames': attempted, 'added_frames': added,
                        'identity_rejections': rejected,
                        'known_extra_frames': sum(o.state != State.UNKNOWN for o in extra),
                        'outcome': 'additional_evidence_gathered' if added else 'insufficient_evidence'})
        return sorted(merged.values(), key=lambda o:o.timestamp_sec), {
            'enabled': True, 'strategy': 'bounded_context_resampling',
            'attempts_per_window': 1, 'frame_budget': self.config['review_max_frames'],
            'attempted_frames': self.config['review_max_frames']-remaining,
            'windows': log, 'budget_exhausted': remaining == 0,
        }
