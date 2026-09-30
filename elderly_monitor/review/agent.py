"""Select bounded context windows and request additional visual evidence.

The agent uses an iterative reasoning loop: findings from each window can
trigger backward or forward follow-up windows before committing a conclusion.
"""
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
            elif (o.state in {State.STANDING, State.WALKING} and o.bed_relation == 'inside'
                  and o.support_evidence == 'uncalibrated'):
                reason = 'ambiguous_upright_over_bed'
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

        # Second pass: tile any remaining UNKNOWN stretches not yet covered.
        covered = lambda t: any(w['start_sec'] <= t < w['end_sec'] for w in windows)
        for o in observations:
            if o.state != State.UNKNOWN:
                continue
            if covered(o.timestamp_sec):
                continue
            if len(windows) >= self.config['review_max_windows']:
                break
            start = max(0.0, o.timestamp_sec - context)
            if windows:
                start = max(start, windows[-1]['end_sec'])
            end = min(duration, o.timestamp_sec + context)
            if end > start:
                windows.append(dict(start_sec=start, end_sec=end, trigger_sec=o.timestamp_sec,
                                    reason='uncovered_unknown_block', action='inspect_additional_frames'))

        return sorted(windows, key=lambda w: w['start_sec'])

    def _assess_finding(self, gathered_obs):
        """Summarize what the dense frames in this window revealed.

        Returns a finding dict that drives follow-up window decisions.
        """
        if not gathered_obs:
            return {'type': 'no_evidence', 'states_found': [], 'reasoning': 'No frames returned for this window.'}
        known = [o for o in gathered_obs if o.state != State.UNKNOWN]
        states = list({o.state.value for o in known})
        upright_near_bed = any(
            o.state in {State.STANDING, State.WALKING} and o.bed_relation in {'inside', 'near'}
            for o in gathered_obs
        )
        in_bed = any(o.state in {State.LYING_IN_BED, State.SITTING_ON_BED} for o in known)
        uncertainty_remains = any(o.state == State.UNKNOWN for o in gathered_obs) and not known

        if upright_near_bed and not in_bed:
            ftype = 'possible_transition'
            reasoning = ('Person is upright near the bed. Cannot determine direction of transition '
                         '(exit vs return) without prior and subsequent context.')
        elif len(states) > 1:
            ftype = 'transition_detected'
            reasoning = f'State change observed within window: {" -> ".join(states)}. Need surrounding context to confirm.'
        elif uncertainty_remains:
            ftype = 'uncertainty_remains'
            reasoning = 'Frames still ambiguous after dense sampling. Adjacent context may help.'
        else:
            ftype = 'context_gathered'
            reasoning = f'Window clarified: {states[0] if states else "no clear state"}.'

        return {
            'type': ftype,
            'states_found': states,
            'upright_near_bed': upright_near_bed,
            'in_bed_found': in_bed,
            'reasoning': reasoning,
        }

    def _follow_up_windows(self, window, finding, duration, visited):
        """Decide whether backward or forward context windows are needed.

        This implements the iterative reasoning step:
          possible_transition -> look backward (what was the prior state?)
                              -> look forward  (where does the person go next?)
          transition_detected -> look backward to anchor the start of the transition
          uncertainty_remains -> look at the adjacent forward window
        """
        follow_ups = []
        context = self.config['review_context_sec']

        if finding['type'] in ('possible_transition', 'transition_detected'):
            # Look backward: was the person in bed before this window?
            back_end = window['start_sec']
            back_start = max(0.0, back_end - context * 2)
            if back_end - back_start > 0.3 and round(back_start, 3) not in visited:
                follow_ups.append(dict(
                    start_sec=back_start, end_sec=back_end,
                    trigger_sec=(back_start + back_end) / 2,
                    reason='backward_context_check',
                    action='analyze_previous_segment',
                    parent_finding=finding['type'],
                ))

            # Look forward: does the person stand up and walk away, or return to bed?
            fwd_start = window['end_sec']
            fwd_end = min(duration, fwd_start + context * 2)
            if fwd_end - fwd_start > 0.3 and round(fwd_start, 3) not in visited:
                follow_ups.append(dict(
                    start_sec=fwd_start, end_sec=fwd_end,
                    trigger_sec=(fwd_start + fwd_end) / 2,
                    reason='forward_context_check',
                    action='analyze_following_segment',
                    parent_finding=finding['type'],
                ))

        elif finding['type'] == 'uncertainty_remains':
            # Extend search into the next unvisited window forward
            fwd_start = window['end_sec']
            fwd_end = min(duration, fwd_start + context)
            if fwd_end - fwd_start > 0.3 and round(fwd_start, 3) not in visited:
                follow_ups.append(dict(
                    start_sec=fwd_start, end_sec=fwd_end,
                    trigger_sec=(fwd_start + fwd_end) / 2,
                    reason='extended_uncertainty_search',
                    action='analyze_following_segment',
                    parent_finding=finding['type'],
                ))

        return follow_ups

    def run(self, observations, duration, gather):
        """Iterative temporal reasoning loop.

        Each window finding can trigger backward or forward follow-up windows.
        Example: uncertain frame near bed -> look back (was lying 8s ago?) ->
        look forward (stands and walks away?) -> BED_EXIT context confirmed.
        """
        merged = {round(o.timestamp_sec, 9): o for o in observations}
        log = []
        remaining = self.config['review_max_frames']
        visited = set()  # start_sec keys already processed

        queue = self.plan(observations, duration)

        while queue and remaining > 0:
            window = queue.pop(0)
            key = round(window['start_sec'], 3)
            if key in visited:
                continue
            visited.add(key)

            extra, attempted, rejected = gather(window, remaining)
            remaining -= attempted
            added = 0
            for o in extra:
                k = round(o.timestamp_sec, 9)
                if k not in merged:
                    merged[k] = o
                    added += 1

            finding = self._assess_finding(extra)

            # Only schedule follow-ups if window budget allows
            total_so_far = len(log) + len(queue)
            if total_so_far < self.config['review_max_windows']:
                follow_ups = self._follow_up_windows(window, finding, duration, visited)
                slots = self.config['review_max_windows'] - total_so_far
                queue.extend(follow_ups[:slots])
            else:
                follow_ups = []

            log.append({
                **window,
                'attempted_frames': attempted,
                'added_frames': added,
                'identity_rejections': rejected,
                'known_extra_frames': sum(o.state != State.UNKNOWN for o in extra),
                'finding': finding,
                'follow_ups_triggered': len(follow_ups),
                'outcome': 'additional_evidence_gathered' if added else 'insufficient_evidence',
            })

        return sorted(merged.values(), key=lambda o: o.timestamp_sec), {
            'enabled': True,
            'strategy': 'iterative_temporal_reasoning',
            'frame_budget': self.config['review_max_frames'],
            'attempted_frames': self.config['review_max_frames'] - remaining,
            'windows': log,
            'budget_exhausted': remaining == 0,
        }
