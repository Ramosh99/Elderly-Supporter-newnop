"""Bounded, bracketed posture corrections without isolated label substitutions."""
from dataclasses import replace
from ..models import State
from ..temporal.tracker import TemporalStateTracker


def apply_sequence_assessments(selected, assessments, minimum, context):
    # Import here to keep transport and policy separately testable.
    from .gemini import validate_assessment
    assessments = validate_assessment({'frames':assessments},len(selected))
    ordered = sorted(assessments,key=lambda a:selected[a['index']].timestamp_sec)
    updates, spans = {}, []
    applications = {a['index']:'advisory_no_agreeing_bracket' for a in ordered}
    for left,right in zip(ordered,ordered[1:]):
        a,b = selected[left['index']],selected[right['index']]
        _no_det_r = {'missing_person_detection','no_person_detected','tracked_target_missing','tracking_id_unavailable'}
        both_no_detection = a.bbox is None and b.bbox is None and a.reason in _no_det_r and b.reason in _no_det_r
        if (left['state'] != right['state'] or not left['target_clear'] or not right['target_clear']
                or min(left['confidence'],right['confidence']) < minimum
                or (not both_no_detection and (a.track_id is None or a.track_id != b.track_id))
                or not 0 < b.timestamp_sec-a.timestamp_sec <= 1.5):
            continue
        state = State(left['state'])
        if state == State.UNKNOWN:
            continue
        between = [o for o in context if a.timestamp_sec <= o.timestamp_sec <= b.timestamp_sec]
        rejection = None
        if (len(between) < 2 or between[0].timestamp_sec != a.timestamp_sec
                or between[-1].timestamp_sec != b.timestamp_sec
                or any(y.timestamp_sec-x.timestamp_sec > 1.0 for x,y in zip(between,between[1:]))):
            rejection = 'rejected_evidence_gap'
        _no_det_reasons = {'missing_person_detection','no_person_detected','tracked_target_missing','tracking_id_unavailable'}
        all_no_detection = all(o.bbox is None and o.reason in _no_det_reasons for o in between)
        for o in between:
            if o.track_id != a.track_id or not o.bed_polygon:
                rejection = 'rejected_identity_or_geometry_gap'
            elif not o.bbox and o.reason not in _no_det_reasons:
                rejection = 'rejected_identity_or_geometry_gap'
            elif o.state == State.UNKNOWN and o.reason not in _no_det_reasons and o.reason != 'ambiguous_posture_or_hidden_legs':
                rejection = 'rejected_visibility_gap'
            elif not all_no_detection and ((state in {State.LYING_IN_BED,State.SITTING_ON_BED} and o.bed_relation != 'inside')
                  or (state in {State.SITTING_OUTSIDE_BED,State.OUT_OF_BED} and o.bed_relation != 'away')
                  or (state in {State.STANDING,State.WALKING} and o.bed_relation not in {'inside','near','away'})):
                rejection = 'rejected_spatial_conflict'
            elif (state == State.WALKING and o.state != State.WALKING
                  or state == State.STANDING and o.state == State.WALKING):
                # Sparse stills must not invent or erase measured locomotion.
                rejection = 'rejected_motion_override'
        if rejection:
            for item in (left,right):
                if not applications[item['index']].startswith('accepted'):
                    applications[item['index']] = rejection
            continue
        changed = []
        for o in between:
            if o.state != state:
                conf_cap = 0.45 if (o.bbox is None and o.reason in _no_det_reasons) else 0.8
                updates[o.timestamp_sec] = replace(o,state=state,
                    confidence=min(conf_cap,left['confidence'],right['confidence']),reason='gemini_sequence_posture')
                changed.append(o.timestamp_sec)
        for item in (left,right):
            applications[item['index']] = 'accepted_bracketed_sequence' if changed else 'agrees_with_sequence'
        spans.append({'start_sec':a.timestamp_sec,'end_sec':b.timestamp_sec,'state':state.value,
                      'changed_timestamps_sec':changed})
    decisions = [{'timestamp_sec':selected[a['index']].timestamp_sec,
                  'original_state':selected[a['index']].state.value,**a,
                  'application':applications[a['index']]} for a in assessments]
    return updates, decisions, spans


def guard_fragmentation(context, changes, duration, config):
    """Reject a proposal if deterministic smoothing creates more UNKNOWN time.

    This is a non-degradation check, not a ground-truth accuracy guarantee.
    """
    kwargs = dict(hold_sec=config['state_hold_sec'],posture_hold_sec=config['posture_hold_sec'],
                  event_hold_sec=config['event_hold_sec'],context_gap_sec=config['context_gap_sec'],
                  min_confidence=config['min_state_confidence'],
                  max_sample_gap_sec=max(1.,1.5/config['sample_fps']))
    unknown = []
    for changed in (False,True):
        tracker = TemporalStateTracker(**kwargs)
        for o in context:
            tracker.update(changes.get(o.timestamp_sec,o) if changed else o)
        unknown.append(tracker.finish(duration)['total_unknown_sec'])
    return unknown[1] <= unknown[0] + .001, {'before_unknown_sec':unknown[0],'proposed_unknown_sec':unknown[1]}
