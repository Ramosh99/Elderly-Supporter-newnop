"""Bound short posture gaps for event support, without relabelling activities."""
from dataclasses import replace
from ..models import State
from .states import IN_BED
from ..vision.surface import event_location


def bridge_support_gaps(intervals, max_gap=.5):
    result = list(intervals)
    reviews = []
    i = 0
    while i < len(intervals):
        if intervals[i][0].state != State.UNKNOWN:
            i += 1
            continue
        start = i
        while i < len(intervals) and intervals[i][0].state == State.UNKNOWN:
            i += 1
        if start == 0 or i == len(intervals):
            continue
        left,right = intervals[start-1][0],intervals[i][0]
        gap = intervals[i-1][1] - intervals[start][0].timestamp_sec
        middle = [o for o,_ in intervals[start:i]]
        if (gap > max_gap+1e-9 or left.track_id is None or left.track_id != right.track_id
                or not left.bbox or not right.bbox
                or left.state not in IN_BED or right.state not in IN_BED
                or event_location(left) != 'in' or event_location(right) != 'in'
                or any(o.track_id != left.track_id or not o.bbox or o.bed_relation != 'inside'
                       or o.reason != 'ambiguous_posture_or_hidden_legs' for o in middle)):
            continue
        for j in range(start,i):
            o,stop = intervals[j]
            result[j] = (replace(o,state=left.state,confidence=min(left.confidence,right.confidence,.6),
                reason='bridged_bed_support_gap',support_evidence=(
                    'uncalibrated' if o.support_evidence == 'uncalibrated' else 'posture_over_mattress')),stop)
        reviews.append({'time_sec':right.timestamp_sec,'reason':'short_posture_gap_with_bracketed_bed_support',
                        'start_sec':middle[0].timestamp_sec,'end_sec':intervals[i-1][1]})
    return result,reviews
