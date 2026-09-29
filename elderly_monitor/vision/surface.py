"""Conservative image-plane support evidence for calibrated fixed cameras."""
from .geometry import distance_to_polygon, point_in_polygon
from ..models import State
from dataclasses import replace
from .geometry import distance


def refresh_support(o, config):
    """Recompute geometry evidence after a VLM posture correction."""
    points = o.keypoints or []
    threshold = config['keypoint_confidence']
    def mid(a,b):
        usable = [points[i][:2] for i in (a,b) if len(points)>i and points[i][2]>=threshold]
        return tuple(sum(p[j] for p in usable)/len(usable) for j in (0,1)) if usable else None
    shoulders,hips = mid(5,6),mid(11,12)
    margin = max(5.,distance(shoulders,hips)*.2) if shoulders and hips else 5.
    return replace(o, support_evidence=support_evidence(o.state,points,o.mattress_polygon,
                   config.get('floor_polygon'),threshold,margin))


def support_evidence(state, points, surface, floor, threshold, margin):
    if not surface:
        return 'uncalibrated'
    def joint(i):
        if points and len(points) > i and points[i][2] >= threshold:
            return points[i][:2]
        return None
    hips = [p for p in (joint(11), joint(12)) if p is not None]
    ankles = [p for p in (joint(15), joint(16)) if p is not None]
    if state == State.UNKNOWN:
        return 'uncertain'
    if state in {State.SITTING_ON_BED, State.LYING_IN_BED} and hips:
        if all(distance_to_polygon(p, surface) <= margin for p in hips):
            return 'posture_over_mattress'
    if state in {State.STANDING, State.WALKING}:
        # Projection is not proof of contact. Feet over the surface must block
        # an automatic floor/exit conclusion, even if the floor polygon overlaps.
        if ankles and any(point_in_polygon(p, surface) for p in ankles):
            return 'feet_over_mattress'
        if len(ankles) == 2 and floor and all(point_in_polygon(p, floor) for p in ankles):
            return 'feet_on_floor_region'
    if state == State.SITTING_OUTSIDE_BED and hips and floor:
        if all(point_in_polygon(p, floor) and distance_to_polygon(p, surface) > margin for p in hips):
            return 'seated_outside_surface'
    return 'uncertain'


def event_location(o):
    """Calibrated support governs events; legacy observations retain old policy."""
    from ..temporal.states import IN_BED, OUT_OF_BED
    if o.support_evidence != 'uncalibrated':
        if o.state in IN_BED and o.support_evidence == 'posture_over_mattress':
            return 'in'
        if o.state in OUT_OF_BED and o.support_evidence in {'feet_on_floor_region', 'seated_outside_surface'}:
            return 'away'
        return None
    return ('in' if o.state in IN_BED else 'away'
            if o.state in OUT_OF_BED and o.bed_relation in {'inside','near','away'} else None)
