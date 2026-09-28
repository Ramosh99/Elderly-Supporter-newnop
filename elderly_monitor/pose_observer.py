from __future__ import annotations

import math

from .geometry import bbox_bed_overlap, distance, point_in_polygon, distance_to_polygon
from .models import Observation, State


class PoseActivityObserver:
    """Camera-dependent posture heuristics using confident COCO body joints."""

    def __init__(self, bed_polygon, walking_speed_px_sec=35.0, keypoint_confidence=0.5):
        self.bed_polygon = bed_polygon
        self.walking_speed = walking_speed_px_sec
        self.threshold = keypoint_confidence
        self.previous = None

    def observe(self, timestamp_sec, bbox, detector_confidence, keypoints=None, track_id=None):
        overlap = bbox_bed_overlap(bbox, self.bed_polygon) if bbox else 0.0
        relation = "unknown"
        def output(state, confidence, reason, speed=0.0):
            return Observation(timestamp_sec, state, confidence, bbox, overlap, speed,
                               keypoints, track_id, reason, bed_relation=relation)
        def joint(index):
            if keypoints is None or len(keypoints) <= index:
                return None
            p = keypoints[index]
            return p[:2] if len(p) >= 3 and p[2] >= self.threshold else None
        def midpoint(a, b):
            points = [p for p in (joint(a), joint(b)) if p is not None]
            return tuple(sum(p[i] for p in points)/len(points) for i in (0, 1)) if points else None

        shoulders, hips = midpoint(5, 6), midpoint(11, 12)
        if bbox is None or shoulders is None or hips is None or distance(shoulders, hips) < 5:
            self.previous = None
            return output(State.UNKNOWN, 0.0, "missing_person_or_torso_keypoints")
        speed = 0.0
        if self.previous and self.previous[2] == track_id:
            elapsed = timestamp_sec - self.previous[0]
            if elapsed > 0:
                speed = distance(hips, self.previous[1]) / elapsed
        self.previous = (timestamp_sec, hips, track_id)
        on_bed = point_in_polygon(hips, self.bed_polygon)
        margin = max(5.0, distance(shoulders, hips) * 0.2)
        relation = "inside" if on_bed else "near" if distance_to_polygon(hips, self.bed_polygon) <= margin else "away"
        torso_angle = math.degrees(math.atan2(abs(shoulders[0]-hips[0]), abs(shoulders[1]-hips[1])))
        if torso_angle >= 55:
            if relation == "near":
                return output(State.UNKNOWN, 0.0, "uncertain_bed_boundary", speed)
            state = State.LYING_IN_BED if on_bed and overlap >= 0.25 else State.OUT_OF_BED
            return output(state, min(detector_confidence, 0.8), "horizontal_torso", speed)

        angles = []
        for h, k, a in ((11, 13, 15), (12, 14, 16)):
            hip, knee, ankle = joint(h), joint(k), joint(a)
            if hip is None or knee is None or ankle is None:
                continue
            u = (hip[0]-knee[0], hip[1]-knee[1])
            v = (ankle[0]-knee[0], ankle[1]-knee[1])
            norm = math.hypot(*u)*math.hypot(*v)
            if norm > 25:
                angles.append(math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/norm)))))
        if not angles or torso_angle > 40:
            return output(State.UNKNOWN, 0.0, "ambiguous_posture_or_hidden_legs", speed)
        if sum(angles)/len(angles) < 145:
            if relation == "near":
                return output(State.UNKNOWN, 0.0, "uncertain_bed_boundary", speed)
            state = State.SITTING_ON_BED if on_bed else State.SITTING_OUTSIDE_BED
            reason = "upright_torso_bent_knees"
        else:
            state = State.WALKING if speed >= self.walking_speed else State.STANDING
            reason = "upright_torso_extended_legs"
        return output(state, min(detector_confidence, 0.75), reason, speed)
