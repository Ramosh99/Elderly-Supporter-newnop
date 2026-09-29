from __future__ import annotations

import math

from .geometry import bbox_bed_overlap, distance, point_in_polygon, distance_to_polygon
from ..models import Observation, State
from .surface import support_evidence


class PoseActivityObserver:
    """Camera-dependent posture heuristics using confident COCO body joints."""

    def __init__(self, bed_polygon, walking_speed_px_sec=35.0, keypoint_confidence=0.5,
                 mattress_polygon=None, floor_polygon=None):
        self.bed_polygon = bed_polygon
        self.mattress_polygon = mattress_polygon
        self.floor_polygon = floor_polygon
        self.walking_speed = walking_speed_px_sec
        self.threshold = keypoint_confidence
        self.previous = None
        self.motion_history = []
        self.last_state = State.UNKNOWN
        self.last_sitting_time = None
        self.motion_evidence = None

    def observe(self, timestamp_sec, bbox, detector_confidence, keypoints=None, track_id=None):
        self.motion_evidence = None
        region = self.mattress_polygon or self.bed_polygon
        overlap = bbox_bed_overlap(bbox, region) if bbox else 0.0
        relation = "unknown"
        support_margin = 5.0
        def output(state, confidence, reason, speed=0.0):
            self.last_state = state
            if state in {State.SITTING_ON_BED, State.SITTING_OUTSIDE_BED}:
                self.last_sitting_time = timestamp_sec
            if state == State.UNKNOWN:
                self.previous = None
                self.motion_history.clear()
                self.last_sitting_time = None
            return Observation(timestamp_sec, state, confidence, bbox, overlap, speed,
                               keypoints, track_id, reason, bed_relation=relation,
                               detector_confidence=detector_confidence,
                               mattress_polygon=self.mattress_polygon,
                               support_evidence=support_evidence(state, keypoints, self.mattress_polygon,
                                   self.floor_polygon, self.threshold, support_margin),
                               motion_evidence=self.motion_evidence)
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
            return output(State.UNKNOWN, 0.0, "missing_person_detection" if bbox is None else "missing_or_degenerate_torso_keypoints")
        if (self.previous is None or track_id is None or self.previous[2] != track_id
                or not 0 < timestamp_sec - self.previous[0] <= 1.0):
            self.motion_history.clear()
            self.last_state = State.UNKNOWN
            self.last_sitting_time = None
        speed = 0.0
        if self.previous and self.previous[2] == track_id:
            elapsed = timestamp_sec - self.previous[0]
            if elapsed > 0:
                speed = distance(hips, self.previous[1]) / elapsed
        self.previous = (timestamp_sec, hips, track_id)
        scale = distance(shoulders, hips)
        ankles = {i: joint(i) for i in (15, 16) if joint(i) is not None}
        self.motion_history = [p for p in self.motion_history if timestamp_sec-p[0] <= 1.0]
        self.motion_history.append((timestamp_sec, hips, ankles, scale))
        on_bed = point_in_polygon(hips, region)
        margin = max(5.0, distance(shoulders, hips) * 0.2)
        support_margin = margin
        relation = "inside" if on_bed else "near" if distance_to_polygon(hips, region) <= margin else "away"
        torso_angle = math.degrees(math.atan2(abs(shoulders[0]-hips[0]), abs(shoulders[1]-hips[1])))
        if torso_angle >= 55:
            if relation == "near":
                return output(State.UNKNOWN, 0.0, "uncertain_bed_boundary", speed)
            state = State.LYING_IN_BED if on_bed and overlap >= 0.25 else State.OUT_OF_BED
            return output(state, min(detector_confidence, 0.8), "horizontal_torso", speed)

        angles = []
        seated_thighs = []
        for h, k, a in ((11, 13, 15), (12, 14, 16)):
            hip, knee, ankle = joint(h), joint(k), joint(a)
            if hip is None or knee is None or ankle is None:
                continue
            # Extended knees do not imply standing: inspect thigh direction
            # relative to the torso, with both legs supported by bed geometry.
            shoulder = joint(5 if h == 11 else 6)
            if shoulder is not None:
                torso = (shoulder[0]-hip[0], shoulder[1]-hip[1])
                thigh = (knee[0]-hip[0], knee[1]-hip[1])
                size = math.hypot(*torso)*math.hypot(*thigh)
                if size > 25:
                    hip_angle = math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(torso,thigh))/size))))
                    seated_thighs.append(hip_angle < 145 and abs(thigh[0]) > abs(thigh[1])
                                         and point_in_polygon(knee,region)
                                         and point_in_polygon(ankle,region))
            u = (hip[0]-knee[0], hip[1]-knee[1])
            v = (ankle[0]-knee[0], ankle[1]-knee[1])
            norm = math.hypot(*u)*math.hypot(*v)
            if norm > 25:
                angles.append(math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/norm)))))
        if not angles or torso_angle > 40:
            return output(State.UNKNOWN, 0.0, "ambiguous_posture_or_hidden_legs", speed)
        if on_bed and len(seated_thighs) == 2 and all(seated_thighs):
            return output(State.SITTING_ON_BED, min(detector_confidence, 0.75),
                          "upright_flexed_hips_legs_over_bed", speed)
        if sum(angles)/len(angles) < 145:
            if relation == "near":
                return output(State.UNKNOWN, 0.0, "uncertain_bed_boundary", speed)
            state = State.SITTING_ON_BED if on_bed else State.SITTING_OUTSIDE_BED
            reason = "upright_torso_bent_knees"
        else:
            state, reason = self._upright_motion(timestamp_sec)
        return output(state, min(detector_confidence, 0.75), reason, speed)

    def _upright_motion(self, timestamp):
        """Require sustained hip AND corresponding ankle displacement.

        Speeds are torso lengths/second. The legacy pixel threshold maps to a
        reference 100-pixel torso, preserving configuration without camera scale bias.
        This assumes a fixed camera; it is not optical-flow compensation.
        """
        first, last = self.motion_history[0], self.motion_history[-1]
        elapsed = last[0] - first[0]
        self.motion_evidence = {'window_sec':elapsed,'sample_count':len(self.motion_history),
                                'visible_ankles':len(last[2])}
        if elapsed < 0.5:
            return State.STANDING, "upright_motion_pending"
        common = first[2].keys() & last[2].keys()
        if not common:
            return State.STANDING, "upright_without_ankle_motion_evidence"
        scale = (first[3] + last[3]) / 2
        # Net displacement loses short reversing steps and turning motion.
        # Sum movement through the window, ignoring sub-2%-torso jitter.
        hip_path = 0.0
        foot_paths = {i: 0.0 for i in common}
        for a, b in zip(self.motion_history, self.motion_history[1:]):
            step_scale = (a[3] + b[3]) / 2
            hip_step = distance(a[1], b[1]) / step_scale
            hip_path += hip_step if hip_step >= .02 else 0.0
            for i in common:
                if i in a[2] and i in b[2]:
                    step = distance(a[2][i], b[2][i]) / step_scale
                    foot_paths[i] += step if step >= .02 else 0.0
        hip_speed = hip_path / elapsed
        foot_speed = max(foot_paths.values()) / elapsed
        threshold = self.walking_speed / 100.0
        self.motion_evidence.update(hip_torso_lengths_per_sec=hip_speed,
            ankle_torso_lengths_per_sec=foot_speed,entry_threshold=threshold,
            matched_ankles=len(common))
        # Hysteresis avoids toggling at the entry threshold, but never retains
        # walking without foot evidence. Planted feet reject vertical rising.
        maintaining = self.last_state == State.WALKING
        # Slow shuffling can move the pelvis little while feet take real steps.
        # Allow that only with stronger foot evidence than the ordinary entry.
        slow_steps = hip_speed >= threshold * .5 and foot_speed >= threshold
        moving = hip_speed >= threshold * (0.65 if maintaining else 1.0) or slow_steps
        stepping = foot_speed >= threshold * (0.35 if maintaining else 0.5)
        recent_sitting = self.last_sitting_time is not None and timestamp-self.last_sitting_time <= 1.0
        if recent_sitting and not stepping:
            return State.STANDING, "rising_with_planted_feet"
        if moving and stepping:
            return State.WALKING, "sustained_hip_and_ankle_motion"
        return State.STANDING, "upright_without_sustained_steps"
