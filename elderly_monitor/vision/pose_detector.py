from __future__ import annotations

from .geometry import bbox_bed_overlap


class YoloPoseDetector:
    """Initialize from one person, or an unambiguous bed occupant; retain their ID.

    Missing IDs yield UNKNOWN rather than silently switching to a visitor.
    ByteTrack is not appearance re-identification after long disappearances.
    """

    def __init__(self, bed_polygon, model="yolo11n-pose.pt", min_confidence=0.35,
                 device="cpu", target_track_id=None):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.bed_polygon = bed_polygon
        self.min_confidence = min_confidence
        self.device = device
        self.target_track_id = target_track_id
        self.last_keypoints = None
        self.last_track_id = None
        self.last_reason = "not_run"

    def select_target(self, candidates):
        if self.target_track_id is None:
            near_bed = [c for c in candidates if bbox_bed_overlap(c["bbox"], self.bed_polygon) >= 0.25]
            eligible = candidates if len(candidates) == 1 else near_bed
            if len(eligible) != 1:
                self.last_reason = "ambiguous_target_identity" if candidates else "no_person_detected"
                return None
            self.target_track_id = eligible[0]["id"]
        selected = next((c for c in candidates if c["id"] == self.target_track_id), None)
        self.last_reason = "target_selected" if selected is not None else "tracked_target_missing"
        return selected

    def detect(self, frame):
        self.last_keypoints = None
        self.last_track_id = None
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=self.min_confidence, device=self.device,
                                  verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.last_reason = "no_person_detected"
            return None, 0.0
        if result.boxes.id is None or result.keypoints is None:
            self.last_reason = "tracking_id_unavailable" if result.boxes.id is None else "pose_keypoints_unavailable"
            return None, 0.0
        candidates = []
        for box, conf, identity, points in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
            result.boxes.id.cpu().tolist(), result.keypoints.data.cpu().tolist()
        ):
            x1, y1, x2, y2 = map(int, box)
            candidates.append(dict(bbox=(x1, y1, x2-x1, y2-y1), confidence=conf,
                                   id=int(identity), keypoints=points))
        selected = self.select_target(candidates)
        if selected is None:
            return None, 0.0
        self.last_keypoints = selected["keypoints"]
        self.last_track_id = selected["id"]
        return selected["bbox"], selected["confidence"]
