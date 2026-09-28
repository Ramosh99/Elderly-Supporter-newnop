from __future__ import annotations

from .geometry import bbox_bed_overlap


class YoloPoseDetector:
    """Lock onto the first unambiguous person at the bed, then retain their ID.

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

    def select_target(self, candidates):
        if self.target_track_id is None:
            near_bed = [c for c in candidates if bbox_bed_overlap(c["bbox"], self.bed_polygon) >= 0.25]
            if len(near_bed) != 1:
                return None
            self.target_track_id = near_bed[0]["id"]
        return next((c for c in candidates if c["id"] == self.target_track_id), None)

    def detect(self, frame):
        self.last_keypoints = None
        self.last_track_id = None
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=self.min_confidence, device=self.device,
                                  verbose=False)[0]
        if result.boxes.id is None or result.keypoints is None:
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
