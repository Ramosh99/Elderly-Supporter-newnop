"""Render observation evidence on video frames."""
from ..models import Observation


class AnnotatedVideoWriter:
    """Lazily open an optional output and release it even if inference fails."""

    def __init__(self, path, fps):
        self.path, self.fps, self.writer = path, fps, None

    def write(self, frame, observation):
        import cv2
        if self.path is None:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise ValueError(f"Could not create annotated video: {self.path}")
        annotated = frame.copy()
        draw_annotation(annotated, observation, observation.bed_polygon or [])
        self.writer.write(annotated)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.writer is not None:
            self.writer.release()


def draw_annotation(frame, observation: Observation, bed_polygon: list[list[int]]) -> None:
    import cv2
    import numpy as np

    polygon = np.array(bed_polygon, dtype=np.int32)
    if len(polygon) >= 3:
        cv2.polylines(frame, [polygon], True, (255, 160, 0), 3)
    if observation.bbox is not None:
        x, y, w, h = observation.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 3)
    points = observation.keypoints or []
    for a, b in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
        if len(points) > max(a,b) and points[a][2] >= 0.5 and points[b][2] >= 0.5:
            cv2.line(frame, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])), (0,255,255), 2)
    for p in points:
        if p[2] >= 0.5:
            cv2.circle(frame, tuple(map(int, p[:2])), 4, (0,0,255), -1)
    label = f"{observation.timestamp_sec:05.1f}s candidate: {observation.state.value} conf={observation.confidence:.2f}"
    cv2.rectangle(frame, (12, 12), (12 + min(760, len(label) * 13), 50), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"ID={observation.track_id} {observation.reason}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

