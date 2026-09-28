"""Video metadata, chronological sampling and bounded segment access."""
from __future__ import annotations

import cv2


class VideoReader:
    def __init__(self, path, sample_fps=3.0):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Could not open video: {path}")
        self.source_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.source_fps if self.frame_count > 0 else 0.0
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if sample_fps <= 0:
            self.close()
            raise ValueError("sample_fps must be positive")
        self.every_n = max(1, round(self.source_fps / sample_fps))
        self.output_fps = self.source_fps / self.every_n

    def frames(self, start_sec=0.0, end_sec=None):
        """Yield timestamp/frame pairs. Optional bounds enable later context review."""
        if start_sec < 0 or (end_sec is not None and end_sec < start_sec):
            raise ValueError("Invalid video segment bounds")
        index = round(start_sec * self.source_fps)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        while end_sec is None or index / self.source_fps < end_sec:
            ok, frame = self.capture.read()
            if not ok:
                break
            if index % self.every_n == 0:
                yield index / self.source_fps, frame
            index += 1

    def close(self):
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
