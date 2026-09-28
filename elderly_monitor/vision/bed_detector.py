from __future__ import annotations

import cv2
import numpy as np


def occupancy_polygon(polygon):
    """Approximate the full bed footprint, bridging person-shaped cutouts.

    This convex envelope is a spatial prior, not a measured mattress surface.
    Keep the raw segmentation separately for inspection.
    """
    return cv2.convexHull(np.asarray(polygon, dtype=np.int32)).reshape(-1, 2).tolist()


def detect_bed_frame(model, frame, config):
    bed_class = next((i for i, name in model.names.items() if name == "bed"), None)
    if bed_class is None:
        raise ValueError("The bed model must include a 'bed' class")
    result = model.predict(frame, classes=[bed_class],
                           conf=float(config.get("bed_confidence", 0.35)),
                           device=config.get("device", "cpu"), verbose=False)[0]
    if result.masks is None or len(result.boxes) != 1:
        return None
    raw = result.masks.xy[0].astype(np.int32)
    if len(raw) < 3:
        return None
    return occupancy_polygon(raw), raw.tolist(), float(result.boxes.conf[0])


def detect_bed_region(video_path, config):
    """Calibrate a fixed-camera bed polygon from consistent segmentation masks."""
    from ultralytics import YOLO

    model = YOLO(config.get("bed_model", "yolo11n-seg.pt"))
    bed_class = next((i for i, name in model.names.items() if name == "bed"), None)
    if bed_class is None:
        raise ValueError("The bed model must include a 'bed' class")
    capture = cv2.VideoCapture(str(video_path))
    candidates = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample the beginning; keep one actual mask rather than a rectangle or hull.
        indices = np.linspace(0, min(max(0, count - 1), int(fps * 4)), 5).astype(int)
        for index in sorted(set(indices.tolist())):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            result = model.predict(frame, classes=[bed_class],
                                   conf=float(config.get("bed_confidence", 0.35)),
                                   device=config.get("device", "cpu"), verbose=False)[0]
            if result.masks is None or len(result.boxes) != 1:
                continue
            polygon = result.masks.xy[0].astype(np.int32)
            if len(polygon) < 3:
                continue
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            candidates.append((polygon, mask, float(result.boxes.conf[0]), int(index)))
    finally:
        capture.release()
    if len(candidates) < 2:
        raise ValueError("Could not identify one bed consistently. Use bed_region_mode='manual' and calibrate this video's bed polygon.")
    scores = []
    for _, mask, _, _ in candidates:
        scores.append([np.count_nonzero(mask & other) / max(1, np.count_nonzero(mask | other))
                       for _, other, _, _ in candidates])
    best = max(range(len(candidates)), key=lambda i: sum(v >= 0.5 for v in scores[i]))
    support = sum(v >= 0.5 for v in scores[best])
    if support < 2:
        raise ValueError("Bed masks disagree across frames. Recalibrate using a manual polygon.")
    polygon, mask, confidence, index = candidates[best]
    region = occupancy_polygon(polygon)
    return region, {
        "source": "yolo_segmentation", "model": config.get("bed_model", "yolo11n-seg.pt"),
        "confidence": round(confidence, 3), "supporting_frames": int(support),
        "selected_frame": index, "frame_size": [mask.shape[1], mask.shape[0]],
        "polygon": region, "raw_polygon": polygon.tolist(),
        "geometry": "convex_occupancy_envelope",
    }
