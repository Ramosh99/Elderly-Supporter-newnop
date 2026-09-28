from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .models import Observation, State
from .pose_detector import YoloPoseDetector
from .pose_observer import PoseActivityObserver
from .tracker import TemporalStateTracker
from .bed_detector import detect_bed_region, detect_bed_frame


def _build_detector(config: dict):
    mode = str(config.get("detector", "yolo_pose")).lower()
    min_confidence = float(config.get("min_detection_confidence", 0.35))
    if mode == "yolo_pose":
        return YoloPoseDetector(config["bed_polygon"], config.get("yolo_model", "yolo11n-pose.pt"),
                                min_confidence, config.get("device", "cpu"),
                                config.get("target_track_id")), "yolo_pose_bytetrack"
    raise ValueError(f"Unsupported detector: {mode}. Set detector to 'yolo_pose'; legacy detectors were removed.")


def _draw_annotation(frame, observation: Observation, bed_polygon: list[list[int]]) -> None:
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


def analyze_video(
    video_path: Path,
    config: dict,
    include_observations: bool = False,
    annotated_video_path: Path | None = None,
) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps if frame_count > 0 else 0.0
    sample_fps = float(config.get("sample_fps", 3.0))
    if sample_fps <= 0:
        capture.release()
        raise ValueError("sample_fps must be positive")
    every_n = max(1, round(source_fps / sample_fps))

    try:
        config = dict(config)
        bed_model = None
        if config.get("bed_region_mode", "manual") == "auto":
            config["bed_polygon"], bed_details = detect_bed_region(video_path, config)
            if config.get("refresh_bed_each_sample", True):
                from ultralytics import YOLO
                bed_model = YOLO(config.get("bed_model", "yolo11n-seg.pt"))
            bed_details["refresh_each_sample"] = bed_model is not None
        else:
            bed_details = {"source": "manual", "polygon": config["bed_polygon"]}
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if any(x < 0 or y < 0 or x > width or y > height for x, y in config["bed_polygon"]):
                raise ValueError("Saved bed polygon lies outside this video. Recalibrate it or set bed_region_mode='auto'.")
        detector, detector_name = _build_detector(config)
    except Exception:
        capture.release()
        raise
    observer = PoseActivityObserver(config["bed_polygon"], float(config.get("walking_speed_px_sec", 35.0)),
                                    float(config.get("keypoint_confidence", 0.5)))
    tracker = TemporalStateTracker(
        float(config.get("state_hold_sec", 1.5)),
        float(config.get("alert_after_out_of_bed_sec", 300.0)),
        posture_hold_sec=float(config.get("posture_hold_sec", 0.6)),
        event_hold_sec=float(config.get("event_hold_sec", 1.5)),
        context_gap_sec=float(config.get("context_gap_sec", 2.0)),
        sitting_monitor_sec=float(config.get("sitting_monitor_sec", 120.0)),
        min_confidence=float(config.get("min_state_confidence", 0.35)),
        max_sample_gap_sec=max(1.0, 1.5 * every_n / source_fps),
    )
    observations: list[Observation] = []
    annotated_writer = None
    frame_index = 0
    last_timestamp = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n == 0:
                timestamp = frame_index / source_fps
                bed_available = True
                if bed_model is not None:
                    detected_bed = detect_bed_frame(bed_model, frame, config)
                    bed_available = detected_bed is not None
                    if bed_available:
                        config["bed_polygon"] = detected_bed[0]
                        detector.bed_polygon = detected_bed[0]
                        observer.bed_polygon = detected_bed[0]
                bbox, confidence = detector.detect(frame)
                if bed_available:
                    observation = observer.observe(timestamp, bbox, confidence, detector.last_keypoints, detector.last_track_id)
                    observation = replace(observation, bed_polygon=config["bed_polygon"])
                else:
                    observer.previous = None
                    observation = Observation(timestamp, State.UNKNOWN, 0.0, bbox,
                                              keypoints=detector.last_keypoints,
                                              track_id=detector.last_track_id, reason="bed_not_detected_or_ambiguous")
                tracker.update(observation)
                observations.append(observation)
                last_timestamp = timestamp
                if annotated_video_path is not None:
                    if annotated_writer is None:
                        annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        annotated_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, source_fps / every_n, (width, height))
                        if not annotated_writer.isOpened():
                            raise ValueError(f"Could not create annotated video: {annotated_video_path}")
                    annotated = frame.copy()
                    _draw_annotation(annotated, observation, observation.bed_polygon or [])
                    annotated_writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        if annotated_writer is not None:
            annotated_writer.release()

    result = tracker.finish(duration or last_timestamp)
    result["video"] = str(video_path)
    result["analysis"] = {
        "source_fps": round(source_fps, 3),
        "sample_fps": sample_fps,
        "sampled_frame_count": len(observations),
        "detector": detector_name,
        "bed_region": bed_details,
        "target_track_id": getattr(detector, "target_track_id", None),
        "known_observation_fraction": round(sum(o.state != State.UNKNOWN for o in observations) / max(1, len(observations)), 3),
        "temporal_policy": {name: getattr(tracker, name) for name in (
            "hold_sec", "posture_hold_sec", "event_hold_sec", "context_gap_sec",
            "sitting_monitor_sec", "alert_after_sec", "min_confidence", "max_sample_gap_sec")},
    }
    result["analysis"]["warnings"] = []
    if result["analysis"]["known_observation_fraction"] < 0.5:
        result["analysis"]["warnings"].append(
            "Most sampled frames have uncertain posture or a missing target. Inspect the annotated video; activity totals are incomplete."
        )
    if annotated_video_path is not None:
        result["annotated_video"] = str(annotated_video_path)
    if include_observations:
        result["observations"] = [observation.to_dict() for observation in observations]
    return result


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    polygon = config.get("bed_polygon")
    if config.get("bed_region_mode", "manual") not in {"auto", "manual"}:
        raise ValueError("bed_region_mode must be auto or manual")
    if config.get("bed_region_mode", "manual") == "manual" and (not isinstance(polygon, list) or len(polygon) < 3):
        raise ValueError("config must contain bed_polygon with at least three [x, y] points")
    return config
