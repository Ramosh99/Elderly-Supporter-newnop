"""Coordinate vision, temporal review and video output."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import resolve_config
from .models import Observation, State
from .vision.pose_detector import YoloPoseDetector
from .vision.pose_observer import PoseActivityObserver
from .vision.bed_detector import detect_bed_region, detect_bed_frame
from .temporal.tracker import TemporalStateTracker
from .video.reader import VideoReader
from .video.annotations import AnnotatedVideoWriter


def _build_detector(config):
    mode = str(config["detector"]).lower()
    if mode != "yolo_pose":
        raise ValueError(f"Unsupported detector: {mode}. Set detector to 'yolo_pose'; legacy detectors were removed.")
    return YoloPoseDetector(config["bed_polygon"], config["yolo_model"],
                            float(config["min_detection_confidence"]), config["device"],
                            config.get("target_track_id")), "yolo_pose_bytetrack"


def _prepare_bed(video_path, config, reader):
    model = None
    if config["bed_region_mode"] == "auto":
        config["bed_polygon"], details = detect_bed_region(video_path, config)
        if config["refresh_bed_each_sample"]:
            from ultralytics import YOLO
            model = YOLO(config["bed_model"])
        details["refresh_each_sample"] = model is not None
    else:
        details = {"source": "manual", "polygon": config["bed_polygon"]}
        if any(x < 0 or y < 0 or x > reader.width or y > reader.height for x, y in config["bed_polygon"]):
            raise ValueError("Saved bed polygon lies outside this video. Recalibrate it or set bed_region_mode='auto'.")
    return model, details


def _build_tracker(config, reader):
    return TemporalStateTracker(
        float(config["state_hold_sec"]), float(config["alert_after_out_of_bed_sec"]),
        posture_hold_sec=float(config["posture_hold_sec"]),
        event_hold_sec=float(config["event_hold_sec"]),
        context_gap_sec=float(config["context_gap_sec"]),
        sitting_monitor_sec=float(config["sitting_monitor_sec"]),
        min_confidence=float(config["min_state_confidence"]),
        max_sample_gap_sec=max(1.0, 1.5 * reader.every_n / reader.source_fps),
    )


def analyze_video(video_path: Path, config: dict, include_observations: bool = False,
                  annotated_video_path: Path | None = None) -> dict:
    config = resolve_config(config)
    observations: list[Observation] = []
    last_timestamp = 0.0
    with VideoReader(video_path, float(config["sample_fps"])) as reader:
        bed_model, bed_details = _prepare_bed(video_path, config, reader)
        detector, detector_name = _build_detector(config)
        observer = PoseActivityObserver(config["bed_polygon"], float(config["walking_speed_px_sec"]),
                                        float(config["keypoint_confidence"]))
        tracker = _build_tracker(config, reader)
        with AnnotatedVideoWriter(annotated_video_path, reader.output_fps) as writer:
            for timestamp, frame in reader.frames():
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
                writer.write(frame, observation)

    result = tracker.finish(reader.duration or last_timestamp)
    result["video"] = str(video_path)
    result["analysis"] = {
        "source_fps": round(reader.source_fps, 3),
        "sample_fps": float(config["sample_fps"]),
        "sampled_frame_count": len(observations),
        "detector": detector_name, "bed_region": bed_details,
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
