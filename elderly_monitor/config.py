"""Application defaults and configuration loading."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {
    "sample_fps": 3.0, "bed_region_mode": "manual",
    "refresh_bed_each_sample": True, "bed_model": "yolo11n-seg.pt",
    "bed_confidence": 0.35, "detector": "yolo_pose",
    "yolo_model": "yolo11n-pose.pt", "device": "cpu",
    "min_detection_confidence": 0.35, "keypoint_confidence": 0.5,
    "walking_speed_px_sec": 35.0, "state_hold_sec": 1.5,
    "alert_after_out_of_bed_sec": 300.0, "posture_hold_sec": 0.6,
    "event_hold_sec": 1.5, "context_gap_sec": 2.0,
    "sitting_monitor_sec": 120.0, "min_state_confidence": 0.35,
}


def resolve_config(config: dict) -> dict:
    """Copy settings, fill defaults and validate the bed-region configuration."""
    config = {**DEFAULTS, **config}
    polygon = config.get("bed_polygon")
    if config["bed_region_mode"] not in {"auto", "manual"}:
        raise ValueError("bed_region_mode must be auto or manual")
    if config["bed_region_mode"] == "manual" and (not isinstance(polygon, list) or len(polygon) < 3):
        raise ValueError("config must contain bed_polygon with at least three [x, y] points")
    return config


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return resolve_config(json.load(handle))
