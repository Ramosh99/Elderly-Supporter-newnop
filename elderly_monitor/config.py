"""Application defaults and configuration loading."""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULTS = {
    "gemini_enabled": False, "gemini_model": "gemini-3.1-flash-lite",
    "gemini_max_requests": 6, "gemini_frames_per_request": 5,
    "gemini_timeout_sec": 30.0, "gemini_min_confidence": 0.8,
    "gemini_max_retries": 2, "gemini_retry_backoff_sec": 1.0,
    "gemini_cache_dir": "output/gemini_cache", "gemini_correction_policy": "sequence",
    "review_enabled": True, "review_context_sec": 2.0,
    "review_sample_fps": 9.0, "review_max_windows": 6,
    "review_max_frames": 120, "review_anchor_gap_sec": 1.0,
    "sample_fps": 3.0, "bed_region_mode": "manual",
    "refresh_bed_each_sample": False, "bed_model": "yolo11n-seg.pt",
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
    import re
    if config['gemini_correction_policy'] not in {'sequence','legacy'}:
        raise ValueError('gemini_correction_policy must be sequence or legacy')
    if type(config['gemini_max_retries']) is not int or not 0 <= config['gemini_max_retries'] <= 3:
        raise ValueError('gemini_max_retries must be an integer from 0 to 3')
    if not isinstance(config['gemini_cache_dir'], str):
        raise ValueError('gemini_cache_dir must be a string (empty disables caching)')
    config['gemini_retry_backoff_sec'] = float(config['gemini_retry_backoff_sec'])
    if not math.isfinite(config['gemini_retry_backoff_sec']) or not 0 <= config['gemini_retry_backoff_sec'] <= 8:
        raise ValueError('gemini_retry_backoff_sec must be between 0 and 8')
    if type(config['gemini_enabled']) is not bool:
        raise ValueError('gemini_enabled must be true or false')
    if not isinstance(config['gemini_model'],str) or not re.fullmatch(r'[A-Za-z0-9._-]+',config['gemini_model']):
        raise ValueError('Invalid Gemini model name')
    for key, low, high in [('gemini_max_requests',0,10),('gemini_frames_per_request',2,8)]:
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f'{key} must be an integer between {low} and {high}')
    for key, low, high in [('gemini_timeout_sec',1,60),('gemini_min_confidence',0,1)]:
        config[key] = float(config[key])
        if not math.isfinite(config[key]) or not low <= config[key] <= high:
            raise ValueError(f'Invalid {key}')
    if not isinstance(config['review_enabled'], bool):
        raise ValueError('review_enabled must be true or false')
    for key in ('review_context_sec', 'review_sample_fps', 'review_anchor_gap_sec'):
        config[key] = float(config[key])
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('review_max_windows', 'review_max_frames'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    polygon = config.get("bed_polygon")
    for name in ('mattress_polygon', 'floor_polygon'):
        points = config.get(name)
        if points is None:
            continue
        if (not isinstance(points, list) or len(points) < 3
                or any(not isinstance(p, (list,tuple)) or len(p) != 2
                       or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in p)
                       for p in points)):
            raise ValueError(f'{name} needs at least three finite nonnegative [x,y] points')
        area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))
        if abs(area) < 1:
            raise ValueError(f'{name} has no usable area')
    if config.get('floor_polygon') and not config.get('mattress_polygon'):
        raise ValueError('floor_polygon requires mattress_polygon')
    if config.get('mattress_polygon'):
        calibration = config.get('surface_calibration')
        if (not isinstance(calibration,dict) or not isinstance(calibration.get('video'),str)
                or not isinstance(calibration.get('frame_size'),list)
                or len(calibration['frame_size']) != 2
                or any(type(v) is not int or v <= 0 for v in calibration['frame_size'])):
            raise ValueError('Surface calibration requires video and frame_size; use the region selector')
    if config["bed_region_mode"] not in {"auto", "manual"}:
        raise ValueError("bed_region_mode must be auto or manual")
    if config["bed_region_mode"] == "manual" and (not isinstance(polygon, list) or len(polygon) < 3):
        raise ValueError("config must contain bed_polygon with at least three [x, y] points")
    return config


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return resolve_config(json.load(handle))
