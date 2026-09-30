from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Click points around the bed to select a polygon")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("config.json"))
    parser.add_argument('--region', choices=['bed','mattress','floor'], default='bed',
                        help='Mattress means the top surface only; floor means the visible walking area')
    parser.add_argument('--time', type=float, default=0, help='Source time in seconds for calibration')
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if args.time < 0:
        raise ValueError('Calibration time must be nonnegative')
    capture.set(cv2.CAP_PROP_POS_MSEC,args.time*1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read first frame from {args.video}")
    import numpy as np
    points = []
    window = f"Click {args.region} boundary | Enter: save | Backspace: undo | Esc: cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
    cv2.setMouseCallback(window, click)
    while True:
        preview = frame.copy()
        if points:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], len(points) >= 3, (255,160,0), 3)
            for p in points:
                cv2.circle(preview, tuple(p), 5, (0,0,255), -1)
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and len(points) >= 3:
            break
        if key in (8, 127) and points:
            points.pop()
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            raise ValueError("Bed selection cancelled; configuration unchanged")
    cv2.destroyAllWindows()
    config = {
        "bed_polygon": points,
        "sample_fps": 3.0,
        "detector": "yolo_pose",
        "yolo_model": "yolo11n-pose.pt",
        "min_detection_confidence": 0.35,
        "state_hold_sec": 1.5,
        "walking_speed_px_sec": 35.0,
        "alert_after_out_of_bed_sec": 300.0,
    }
    if args.output.exists():
        config.update(json.loads(args.output.read_text(encoding="utf-8")))
    if args.region == 'bed':
        config.update(bed_polygon=points, bed_region_mode="manual")
    else:
        calibration = {'video':str(args.video.resolve()),'frame_size':[frame.shape[1],frame.shape[0]],
                       'timestamp_sec':args.time}
        if args.region == 'floor':
            previous = config.get('surface_calibration',{})
            if not config.get('mattress_polygon') or previous.get('video') != calibration['video'] or previous.get('frame_size') != calibration['frame_size']:
                raise ValueError('Calibrate the mattress for this video into this config first')
            config['floor_polygon'] = points
        else:
            config['mattress_polygon'] = points
            config['surface_calibration'] = calibration
            config.pop('floor_polygon',None)  # Previous floor calibration may belong to another view.
        config['refresh_bed_each_sample'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {args.region} region to {args.output}")


if __name__ == "__main__":
    main()
