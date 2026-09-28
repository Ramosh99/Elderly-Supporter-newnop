from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_monitor.pipeline import analyze_video
from elderly_monitor.config import load_config
from elderly_monitor.environment import load_project_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze elderly activity in a fixed-camera video")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("output/result.json"))
    parser.add_argument("--annotated-video", type=Path, help="optional MP4 showing bed ROI, detections, and states")
    parser.add_argument("--include-observations", action="store_true")
    args = parser.parse_args()
    load_project_env()

    result = analyze_video(args.video, load_config(args.config), args.include_observations, args.annotated_video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Analysis written to {args.output}")
    if args.annotated_video:
        print(f"Annotated video written to {args.annotated_video}")
    print(f"Final state: {result['final_state']} | Decision: {result['decision']}")


if __name__ == "__main__":
    main()
