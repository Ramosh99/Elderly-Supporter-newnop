# Elderly Activity and Bed-Exit Monitor

An assignment prototype combining pretrained YOLO pose estimation, bed segmentation,
temporal tracking and bounded Gemini visual review. It processes continuous recorded
video offline and produces activities, durations, bed events and monitoring decisions.
Live streaming is not implemented.

## Submission evidence

See [deliverables](deliverables/README.md), [architecture](deliverables/ARCHITECTURE.md),
[evaluation](deliverables/EVALUATION.md) and [three failure cases](deliverables/FAILURE_CASES.md).

Saved ten-clip results (160 seconds): **80.00% duration-weighted activity accuracy**,
**77.83% mean clip accuracy**. Bed-exit precision/recall: **25%/20%**. Return
precision/recall: **25%/33.3%**. These development results do not establish deployment
reliability. Review the evidence manifest and replay notes before comparing versions.

## Run instructions

Python 3.10+; locally tested with Python 3.12. Run from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Fresh checkout only; preserve an existing configuration:
Copy-Item config.example.json config.json
python monitor.py videos/standing_bed.mp4 --config config.json --output output/result.json --annotated-video output/annotated.mp4 --include-observations
python run_all.py
python -m unittest discover -s tests
```

The example config enables automatic bed segmentation but **disables Gemini**. Set
`gemini_enabled` to `true` in config.json and put `GEMINI_API_KEY=your-key-here` in
local `.env` to enable it. Never commit the key. Network/API access is required and
calls may incur charges. YOLO weights download on first use if absent.

Place videos in `videos/`. The batch runner evaluates matching reviewed files in
`evaluation/labels/` and overwrites `output/evaluation/final/`. Per-video settings are
read from `config/cameras/<video_stem>.json`. Media and model weights are not bundled
in the deliverables snapshot. Run `python tools/select_bed_region.py --help` for
optional bed/mattress/floor calibration.

Regenerate portable submission evidence without inference or API calls:

```powershell
python -m evaluation.build_deliverables
```

This re-evaluates saved predictions; it does not update predictions for changed code.
Final numeric durations, timelines and events are in `deliverables/examples/*.json`.
Confusion matrices and duration errors are in `deliverables/evaluation_metrics.json`.

## Pipeline and decisions

Frames are sampled in timestamp order, normally at 3 fps. YOLO produces the bed outline
and tracked person keypoints. Geometry, posture and motion rules infer seven states:
LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, UNKNOWN.
Bed overlap alone does not prove physical contact. Optional mattress/floor calibration
provides additional support evidence.

The review agent selects uncertain observations, activity/bed transitions and ambiguous
upright poses over an uncalibrated bed. It gathers denser frames within budgets and
optionally sends selected images to Gemini. Adjacent agreeing assessments, spatial checks
and temporal checks restrict corrections. Correct frames can be included as context;
request limits can leave gaps. Retries and caching improve availability, not accuracy.

Temporal tracking produces the final timeline and confirms bed events. NORMAL is the
default. UNKNOWN time, prolonged sitting (default 120 seconds), or an unreturned exit
cause MONITOR. Confirmed prolonged absence (default 300 seconds) causes ALERT. These
are configurable assignment rules, not clinically validated thresholds. Short clips
do not demonstrate long-duration alert performance. Confidence is not calibrated.

**Annotated MP4 files show first-pass candidate states**, not the final Gemini-reviewed
timeline. Present them alongside final JSON; candidate overlays are not final results.

## Source map

| Location | Responsibility |
|---|---|
| `monitor.py`, `run_all.py` | Single-video CLI and batch evaluation |
| `elderly_monitor/pipeline.py` | Orchestration |
| `elderly_monitor/vision/` | Pose, bed detection, geometry, support, motion |
| `elderly_monitor/temporal/` | Smoothing, events and alert policy |
| `elderly_monitor/review/` | Window planning, resampling and Gemini |
| `elderly_monitor/video/` | Sampling and candidate overlays |
| `evaluation/evaluate.py` | Accuracy, confusion, duration and event metrics |
| `tests/` | 109 passing unit tests at wrap-up |

Controlled experiments: [thigh ratio](evaluation/thigh_ratio_findings.md) and
[transition review](evaluation/transition_review_findings.md). Their isolated replay
scores must not be substituted for full-run results.
