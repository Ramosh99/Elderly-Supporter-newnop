# Elderly Activity and Bed-Exit Monitor

An Agentic AI + Vision system that analyses fixed-camera indoor video of an elderly person and determines:

1. What the person is doing over time (activity timeline)
2. Whether they leave or return to bed (bed-exit/return events)
3. How much time they spend in each activity state
4. Whether the recording warrants monitoring or an alert

---

## System Architecture

```
VIDEO
  │
  ├─── Scene understanding          Person understanding
  │         │                              │
  │   Bed segmentation              YOLO Pose + ByteTrack
  │         │                              │
  └──────────────────┬───────────────────┘
                     │
            Structured evidence
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        Pose       Spatial    Motion
       evidence   evidence   evidence
          │          │          │
          └──────────┼──────────┘
                     │
             VLM (Gemini) when uncertain
             — including low-light / IR frames
                     │
             Evidence Fusion
                     │
          Temporal State Model
                     │
           Event State Machine
                     │
          BED_EXIT / RETURN
                     │
          NORMAL / MONITOR / ALERT
```

---

## Recognised States

| State | Description |
|---|---|
| `LYING_IN_BED` | Person lying horizontally on the bed |
| `SITTING_ON_BED` | Person seated on the bed surface |
| `SITTING_OUTSIDE_BED` | Person seated away from the bed |
| `STANDING` | Person upright, not walking |
| `WALKING` | Person moving (sustained hip + ankle displacement) |
| `OUT_OF_BED` | Person confirmed away, activity unclear |
| `UNKNOWN` | Insufficient evidence to classify |

---

## Project Structure

```
monitor.py                  CLI entry point — run a single video
run_all.py                  Batch evaluation across all videos
config.json                 Active configuration
config.example.json         Configuration template
requirements.txt
.env                        API keys (not committed)

elderly_monitor/
  pipeline.py               Orchestrates all stages
  config.py                 Defaults, loading and validation
  models.py                 Observation, State, TimelineSegment, Event
  environment.py            .env loader
  vision/
    bed_detector.py         YOLO segmentation → bed polygon
    pose_detector.py        YOLO Pose + ByteTrack identity lock
    pose_observer.py        Keypoint geometry → activity evidence
    geometry.py             Polygon / overlap utilities
    surface.py              Calibrated mattress/floor support evidence
  temporal/
    tracker.py              Offline temporal smoothing and summary
    events.py               Bed-exit / return confirmation
    alerts.py               NORMAL / MONITOR / ALERT decision
    states.py               In-bed / out-of-bed state groups
    support_gaps.py         Short posture gap bridging
  review/
    agent.py                Plans context windows for uncertain regions
    evidence.py             Re-samples frames, validates identity
    gemini.py               Gemini VLM visual review with strict gates
    chronology.py           Chronological motion history reclassification
    sequence.py             Bracketed sequence corrections
  video/
    reader.py               Frame sampling
    annotations.py          Skeleton / bed / state overlay drawing

evaluation/
  evaluate.py               Metrics: accuracy, confusion matrix, event P/R
  labels/                   Ground-truth label files (10 clips)

tests/                      102 unit tests
tools/
  select_bed_region.py      Interactive GUI for manual bed calibration
videos/                     Test video clips
```

---

## Quick Start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Copy config template
cp config.example.json config.json

# Run on a single video
python monitor.py videos/standing_bed.mp4 --output output/result.json --annotated-video output/annotated.mp4

# Run full evaluation across all videos
python run_all.py

# Run tests
python -m pytest tests/
```

---

## Configuration

Key settings in `config.json`:

| Setting | Default | Description |
|---|---|---|
| `bed_region_mode` | `auto` | `auto` uses YOLO segmentation; `manual` uses `bed_polygon` |
| `gemini_enabled` | `true` | Enable Gemini VLM review |
| `gemini_model` | `gemini-3.1-flash-lite` | Gemini model to use |
| `sample_fps` | `3.0` | Frames per second to sample |
| `state_hold_sec` | `1.5` | Minimum duration for activity state to commit |
| `alert_after_out_of_bed_sec` | `300` | Seconds away before ALERT |

### Gemini API Key

Create a key at [Google AI Studio](https://aistudio.google.com/apikey) and add to `.env`:

```
GEMINI_API_KEY=your-key-here
```

---

## Output Format

```json
{
  "observation_duration_sec": 40.0,
  "activity_duration_sec": {
    "lying_in_bed": "26s",
    "sitting_on_bed": "12s",
    "standing": "0s",
    "walking": "0s",
    "unknown": "1s"
  },
  "bed_exit_count": 1,
  "bed_return_count": 0,
  "final_state": "WALKING",
  "decision": "MONITOR",
  "decision_reasons": ["bed_exit_without_confirmed_return"],
  "timeline": [
    { "start_sec": 0.0, "end_sec": 21.75, "state": "LYING_IN_BED", "confidence": 0.8 },
    { "start_sec": 21.75, "end_sec": 26.75, "state": "SITTING_ON_BED", "confidence": 0.75 }
  ],
  "events": [
    { "event": "bed_exit", "start_time_sec": 5.6, "confirmed_time_sec": 7.1, "confidence": 0.75 }
  ]
}
```

---

## Evaluation Results

Evaluated on 10 video clips with human-reviewed ground-truth labels. Gemini VLM review enabled.

| Clip | Accuracy | Notes |
|---|---|---|
| `sleeping_turn_aruond` | **100%** | Person rolling in bed throughout |
| `sleep_sit` | **95.9%** | Repeated lying/sitting transitions |
| `night_view` | **90.9%** | Night vision camera |
| `standing_bed` | **87.5%** | Bed exit with walking, event detection 100% P/R |
| `walking` | **76.7%** | Walking + sitting outside bed |
| `night_time` | **74.0%** | Low-light — VLM rescued from 23.3% (no pose keypoints) |
| `japan_cctv` | **70.8%** | Caregiver overlap during transitions |
| `granny` | **68.6%** | Multiple bed exits and returns |
| `chair_sitting` | **59.0%** | Video starts mid-rise, no in-bed baseline |
| `UV_camera` | **54.9%** | Infrared/UV camera, YOLO not trained on this modality |

**Average accuracy: 77.8%**

### VLM Impact

Gemini visual review improved `night_time` from **23.3% → 74.0%** (+50.7%) by classifying frames where YOLO found no pose keypoints. This demonstrates the core VLM-as-fallback design: when pose detection fails due to lighting or camera modality, Gemini reads the scene directly from raw pixels.

### Known Failure Cases

**`chair_sitting` (59%)** — Video starts with person mid-rise. No in-bed baseline means the system cannot confirm a bed exit. The cold-start UNKNOWN block (first ~3s) is expected behaviour, not a misclassification.

**`UV_camera` (54.9%)** — Infrared/UV camera footage. YOLO pose was not trained on this spectrum; keypoints are unreliable. Gemini partially compensates but spatial grounding is absent.

**`japan_cctv` (70.8%)** — A caregiver overlaps with the patient during transitions, mixing keypoints from two people. The single-target lock loses the patient briefly. This is a fundamental constraint of single-person tracking.

---

## Design Decisions

**Why temporal smoothing rather than per-frame classification?**
Single frames are noisy — a momentary pause classifies as STANDING mid-walk. The temporal state model requires sustained evidence before committing, which matches clinical monitoring needs (brief stands don't count as bed exits).

**Why VLM only for uncertain frames?**
Gemini is slow and costly. Restricting it to UNKNOWN and low-confidence frames keeps API usage bounded (≤6 requests per video) while targeting the frames that actually need visual reasoning.

**Why strict bracketing gates on Gemini corrections?**
Without gates, Gemini can introduce isolated wrong labels (e.g., calling a blurry standing frame LYING_IN_BED). The bracketing rule requires two nearby assessments to agree and no spatial conflicts in between, preventing isolated substitutions.
