"""run_all.py - Full pipeline run + evaluation for all 6 clips.

Usage (from project root, with .venv active):
    python run_all.py

Outputs (all inside output/evaluation/final/):
  <clip>.json            - analysis result for each video
  <clip>_annotated.mp4   - annotated video for each clip
  evaluation_report.json - per-clip metrics (accuracy, confusion, events)
  evaluation_summary.md  - human-readable Markdown report
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# paths
ROOT        = Path(__file__).parent
VIDEOS_DIR  = ROOT / "videos"
LABELS_DIR  = ROOT / "evaluation" / "labels"
OUT_DIR     = ROOT / "output" / "evaluation" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "config.json"

# bootstrap env (loads GEMINI_API_KEY from .env if present)
from elderly_monitor.environment import load_project_env
load_project_env()

from elderly_monitor.config import load_config, resolve_config
from elderly_monitor.pipeline import analyze_video
from evaluation.evaluate import evaluate as run_evaluate

# Respect config.json, including Gemini review and dense sampling settings.
BASE_CONFIG = load_config(CONFIG_PATH)


# STEP 1 - Run pipeline on every video
def run_pipeline(videos: list) -> dict:
    """Analyze each video; return {stem: result_dict}. Failures stored as None."""
    results = {}
    for video in videos:
        stem = video.stem
        out_json  = OUT_DIR / f"{stem}.json"
        out_video = OUT_DIR / f"{stem}_annotated.mp4"
        print(f"\n{'='*60}")
        print(f"[1/2] Analyzing  : {video.name}")
        print(f"      Result JSON : {out_json.name}")
        print(f"      Annotated   : {out_video.name}")
        t0 = time.time()
        try:
            cfg = dict(BASE_CONFIG)   # fresh copy per video
            profile = ROOT / 'config' / 'cameras' / f'{stem}.json'
            if profile.exists():
                cfg = resolve_config({**cfg, **json.loads(profile.read_text(encoding='utf-8'))})
                print(f"      Camera profile: {profile.name}")
            result = analyze_video(video, cfg,
                                   include_observations=True,
                                   annotated_video_path=out_video)
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            elapsed = time.time() - t0
            print(f"      Done in {elapsed:.1f}s  |  "
                  f"decision={result['decision']}  "
                  f"exits={result['bed_exit_count']}  "
                  f"unknown={result['total_unknown_sec']:.1f}s")
            results[stem] = result
            gemini = result.get('gemini_review',{})
            if gemini.get('enabled'):
                requests = gemini.get('requests',[])
                print(f"      Gemini: {gemini.get('status')} | "
                      f"reviewed={sum(r.get('status') == 'reviewed' for r in requests)}/{len(requests)} | "
                      f"network attempts={gemini.get('network_attempts',0)} | cache hits={gemini.get('cache_hits',0)}")
        except Exception:
            elapsed = time.time() - t0
            print(f"      FAILED after {elapsed:.1f}s")
            traceback.print_exc()
            results[stem] = None
    return results


# STEP 2 - Evaluate each clip against ground-truth labels
def run_evaluation(pipeline_results: dict) -> dict:
    """Return {stem: metrics_dict}. Skips clips without a reviewed label or failed pipeline."""
    metrics = {}
    for stem, result in pipeline_results.items():
        label_path = LABELS_DIR / f"{stem}.json"
        if result is None:
            print(f"\n[2/2] SKIP  {stem}: pipeline failed")
            metrics[stem] = None
            continue
        if not label_path.exists():
            print(f"\n[2/2] SKIP  {stem}: no label file at {label_path}")
            metrics[stem] = None
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        if labels.get("reviewed") is not True:
            print(f"\n[2/2] SKIP  {stem}: labels not marked reviewed=true")
            metrics[stem] = None
            continue
        print(f"\n[2/2] Evaluating : {stem}")
        try:
            m = run_evaluate(labels, result, tolerance=1.0)
            metrics[stem] = m
            acc  = m["accuracy_time_weighted"]
            kacc = m.get("accuracy_on_known_ground_truth")
            exits = m["events"]["bed_exit"]
            rets  = m["events"]["return_to_bed"]
            print(f"      Time-weighted accuracy : {acc:.1%}")
            if kacc is not None:
                print(f"      Accuracy (known GT)    : {kacc:.1%}")
            print(f"      Bed-exit  TP/FP/FN : "
                  f"{exits['true_positives']}/{exits['false_positives']}/{exits['false_negatives']}")
            print(f"      Return    TP/FP/FN : "
                  f"{rets['true_positives']}/{rets['false_positives']}/{rets['false_negatives']}")
        except Exception as exc:
            print(f"      Evaluation error: {exc}")
            metrics[stem] = None
    return metrics


# STEP 3 - Write combined JSON report + Markdown summary
def _fmt(sec) -> str:
    if sec is None:
        return "-"
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}" if m else f"{s}s"

def _pct(v) -> str:
    return f"{v:.1%}" if v is not None else "-"

def _ev(ev: dict) -> str:
    p = _pct(ev.get("precision"))
    r = _pct(ev.get("recall"))
    return f"TP={ev['true_positives']} FP={ev['false_positives']} FN={ev['false_negatives']} P={p} R={r}"


def write_report(pipeline_results: dict, metrics: dict) -> None:
    STATES = ["LYING_IN_BED","SITTING_ON_BED","SITTING_OUTSIDE_BED",
              "STANDING","WALKING","OUT_OF_BED","UNKNOWN"]

    # combined JSON
    report = {}
    for stem in pipeline_results:
        pr = pipeline_results[stem]
        ev = metrics.get(stem)
        report[stem] = {
            "pipeline": {
                "decision":         pr["decision"] if pr else "FAILED",
                "bed_exit_count":   pr["bed_exit_count"] if pr else None,
                "bed_return_count": pr["bed_return_count"] if pr else None,
                "total_unknown_sec": pr["total_unknown_sec"] if pr else None,
                "final_state":      pr["final_state"] if pr else None,
            } if pr else {"status": "pipeline_failed"},
            "evaluation": ev,
        }
    report_path = OUT_DIR / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCombined JSON report --> {report_path}")

    # Markdown summary
    lines = [
        "# Elderly Monitor - Full Evaluation Report",
        "",
        "Generated by `run_all.py`.  "
        "Gemini review: **disabled**.  Agentic review: **enabled**.",
        "",
        "---",
        "",
        "## Per-Clip Summary",
        "",
        "| Clip | Decision | Exits | Returns | Unknown | Time-Acc | Known-Acc |",
        "|------|----------|-------|---------|---------|----------|-----------|",
    ]
    for stem in sorted(pipeline_results):
        pr = pipeline_results[stem]
        ev = metrics.get(stem)
        dec   = pr["decision"] if pr else "FAILED"
        exits = pr["bed_exit_count"] if pr else "-"
        rets  = pr["bed_return_count"] if pr else "-"
        unk   = _fmt(pr["total_unknown_sec"] if pr else None)
        tacc  = _pct(ev["accuracy_time_weighted"] if ev else None)
        kacc  = _pct(ev.get("accuracy_on_known_ground_truth") if ev else None)
        lines.append(f"| {stem} | {dec} | {exits} | {rets} | {unk} | {tacc} | {kacc} |")

    for stem in sorted(pipeline_results):
        pr = pipeline_results[stem]
        ev = metrics.get(stem)
        lines += ["", "---", "", f"## {stem}", ""]

        if pr is None:
            lines.append("**Pipeline failed - no analysis available.**")
            continue

        # activity durations
        lines += ["### Activity Durations", "",
                  "| State | Predicted |",
                  "|-------|-----------|"]
        for s in STATES:
            dur = pr.get("activity_duration_sec", {}).get(s.lower(), 0)
            lines.append(f"| {s} | {_fmt(dur)} |")

        lines += ["",
                  f"**Total in-bed:** {_fmt(pr.get('total_in_bed_sec'))}  ",
                  f"**Total out-of-bed:** {_fmt(pr.get('total_out_of_bed_sec'))}  ",
                  f"**Longest out-of-bed period:** {_fmt(pr.get('longest_out_of_bed_period_sec'))}  ",
                  f"**Decision:** `{pr['decision']}`  ", ""]

        # timeline
        lines += ["### Timeline", ""]
        for seg in pr.get("timeline", []):
            s_m, s_s = divmod(int(seg["start_sec"]), 60)
            e_m, e_s = divmod(int(seg["end_sec"]),   60)
            lines.append(f"- `{s_m:02d}:{s_s:02d}` -> `{e_m:02d}:{e_s:02d}`  {seg['state']}")

        # events
        events = pr.get("events", [])
        if events:
            lines += ["", "### Events", ""]
            for ev_item in events:
                lines.append(
                    f"- **{ev_item['event']}**  "
                    f"start={_fmt(ev_item.get('start_time_sec'))}  "
                    f"confirmed={_fmt(ev_item.get('confirmed_time_sec'))}  "
                    f"conf={ev_item.get('confidence', 0):.2f}"
                )

        # evaluation metrics
        if ev:
            lines += ["", "### Evaluation Metrics", "",
                      f"- **Time-weighted accuracy:** {_pct(ev['accuracy_time_weighted'])}",
                      f"- **Accuracy on known GT:** {_pct(ev.get('accuracy_on_known_ground_truth'))}",
                      f"- **Predicted UNKNOWN fraction:** {_pct(ev.get('predicted_unknown_fraction'))}",
                      ""]

            # duration errors
            lines += ["#### Duration Errors", "",
                      "| State | Ground Truth | Predicted | Abs Error |",
                      "|-------|-------------|-----------|-----------|"]
            for s in STATES:
                de = ev.get("duration_errors", {}).get(s, {})
                gt   = _fmt(de.get("ground_truth_sec"))
                pred = _fmt(de.get("predicted_sec"))
                err  = _fmt(de.get("absolute_error_sec"))
                lines.append(f"| {s} | {gt} | {pred} | {err} |")

            # confusion matrix
            confusion = ev.get("confusion_matrix_seconds", {})
            if confusion:
                header_states = [s[:6] for s in STATES]
                lines += ["", "#### Confusion Matrix (seconds, rows=GT cols=predicted)", ""]
                lines.append("| GT \\ Pred | " + " | ".join(header_states) + " |")
                lines.append("|" + "---|" * (len(STATES) + 1))
                for row_s in STATES:
                    row = confusion.get(row_s, {})
                    vals = " | ".join(
                        f"**{row.get(col_s, 0):.0f}**" if row_s == col_s
                        else f"{row.get(col_s, 0):.0f}"
                        for col_s in STATES
                    )
                    lines.append(f"| {row_s[:6]} | {vals} |")

            # bed events
            lines += ["", "#### Bed Events", ""]
            for kind in ("bed_exit", "return_to_bed"):
                ev_data = ev.get("events", {}).get(kind, {})
                lines.append(f"- **{kind}**: {_ev(ev_data)}")
        else:
            lines.append("*Evaluation not available for this clip.*")

    md_path = OUT_DIR / "evaluation_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown summary    --> {md_path}")


# MAIN
def main() -> None:
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {VIDEOS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} video(s): {[v.name for v in videos]}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Config: gemini_enabled={BASE_CONFIG['gemini_enabled']}, review_enabled={BASE_CONFIG['review_enabled']}")

    t_start = time.time()

    pipeline_results = run_pipeline(videos)
    metrics = run_evaluation(pipeline_results)
    write_report(pipeline_results, metrics)

    total = time.time() - t_start
    passed = sum(1 for v in pipeline_results.values() if v is not None)
    print(f"\n{'='*60}")
    print(f"Done.  {passed}/{len(videos)} clips succeeded.  Total time: {total/60:.1f} min")
    print(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
