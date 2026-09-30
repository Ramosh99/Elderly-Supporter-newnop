"""Compare the thigh cue on identical saved poses, without new API calls.

Run: python -m evaluation.thigh_ratio_replay
Scores are local pose + temporal tracking, not a fresh full Gemini run.
"""
import json
from pathlib import Path
from dataclasses import replace
from elderly_monitor.models import Observation, State
from elderly_monitor.config import DEFAULTS
from elderly_monitor.vision.pose_observer import PoseActivityObserver
from elderly_monitor.temporal.tracker import TemporalStateTracker
from .evaluate import evaluate


def main():
    source = Path('output/evaluation/final')
    target = Path('output/evaluation/thigh_ratio')
    target.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULTS, **json.loads(Path('config.json').read_text())}
    report = {}
    for path in sorted(source.glob('*.json')):
        saved = json.loads(path.read_text())
        raw = saved.get('observation_stages', {}).get('pre_gemini')
        label_path = Path('evaluation/labels') / path.name
        if not raw or not label_path.exists():
            continue
        labels = json.loads(label_path.read_text())
        report[path.stem] = {}
        for enabled in (False, True):
            observer = PoseActivityObserver(raw[0]['bed_polygon'], config['walking_speed_px_sec'],
                                            config['keypoint_confidence'], raw[0].get('mattress_polygon'))
            observer.foreshortened_sitting_enabled = enabled
            tracker = TemporalStateTracker(**saved['analysis']['temporal_policy'])
            observations = []
            for data in raw:
                o = Observation(**{**data, 'state': State(data['state'])})
                if o.bbox is None or o.track_id is None or not o.bed_polygon or o.detector_confidence is None:
                    observer.previous = None
                    classified = o
                elif o.detector_confidence < config['min_detection_confidence']:
                    observer.previous = None
                    classified = replace(o, state=State.UNKNOWN, confidence=0)
                else:
                    observer.bed_polygon = o.bed_polygon
                    classified = observer.observe(o.timestamp_sec, o.bbox, o.detector_confidence, o.keypoints, o.track_id)
                    classified = replace(classified, bed_polygon=o.bed_polygon)
                tracker.update(classified)
                observations.append(classified.to_dict())
            result = tracker.finish(saved['observation_duration_sec'])
            result['video'] = saved['video']
            metrics = evaluate(labels, result)
            mode = 'with_ratio' if enabled else 'baseline'
            report[path.stem][mode] = metrics
            result['observations'] = observations
            (target / f'{path.stem}_{mode}.json').write_text(json.dumps(result, indent=2))
        row = report[path.stem]
        print(path.stem, *(f'{mode}={v["accuracy_time_weighted"]:.2%}' for mode, v in row.items()))
    (target / 'comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
