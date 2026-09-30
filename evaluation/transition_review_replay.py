"""Replay saved VLM assessments to isolate the transition guard change."""
import json
from pathlib import Path
from elderly_monitor.models import Observation, State
from elderly_monitor.config import DEFAULTS
from elderly_monitor.review.sequence import apply_sequence_assessments, guard_fragmentation
from elderly_monitor.temporal.tracker import TemporalStateTracker
from .evaluate import evaluate


def main():
    target = Path('output/evaluation/transition_review')
    target.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULTS, **json.loads(Path('config.json').read_text())}
    report = {}
    for path in Path('output/evaluation/final').glob('*.json'):
        saved = json.loads(path.read_text())
        raw = saved.get('observation_stages', {}).get('pre_gemini')
        label = Path('evaluation/labels') / path.name
        if not raw or not label.exists():
            continue
        context = [Observation(**{**o, 'state':State(o['state'])}) for o in raw]
        original = {o.timestamp_sec:o for o in context}
        report[path.stem] = {}
        for mode in ('strict', 'supported_transition'):
            updated = dict(original)
            logs = []
            for req in saved.get('gemini_review', {}).get('requests', []):
                decisions = req.get('decisions', [])
                if req.get('status') != 'reviewed' or not decisions:
                    continue
                ordered = sorted(decisions, key=lambda d:d['index'])
                selected = [original[d['timestamp_sec']] for d in ordered]
                assessments = [{k:d[k] for k in ('index','state','confidence','target_clear','evidence')} for d in ordered]
                changes,_,_ = apply_sequence_assessments(selected, assessments, config['gemini_min_confidence'], context)
                accepted,info = guard_fragmentation(sorted(updated.values(),key=lambda o:o.timestamp_sec), changes,
                                                     saved['observation_duration_sec'], config)
                if mode == 'strict':
                    accepted = info['proposed_unknown_sec'] <= info['before_unknown_sec'] + .001
                if accepted:
                    updated.update(changes)
                logs.append({**info, 'accepted':accepted, 'start_sec':req['start_sec']})
            tracker = TemporalStateTracker(**saved['analysis']['temporal_policy'])
            for o in sorted(updated.values(),key=lambda o:o.timestamp_sec):
                tracker.update(o)
            result = tracker.finish(saved['observation_duration_sec'])
            result['video'] = saved['video']
            result['replayed_reviews'] = logs
            report[path.stem][mode] = evaluate(json.loads(label.read_text()),result)
            (target / f'{path.stem}_{mode}.json').write_text(json.dumps(result,indent=2))
        print(path.stem, [(m,round(v['accuracy_time_weighted']*100,2)) for m,v in report[path.stem].items()])
    (target/'comparison.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
