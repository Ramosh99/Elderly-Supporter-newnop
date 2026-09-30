"""Run with python -m evaluation.evaluate --help."""
import argparse
import json
import math
from pathlib import Path

from elderly_monitor.models import State

STATES = [s.value for s in State]


def timeline(data, duration):
    rows = data.get('timeline')
    if not isinstance(rows,list) or not rows:
        raise ValueError('A complete nonempty timeline is required')
    result, previous = [], 0.0
    for row in rows:
        start, end = float(row['start_sec']), float(row['end_sec'])
        state = row['state']
        if state not in STATES or not all(math.isfinite(t) for t in (start,end)):
            raise ValueError('Invalid timeline state or time')
        if abs(start-previous) > 0.002 or end <= start or end > duration+0.002:
            raise ValueError('Timeline must cover the video without gaps or overlaps')
        result.append((previous,min(end,duration),state))
        previous=end
    if abs(previous-duration) > 0.002:
        raise ValueError('Timeline does not cover the full duration')
    return result


def event_times(data, kind, duration):
    events=data.get('events')
    if not isinstance(events,list):
        raise ValueError('events must be a list; use [] for no events')
    times=[]
    for event in events:
        if event.get('event') not in ('bed_exit','return_to_bed'):
            raise ValueError('Invalid event type')
        t=float(event['start_time_sec'])
        if not math.isfinite(t) or not 0 <= t <= duration:
            raise ValueError('Invalid event timestamp')
        if event['event']==kind:
            times.append(t)
    return sorted(times)


def match_events(truth, predicted, tolerance):
    """Maximum one-to-one matches, then minimum total start-time error.

    Ordered matching suffices for absolute time distances with a fixed tolerance.
    """
    dp=[[(0,0.0) for _ in range(len(predicted)+1)] for _ in range(len(truth)+1)]
    for i,a in enumerate(truth,1):
        for j,b in enumerate(predicted,1):
            options=[dp[i-1][j],dp[i][j-1]]
            if abs(a-b)<=tolerance:
                n,error=dp[i-1][j-1]
                options.append((n+1,error+abs(a-b)))
            dp[i][j]=max(options,key=lambda pair:(pair[0],-pair[1]))
    matched,error=dp[-1][-1]
    return {'true_positives':matched,'false_positives':len(predicted)-matched,
            'false_negatives':len(truth)-matched,
            'precision':matched/len(predicted) if predicted else None,
            'recall':matched/len(truth) if truth else None,
            'mean_start_time_error_sec':error/matched if matched else None}


def evaluate(labels, prediction, tolerance=1.0):
    if labels.get('reviewed') is not True:
        raise ValueError('Ground truth is not reviewed. Complete manual labels and set reviewed=true.')
    if labels.get('human_reviewed') is False:
        raise ValueError('AI draft labels require human review; set human_reviewed=true only after checking the video.')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Event tolerance must be finite and nonnegative')
    duration=float(labels['observation_duration_sec'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Duration must be positive and finite')
    actual=float(prediction['observation_duration_sec'])
    if not math.isfinite(actual) or abs(actual-duration)>0.01:
        raise ValueError('Prediction and label durations differ')
    # Accept slash styles from Windows or POSIX results.
    name=lambda p:str(p).replace('\\','/').rsplit('/',1)[-1]
    if not labels.get('video') or name(labels['video'])!=name(prediction.get('video','')):
        raise ValueError('Prediction and labels refer to different videos')
    truth=timeline(labels,duration); predicted=timeline(prediction,duration)
    confusion={a:{b:0.0 for b in STATES} for a in STATES}
    i=j=0
    while i<len(truth) and j<len(predicted):
        a,b,s=truth[i]; c,d,t=predicted[j]
        confusion[s][t]+=max(0.0,min(b,d)-max(a,c))
        if b<=d: i+=1
        if d<=b: j+=1
    durations={}
    for state in STATES:
        gt=sum(confusion[state].values())
        pred=sum(confusion[s][state] for s in STATES)
        durations[state]={'ground_truth_sec':gt,'predicted_sec':pred,
                          'signed_error_sec':pred-gt,'absolute_error_sec':abs(pred-gt)}
    known=duration-durations['UNKNOWN']['ground_truth_sec']
    correct=sum(confusion[s][s] for s in STATES)
    return {'video':labels['video'],'duration_sec':duration,
            'accuracy_time_weighted':correct/duration,
            'accuracy_on_known_ground_truth':sum(confusion[s][s] for s in STATES if s!='UNKNOWN')/known if known else None,
            'predicted_unknown_fraction':durations['UNKNOWN']['predicted_sec']/duration,
            'confusion_matrix_seconds':confusion,'duration_errors':durations,
            'event_tolerance_sec':tolerance,
            'events':{kind:match_events(event_times(labels,kind,duration),event_times(prediction,kind,duration),tolerance)
                      for kind in ('bed_exit','return_to_bed')}}


def create_template(video, output):
    import cv2
    capture=cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open {video}')
        fps=capture.get(cv2.CAP_PROP_FPS)
        count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps<=0 or count<=0:
            raise ValueError('Cannot determine duration')
        duration=count/fps
    finally:
        capture.release()
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents replacing manually completed labels.
    with output.open('x',encoding='utf-8') as handle:
        json.dump({'video':str(video),'observation_duration_sec':duration,'reviewed':False,
                   'annotation_notes':'Watch the source video, fill timeline and events, then set reviewed=true.',
                   'timeline':[],'events':[]},handle,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-video',type=Path)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--prediction',action='append',default=[],help='NAME=path.json; repeat to compare variants')
    parser.add_argument('--event-tolerance',type=float,default=1.0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.init_video:
        create_template(args.init_video,args.output)
    else:
        if not args.labels or not args.prediction:
            parser.error('--labels and at least one --prediction are required')
        labels=json.loads(args.labels.read_text(encoding='utf-8'))
        results={}
        for entry in args.prediction:
            name,separator,path=entry.partition('=')
            if not separator or not name or name in results:
                parser.error('Use unique NAME=path.json predictions')
            results[name]=evaluate(labels,json.loads(Path(path).read_text(encoding='utf-8')),args.event_tolerance)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'variants':results},indent=2,allow_nan=False),encoding='utf-8')
        for name,result in results.items():
            print(f"{name}: time-weighted accuracy={result['accuracy_time_weighted']:.1%}, bed exits={result['events']['bed_exit']}")
    print(f'Written to {args.output}')


if __name__=='__main__':
    main()
