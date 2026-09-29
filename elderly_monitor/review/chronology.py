"""Reclassify merged keypoints once, using one chronological motion history."""
from dataclasses import replace
from ..models import State
from ..vision.pose_observer import PoseActivityObserver


def classify_merged(observations, config):
    observer = PoseActivityObserver(config['bed_polygon'], config['walking_speed_px_sec'],
                                    config['keypoint_confidence'], config.get('mattress_polygon'),
                                    config.get('floor_polygon'))
    output = []
    previous_time = -1
    for o in observations:
        if o.timestamp_sec <= previous_time:
            raise ValueError('Merged observations must have unique increasing timestamps')
        previous_time = o.timestamp_sec
        if (o.bbox is None or o.track_id is None or not o.bed_polygon
                or o.detector_confidence is None):
            observer.previous = None
            output.append(o)
            continue
        if o.detector_confidence < config['min_detection_confidence']:
            observer.previous = None
            output.append(replace(o, state=State.UNKNOWN, confidence=0,
                                  reason='low_detection_confidence'))
            continue
        observer.bed_polygon = o.bed_polygon
        classified = observer.observe(o.timestamp_sec, o.bbox, o.detector_confidence,
                                      o.keypoints, o.track_id)
        output.append(replace(classified, bed_polygon=o.bed_polygon))
    return output
