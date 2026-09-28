"""Gather extra frames without rewinding or corrupting the first-pass tracker."""
from dataclasses import replace

from .agent import identity_anchor, select_identity
from ..models import Observation, State
from ..video.reader import VideoReader
from ..vision.pose_observer import PoseActivityObserver
from ..vision.bed_detector import detect_bed_frame


class VideoEvidence:
    def __init__(self, video_path, config, observations):
        self.video_path, self.config, self.observations = video_path, config, observations
        self.model = None
        self.bed_model = None
        self.existing = {round(o.timestamp_sec, 9) for o in observations}

    def __call__(self, window, budget):
        from ultralytics import YOLO
        if self.model is None:
            self.model = YOLO(self.config['yolo_model'])
        if self.config['bed_region_mode'] == 'auto' and self.config['refresh_bed_each_sample'] and self.bed_model is None:
            self.bed_model = YOLO(self.config['bed_model'])
        observer = PoseActivityObserver(self.config['bed_polygon'], self.config['walking_speed_px_sec'], self.config['keypoint_confidence'])
        extra, attempted, rejected = [], 0, 0
        with VideoReader(self.video_path, self.config['review_sample_fps']) as reader:
            for timestamp, frame in reader.frames(window['start_sec'], window['end_sec']):
                if round(timestamp, 9) in self.existing:
                    continue
                if attempted >= budget:
                    break
                attempted += 1
                anchor = identity_anchor(self.observations, timestamp, self.config['review_anchor_gap_sec'])
                if anchor is None:
                    rejected += 1
                    observer.previous = None
                    continue
                result = self.model.predict(frame, conf=self.config['min_detection_confidence'],
                                            device=self.config['device'], verbose=False)[0]
                candidates = []
                if result.keypoints is not None:
                    for box, conf, points in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.keypoints.data.cpu().tolist()):
                        x,y,right,bottom = box
                        candidates.append(dict(bbox=(int(x),int(y),int(right-x),int(bottom-y)),confidence=conf,keypoints=points))
                selected = select_identity(candidates, anchor)
                if selected is None:
                    rejected += 1
                    observer.previous = None
                    extra.append(Observation(timestamp, State.UNKNOWN, 0.0, None, reason='review_identity_uncertain'))
                    continue
                polygon = self.config['bed_polygon']
                if self.bed_model is not None:
                    bed = detect_bed_frame(self.bed_model, frame, self.config)
                    if bed is None:
                        observer.previous = None
                        extra.append(Observation(timestamp,State.UNKNOWN,0.0,selected['bbox'],reason='review_bed_uncertain'))
                        continue
                    polygon = bed[0]
                observer.bed_polygon = polygon
                o = observer.observe(timestamp,selected['bbox'],selected['confidence'],selected['keypoints'],anchor[1])
                extra.append(replace(o,bed_polygon=polygon))
        return extra, attempted, rejected
