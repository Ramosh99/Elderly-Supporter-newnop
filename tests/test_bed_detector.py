import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from elderly_monitor.bed_detector import detect_bed_region, occupancy_polygon, detect_bed_frame
from elderly_monitor.geometry import point_in_polygon
from elderly_monitor.pose_observer import PoseActivityObserver
from elderly_monitor.models import State
from elderly_monitor.pipeline import analyze_video


class BedDetectorTests(unittest.TestCase):
    def test_person_cutout_does_not_make_seated_hips_outside_bed(self):
        raw = [[0,0],[300,0],[300,300],[180,300],[180,100],[80,100],[80,300],[0,300]]
        self.assertFalse(point_in_polygon((130,150), raw))
        region = occupancy_polygon(raw)
        points = [[0,0,0] for _ in range(17)]
        for i, xy in {5:(130,50),11:(130,150),13:(210,150),15:(210,250)}.items():
            points[i] = [*xy,0.95]
        observer = PoseActivityObserver(region)
        observation = observer.observe(0,(100,20,140,260),0.9,points,1)
        self.assertEqual(observation.state,State.SITTING_ON_BED)
        shifted = [[x+400,y,c] for x,y,c in points]
        self.assertEqual(observer.observe(1,(500,20,140,260),0.9,shifted,1).state,State.SITTING_OUTSIDE_BED)

    def test_envelope_preserves_perspective_instead_of_rectangle(self):
        region = occupancy_polygon([[100,0],[200,100],[100,200],[0,100]])
        self.assertTrue(point_in_polygon((100,100),region))
        self.assertFalse(point_in_polygon((10,10),region))

    def test_frame_without_bed_returns_no_region(self):
        model = MagicMock()
        model.names = {59:'bed'}
        model.predict.return_value = [SimpleNamespace(masks=None)]
        self.assertIsNone(detect_bed_frame(model,np.zeros((100,100,3),dtype=np.uint8),{}))

    def test_consistent_masks_produce_serializable_polygon(self):
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.conf = [0.9]
        result = SimpleNamespace(boxes=boxes, masks=SimpleNamespace(
            xy=[np.array([[10,10],[80,15],[70,70],[10,60]], dtype=float)]))
        model = MagicMock()
        model.names = {59: 'bed'}
        model.predict.return_value = [result]
        capture = MagicMock()
        capture.get.return_value = 24
        capture.read.return_value = (True, np.zeros((100,100,3), dtype=np.uint8))
        with patch('ultralytics.YOLO', return_value=model), patch('cv2.VideoCapture', return_value=capture):
            polygon, details = detect_bed_region('unused.mp4', {})
        self.assertEqual(len(polygon), 4)
        self.assertEqual(details['supporting_frames'], 5)
        json.dumps(details)
        capture.release.assert_called_once()

    def test_no_detection_does_not_reuse_old_roi(self):
        model = MagicMock()
        model.names = {59: 'bed'}
        model.predict.return_value = [SimpleNamespace(masks=None)]
        capture = MagicMock()
        capture.get.return_value = 24
        capture.read.return_value = (True, np.zeros((100,100,3), dtype=np.uint8))
        with patch('ultralytics.YOLO', return_value=model), patch('cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'Could not identify'):
                detect_bed_region('unused.mp4', {'bed_polygon': [[0,0],[10,0],[10,10]]})
        capture.release.assert_called_once()

    def test_old_roi_outside_frame_is_rejected(self):
        capture = MagicMock()
        capture.get.return_value = 100
        with patch('cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'outside this video'):
                analyze_video('unused.mp4', {'bed_polygon': [[0,0],[720,0],[720,1011]]})
        capture.release.assert_called_once()
