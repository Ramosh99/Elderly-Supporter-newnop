import unittest
from unittest.mock import patch

import cv2

from elderly_monitor.video.reader import VideoReader
from elderly_monitor.config import resolve_config


class FakeCapture:
    def __init__(self):
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 6, cv2.CAP_PROP_FRAME_COUNT: 12,
                cv2.CAP_PROP_FRAME_WIDTH: 100, cv2.CAP_PROP_FRAME_HEIGHT: 100}[prop]

    def set(self, prop, value):
        self.position = int(value)

    def read(self):
        if self.position >= 12:
            return False, None
        frame = self.position
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class VideoReaderTests(unittest.TestCase):
    def test_sampling_and_revisiting_segments_preserve_source_timestamps(self):
        capture = FakeCapture()
        with patch('cv2.VideoCapture', return_value=capture):
            with VideoReader('unused.mp4', sample_fps=3) as reader:
                self.assertEqual(list(reader.frames()), [(i/6,i) for i in range(0,12,2)])
                self.assertEqual(list(reader.frames(1,1.5)), [(1.0,6),(8/6,8)])
                self.assertEqual(list(reader.frames(0,0.5)), [(0.0,0),(2/6,2)])
        self.assertTrue(capture.released)

    def test_error_in_processing_releases_video(self):
        capture = FakeCapture()
        with patch('cv2.VideoCapture', return_value=capture):
            with self.assertRaises(RuntimeError):
                with VideoReader('unused.mp4'):
                    raise RuntimeError('model failed')
        self.assertTrue(capture.released)

    def test_config_resolution_does_not_mutate_caller(self):
        supplied = {'bed_region_mode':'auto', 'sample_fps':2}
        resolved = resolve_config(supplied)
        self.assertEqual(supplied, {'bed_region_mode':'auto', 'sample_fps':2})
        self.assertEqual(resolved['sample_fps'],2)
        self.assertEqual(resolved['yolo_model'],'yolo11n-pose.pt')
