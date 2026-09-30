"""Check red filtering and multi-frame target confirmation without Webots."""

import math
import sys
import types
import unittest

import numpy as np
import cv2

sys.modules.setdefault("controller", types.SimpleNamespace(Robot=object))
from tb3_rescue import RedAppleDetector


class FakeCamera:
    def getWidth(self):
        return 160

    def getHeight(self):
        return 120

    def getFov(self):
        return math.pi / 3


class FakeBox:
    def __init__(self, coords, label=47):
        self.xyxy = [types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(tolist=lambda: coords))]
        self.conf = [types.SimpleNamespace(cpu=lambda: 0.85)]
        self.cls = [types.SimpleNamespace(cpu=lambda: label)]


class FakeModel:
    def predict(self, **_):
        return [types.SimpleNamespace(boxes=[FakeBox((30, 70, 50, 90)),
                                                  FakeBox((60, 70, 80, 90)),
                                                  FakeBox((90, 70, 110, 90))])]


class AppleDetectionTests(unittest.TestCase):
    def test_only_red_yolo_box_is_confirmed(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.camera = FakeCamera()
        detector.width, detector.height = 160, 120
        detector.focal = 160 / (2 * math.tan(math.pi / 6))
        detector.model = FakeModel()
        detector.last_inference_time = -10
        detector.tracks = []
        detector.boxes = []
        bgra = np.zeros((120, 160, 4), np.uint8)
        bgra[70:90, 30:50, :3] = (0, 0, 255)  # red in BGRA
        bgra[70:90, 60:80, :3] = (51, 204, 51)  # green in BGRA
        bgra[70:90, 90:110, :3] = (0, 185, 255)  # orange
        for time in (0.0, 0.3, 0.6):
            observations = detector.detect(bgra.tobytes(), (0, 0, 0), time)
            self.assertEqual(len(observations), 1)
            detector.update_tracks(observations, time)
        self.assertEqual(len(detector.confirmed), 1)
        self.assertEqual(len(detector.visited), 1)
        self.assertEqual(detector.boxes[0][-1], "YOLO")
        self.assertEqual(detector.boxes[0][:4], (30, 70, 50, 90))

    def test_red_blob_never_confirms_without_semantic_evidence(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.camera = FakeCamera()
        detector.width, detector.height = 160, 120
        detector.focal = 160 / (2 * math.tan(math.pi / 6))
        detector.model = types.SimpleNamespace(
            predict=lambda **_: [types.SimpleNamespace(boxes=[])])
        detector.last_inference_time = -10
        detector.tracks = []
        detector.boxes = []
        bgra = np.zeros((120, 160, 4), np.uint8)
        cv2.circle(bgra, (80, 85), 8, (0, 0, 255, 255), -1)
        for frame in range(20):
            now = frame * 0.3
            observations = detector.detect(bgra.tobytes(), (0, 0, 0), now)
            self.assertEqual(len(observations), 1)
            detector.update_tracks(observations, now)
            self.assertEqual(len(detector.confirmed), 0)
        self.assertEqual(detector.boxes[0][-1], "color")

    def test_red_cup_and_person_are_excluded_even_with_apple_box(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.width, detector.height = 160, 120
        detector.focal = 160 / (2 * math.tan(math.pi / 6))
        detector.last_inference_time = -10
        detector.model = types.SimpleNamespace(predict=lambda **_: [types.SimpleNamespace(
            boxes=[FakeBox((30, 70, 50, 90)), FakeBox((30, 65, 55, 95), 41),
                   FakeBox((90, 50, 130, 110), 0)])])
        bgra = np.zeros((120, 160, 4), np.uint8)
        bgra[70:90, 30:50, :3] = (0, 0, 255)
        cv2.circle(bgra, (110, 85), 8, (0, 0, 255, 255), -1)
        self.assertEqual(detector.detect(bgra.tobytes(), (0, 0, 0), 0), [])
        self.assertEqual(len(detector.people_boxes), 1)

    def test_sparse_sightings_do_not_accumulate_confirmation(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.tracks = []
        obs = (1, 0, .9, 0, 1, 80, 85, 8, .9, .8, 'YOLO')
        for now in (0, 2, 4, 6):
            detector.update_tracks([obs], now)
        self.assertFalse(detector.confirmed)

    def test_distant_apple_is_an_approach_goal_then_both_counts_increment(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.tracks = []
        def observe(distance, now, source='YOLO'):
            return detector.update_tracks([(2, 0, .9, 0, distance, 80, 85, 8, .9, .9, source)], now)
        for now in (0, .3, .6):
            observe(1.5, now)
        self.assertEqual(len(detector.identified), 1)
        self.assertEqual(len(detector.confirmed), 0)
        self.assertEqual(len(detector.visited), 0)
        observe(.7, .9, 'color')
        self.assertEqual(len(detector.visited), 0)
        self.assertEqual(len(observe(.7, 1.2)), 1)
        self.assertEqual(len(detector.confirmed), 1)
        self.assertEqual(len(detector.visited), 1)
        self.assertEqual(observe(.6, 1.5), [])
        self.assertEqual(len(detector.visited), 1)


if __name__ == "__main__":
    unittest.main()
