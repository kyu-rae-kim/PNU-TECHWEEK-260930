"""Low camera obstacles must affect control without corrupting LiDAR data."""
import math
import types
import unittest

import numpy as np

from test_apple_detection import FakeBox, RedAppleDetector
from visual_obstacles import LowObstacleMap, floor_obstacle


class VisualObstacleTests(unittest.TestCase):
    def test_other_apples_containers_and_cat_are_obstacles_not_targets(self):
        detector = RedAppleDetector.__new__(RedAppleDetector)
        detector.width, detector.height = 160, 120
        detector.focal = 160 / (2 * math.tan(math.pi / 6))
        detector.last_inference_time = -10
        detector.model = types.SimpleNamespace(predict=lambda **_: [types.SimpleNamespace(
            boxes=[FakeBox((10 + i * 30, 70, 30 + i * 30, 100), label)
                   for i, label in enumerate((47, 39, 41, 15))])])
        frame = np.zeros((120, 160, 4), np.uint8)
        frame[70:100, 10:30, :3] = (51, 204, 51)
        self.assertEqual(detector.detect(frame.tobytes(), (0, 0, 0), 0), [])
        self.assertEqual([o['label'] for o in detector.low_observations],
                         ['apple', 'container', 'container', 'cat'])

    def test_projection_rejects_above_horizon_and_rotates_with_robot(self):
        self.assertIsNone(floor_obstacle((60, 10, 100, 50), 'apple', (0, 0, 0), 140, 160, 120))
        obj = floor_obstacle((60, 70, 100, 100), 'apple', (1, 2, math.pi / 2), 140, 160, 120)
        self.assertAlmostEqual(obj['x'], 1)
        self.assertAlmostEqual(obj['y'], 2 + .073 * 140 / 40 + .02)

    def test_control_detects_low_object_without_modifying_sensor_or_invalid_beams(self):
        obstacles = LowObstacleMap()
        obstacles.update([dict(x=.4, y=0, radius=.08, label='apple')], 0)
        raw = np.full(360, np.inf)
        raw[179], raw[181] = np.nan, 0
        merged = obstacles.control_ranges(raw, (0, 0, 0), 0)
        self.assertAlmostEqual(merged[180], .35)
        self.assertTrue(np.isinf(raw[180]))
        self.assertTrue(np.isnan(merged[179]))
        self.assertEqual(merged[181], 0)
        self.assertTrue(np.isinf(merged[0]))

    def test_tracks_refresh_and_expire_without_permanent_ghosts(self):
        obstacles = LowObstacleMap()
        observations = [dict(x=.5, y=0, radius=.08, label=label) for label in ('apple', 'cat')]
        obstacles.update(observations, 0)
        obstacles.update(observations, 1)
        self.assertEqual(len(obstacles.active(1)), 2)
        self.assertEqual([o['label'] for o in obstacles.active(2.6)], ['apple'])
        self.assertEqual(obstacles.active(21), [])


if __name__ == '__main__':
    unittest.main()
