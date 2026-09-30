"""Behavioral regressions for navigation, people, and mission transitions."""
import math
import sys
import types
import unittest

import numpy as np

sys.modules.setdefault('controller', types.SimpleNamespace(Robot=object))
from tb3_rescue import MissionController, OccupancyGrid, PeopleTracker, RedAppleDetector


class SafetyTests(unittest.TestCase):
    def mission(self):
        mission = MissionController.__new__(MissionController)
        mission.grid = OccupancyGrid(size=60, resolution=.1)
        mission.grid.log_odds[:] = -5
        mission.people = PeopleTracker()
        mission.apple_detector = RedAppleDetector.__new__(RedAppleDetector)
        mission.apple_detector.tracks = []
        mission.state = mission.EXPLORE
        mission.path = []
        return mission

    def test_smoothing_does_not_cross_unknown_space(self):
        grid = self.mission().grid
        grid.log_odds[20:40, 25:35] = 0
        path = grid.astar((15, 30), (45, 30))
        self.assertTrue(path)
        smoothed = grid.smooth_path(path, grid.occupied_inflated())
        self.assertGreater(len(smoothed), 2)
        self.assertFalse(grid.line_is_free((15, 30), (45, 30), grid.occupied_inflated()))

    def test_return_does_not_erase_obstacles(self):
        mission = self.mission()
        home = mission.grid.world_to_grid(0, 0)
        mission.grid.log_odds[home[1], home[0]] = 12
        before = mission.grid.log_odds.copy()
        self.assertEqual(mission.choose_home_path((1, 0, 0)), [])
        np.testing.assert_array_equal(before, mission.grid.log_odds)

    def test_infinite_scan_clears_free_space(self):
        grid = OccupancyGrid(size=80, resolution=.1)
        for _ in range(4):
            grid.integrate_scan((0, 0, 0), [float('inf')] * 360, 2)
        gx, gy = grid.world_to_grid(1, 0)
        self.assertLessEqual(grid.log_odds[gy, gx], grid.FREE_LIMIT)
        self.assertFalse(np.any(grid.log_odds >= grid.OCCUPIED_LIMIT))

    def test_person_hit_does_not_become_permanent_wall(self):
        grid = OccupancyGrid(size=80, resolution=.1)
        ranges = np.full(360, np.nan)
        ranges[180] = 1
        mask = np.zeros(360, bool)
        mask[180] = True
        for _ in range(8):
            grid.integrate_scan((0, 0, 0), ranges, 3, dynamic_mask=mask)
        gx, gy = grid.world_to_grid(1, 0)
        self.assertEqual(grid.log_odds[gy, gx], 0)

    def test_person_velocity_and_expiry(self):
        people = PeopleTracker()
        ranges = np.full(360, 1.5)
        mask = people.update([(70, 0, 90, 100)], 100, 160, (0, 0, 0), ranges, 0)
        self.assertTrue(mask[180])
        people.update([(70, 0, 90, 100)], 100, 160, (.1, 0, 0), ranges, .5)
        self.assertEqual(len(people.tracks), 1)
        self.assertAlmostEqual(people.tracks[0]['vx'], .2)
        people.update([], 100, 160, (0, 0, 0), ranges, 1.4)
        self.assertFalse(people.tracks)

    def test_blocked_local_controller_stops(self):
        mission = self.mission()
        self.assertEqual(mission.local_control((0, 0, 0), (1, 0), [.15] * 360), (0, 0))
        self.assertTrue(mission.emergency_required([float('nan')] * 360, (0, 0, 0)))

    def test_crossing_person_stops_forward_motion(self):
        mission = self.mission()
        mission.people.tracks = [dict(x=.65, y=.45, vx=0, vy=-.7)]
        left, right = mission.local_control((0, 0, 0), (2, 0), [4] * 360)
        self.assertLessEqual(left + right, 0)

    def apple(self, x, now=1, source='YOLO'):
        return dict(x=x, y=0, confirmed=True, visited=False, last_yolo_seen=now,
                    observation=dict(source=source, distance_m=.4))

    def test_two_nearby_confirmations_increment_both_counts_then_return(self):
        mission = self.mission()
        detector = mission.apple_detector
        for x, start in ((1, 0), (3, 2)):
            for offset in (0, .3, .6):
                obs = (x, 0, .9, 0, .7, 80, 85, 8, .9, .9, 'YOLO')
                detector.update_tracks([obs], start + offset)
            self.assertEqual(len(detector.confirmed), len(detector.visited))
        mission.update_mission((2.3, 0, 0), 2.6)
        self.assertEqual(len(detector.visited), 2)
        self.assertEqual(mission.state, mission.RETURN)
        mission.update_mission((.1, 0, 0), 3)
        self.assertEqual(mission.state, mission.COMPLETE)

    def test_duplicate_and_color_observations_do_not_increment_counts(self):
        mission = self.mission()
        detector = mission.apple_detector
        for x, source, start in ((1, 'YOLO', 0), (1.6, 'YOLO', 2), (3, 'color', 4)):
            for offset in (0, .3, .6):
                obs = (x, 0, .9, 0, .6, 80, 85, 8, .9, .9, source)
                detector.update_tracks([obs], start + offset)
        mission.update_mission((0, 0, 0), 5)
        self.assertEqual(len(detector.confirmed), 1)
        self.assertEqual(len(detector.visited), 1)
        self.assertEqual(mission.state, mission.EXPLORE)

    def test_emergency_stop_bypasses_acceleration_limit(self):
        mission = self.mission()
        mission.timestep = 32
        mission.last_wheels = [6, 6]
        velocities = []
        mission.left_motor = mission.right_motor = types.SimpleNamespace(setVelocity=velocities.append)
        mission.set_wheels(0, 0, emergency=True)
        self.assertEqual(velocities, [0, 0])

    def test_known_room_still_gets_camera_exploration_goal(self):
        mission = self.mission()
        mission.camera_seen = np.ones_like(mission.grid.log_odds, dtype=bool)
        mission.camera_seen[20:35, 40:55] = False
        mission.visit_counts = np.zeros_like(mission.grid.log_odds, dtype=np.uint16)
        mission.frontier_goal = None
        mission.rejected_frontiers = []
        path = mission.choose_frontier_path((0, 0, 0), 0)
        self.assertTrue(path)
        self.assertGreater(mission.frontier_goal[0], mission.grid.origin)

    def test_camera_coverage_does_not_see_through_wall(self):
        mission = self.mission()
        mission.camera_seen = np.zeros_like(mission.grid.log_odds, dtype=bool)
        mission.camera = types.SimpleNamespace(getFov=lambda: math.pi / 3)
        mission.grid.log_odds[:, 35] = 12
        mission.record_camera_coverage((0, 0, 0))
        self.assertTrue(mission.camera_seen[30, 33])
        self.assertFalse(np.any(mission.camera_seen[:, 36:]))

    def test_unreachable_nearest_apple_does_not_hide_reachable_one(self):
        mission = self.mission()
        a, b = self.apple(1), self.apple(3)
        mission.apple_detector.tracks = [a, b]
        mission.frontier_goal = None
        mission.no_frontier_count = 0
        mission.choose_apple_path = lambda pose, apple: [] if apple is a else [(30, 30), (40, 30)]
        mission.plan((0, 0, 0), 1)
        self.assertIs(mission.apple_goal, b)
        self.assertTrue(mission.path)

    def test_red_extinguisher_never_interrupts_exploration(self):
        mission = self.mission()
        mission.frontier_goal = None
        mission.no_frontier_count = 0
        candidate = dict(x=.15, y=.65, confirmed=False, hits=20, yolo_hits=0, last_seen=0)
        mission.apple_detector.tracks = [candidate]
        mission.choose_apple_path = lambda *args, **kwargs: self.fail('Color-only object selected for approach')
        mission.choose_frontier_path = lambda *args: [(30, 30), (40, 30)]
        for now in (0, 4, 8, 20, 100):
            candidate['last_seen'] = now
            mission.plan((0, 0, 1.16), now)
            self.assertIsNone(mission.inspection_goal)
            self.assertEqual(mission.path[-1], (40, 30))

    def test_unconfirmed_apple_inspection_has_short_deadline(self):
        mission = self.mission()
        mission.frontier_goal = None
        mission.no_frontier_count = 0
        candidate = dict(x=.5, y=0, confirmed=False, hits=3, yolo_hits=1, last_seen=0)
        mission.apple_detector.tracks = [candidate]
        mission.choose_apple_path = lambda *args, **kwargs: [(30, 30)]
        mission.choose_frontier_path = lambda *args: [(30, 30), (40, 30)]
        mission.plan((0, 0, 0), 0)
        self.assertIs(mission.inspection_goal, candidate)
        candidate['last_seen'] = 7
        mission.plan((0, 0, 0), 7)
        self.assertIsNone(mission.inspection_goal)
        self.assertGreater(candidate['retry_after'], 7)
        self.assertEqual(mission.path[-1], (40, 30))

    def test_stationary_startup_survey_cannot_hold_navigation_forever(self):
        mission = self.mission()
        mission.survey_remaining = 2 * math.pi
        mission.survey_heading = mission.survey_started = None
        self.assertIsNotNone(mission.survey_control((0, 0, 1.16), 0))
        self.assertIsNone(mission.survey_control((0, 0, 1.16), 13))
        self.assertEqual(mission.survey_remaining, 0)

    def test_safe_turn_allowed_when_wall_inside_translation_margin(self):
        mission = self.mission()
        ranges = np.full(360, 4.)
        ranges[0] = .18  # rear point is .21m from base centre (LiDAR offset)
        left, right = mission.local_control((0, 0, 0), (0, 1), ranges)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        self.assertAlmostEqual(left + right, 0)

    def survey_mission(self):
        mission = self.mission()
        mission.camera_seen = np.zeros_like(mission.grid.log_odds, dtype=bool)
        mission.survey_remaining = 0
        mission.last_survey_time = 10
        mission.last_survey_position = (0, 0)
        mission.frontier_goal = (40, 30)
        mission.rejected_frontiers = []
        return mission

    def test_nearby_or_recent_frontier_does_not_start_survey(self):
        mission = self.survey_mission()
        mission.reject_frontier(20, 'reached', (2.1, 0, 0))
        self.assertEqual(mission.survey_remaining, 0)
        mission.reject_frontier(60, 'reached', (.6, 0, 0))
        self.assertEqual(mission.survey_remaining, 0)

    def test_camera_observed_room_does_not_start_survey(self):
        mission = self.survey_mission()
        mission.camera_seen[:] = True
        self.assertFalse(mission.request_survey((2.1, 0, 0), 60))

    def test_survey_is_allowed_after_22_seconds_and_one_metre(self):
        mission = self.survey_mission()
        self.assertTrue(mission.request_survey((1.1, 0, 0), 33))

    def test_existing_camera_detects_person_without_counting_an_apple(self):
        mission = self.mission()
        detector = mission.apple_detector
        detector.width, detector.height = 160, 120
        detector.focal = 160 / (2 * math.tan(math.pi / 6))
        detector.last_inference_time = -10
        calls = []
        def predict(**kwargs):
            calls.append(kwargs['classes'])
            box = types.SimpleNamespace(
                cls=[types.SimpleNamespace(cpu=lambda: 0)],
                xyxy=[types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
                    tolist=lambda: [60, 10, 100, 115]))])
            return [types.SimpleNamespace(boxes=[box])]
        detector.model = types.SimpleNamespace(predict=predict)
        observations = detector.detect(bytes(160 * 120 * 4), (0, 0, 0), 1)
        mask = mission.people.update(detector.people_boxes, detector.focal,
                                     detector.width, (0, 0, 0), [1.] * 360, 1)
        self.assertIn(0, calls[0])
        self.assertIn(47, calls[0])
        self.assertEqual(observations, [])
        self.assertEqual(len(mission.people.tracks), 1)
        self.assertTrue(mask[180])

    def test_emergency_distance_uses_robot_centre_not_lidar_origin(self):
        mission = self.mission()
        ranges = np.full(360, 3.)
        ranges[0] = .16  # rear: .19m from robot, not an emergency
        self.assertFalse(mission.emergency_required(ranges, (0, 0, 0)))
        ranges[0] = 3.
        ranges[180] = .20  # front: .17m from robot, must stop
        self.assertTrue(mission.emergency_required(ranges, (0, 0, 0)))

    def test_close_obstacle_escape_is_delayed_and_limited(self):
        mission = self.mission()
        mission.recovery = types.SimpleNamespace(active=False)
        mission.estimator = types.SimpleNamespace(localization_valid=True)
        mission.control_reason = 'OBSTACLE_TOO_CLOSE'
        mission.close_stop_since = mission.escape_started = None
        mission.escape_origin = mission.escape_anchor = None
        mission.escape_attempts = 0
        ranges = np.full(360, 3.)
        ranges[180] = .195
        self.assertIsNone(mission.proximity_escape((0, 0, 0), ranges, 0))
        self.assertIsNotNone(mission.proximity_escape((0, 0, 0), ranges, 1.1))
        self.assertIsNone(mission.proximity_escape((0, 0, 0), ranges, 2.7))
        self.assertIsNotNone(mission.proximity_escape((0, 0, 0), ranges, 3.8))
        self.assertIsNone(mission.proximity_escape((0, 0, 0), ranges, 5.4))
        self.assertIsNone(mission.proximity_escape((0, 0, 0), ranges, 7))

    def test_new_unseen_room_gets_one_bounded_half_turn(self):
        mission = self.survey_mission()
        self.assertTrue(mission.request_survey((2.1, 0, 0), 60))
        self.assertAlmostEqual(mission.survey_remaining, math.pi)
        mission.survey_control((2.1, 0, 0), 60)
        self.assertFalse(mission.request_survey((2.1, 0, 0), 62))
        self.assertEqual(mission.survey_started, 60)
        self.assertIsNone(mission.survey_control((2.1, 0, 0), 67))
        self.assertEqual(mission.last_survey_time, 67)

    def test_apple_approach_preempts_survey(self):
        mission = self.survey_mission()
        mission.request_survey((2.1, 0, 0), 60)
        mission.apple_goal = self.apple(2.5)
        self.assertIsNone(mission.survey_control((2.1, 0, 0), 60))
        self.assertEqual(mission.survey_remaining, 0)


if __name__ == '__main__':
    unittest.main()
