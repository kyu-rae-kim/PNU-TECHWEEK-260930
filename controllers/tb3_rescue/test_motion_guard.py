"""Synthetic range geometry, wheel slip, and bounded recovery regressions."""
import math
import sys
import types
import unittest

import numpy as np

sys.modules.setdefault('controller', types.SimpleNamespace(Robot=object))
from motion_guard import MotionMonitor, Recovery, clearance_escape, match_scans, scan_points
from tb3_rescue import MissionController, OccupancyGrid, PoseEstimator


def room_scan(pose=(0, 0, 0), maximum=3.5):
    """Ray-cast an asymmetric room; no Webots/ground truth is used by the controller."""
    walls = [((-2, -1.6), (2.7, -1.6)), ((2.7, -1.6), (2.7, 2.3)),
             ((2.7, 2.3), (-2, 2.3)), ((-2, 2.3), (-2, -1.6)),
             ((1.1, .8), (1.1, 1.5)), ((1.1, .8), (1.8, .8))]
    x = pose[0] - .03 * math.cos(pose[2])
    y = pose[1] - .03 * math.sin(pose[2])
    ranges = []
    for i in range(360):
        heading = pose[2] + math.pi - 2 * math.pi * i / 360
        dx, dy = math.cos(heading), math.sin(heading)
        hits = []
        for (ax, ay), (bx, by) in walls:
            sx, sy = bx - ax, by - ay
            det = dx * sy - dy * sx
            if abs(det) < 1e-9:
                continue
            t = ((ax - x) * sy - (ay - y) * sx) / det
            u = ((ax - x) * dy - (ay - y) * dx) / det
            if t > .12 and 0 <= u <= 1:
                hits.append(t)
        value = min(hits, default=float('inf'))
        ranges.append(value if value < maximum else float('inf'))
    return np.asarray(ranges)


class MotionTests(unittest.TestCase):
    def test_soft_margin_allows_only_motion_away_from_front_obstacle(self):
        ranges = np.full(360, 3.)
        ranges[180] = .195  # base distance .165m
        command = clearance_escape(ranges, (0, 0, 0), [])
        self.assertIsNotNone(command)
        self.assertLess(command[0], 0)
        self.assertEqual(command[0], command[1])

    def test_clearance_escape_rejects_contact_person_and_blocked_rear(self):
        ranges = np.full(360, 3.)
        ranges[180] = .17  # inside hard footprint
        self.assertIsNone(clearance_escape(ranges, (0, 0, 0), []))
        ranges[180] = .195
        person = dict(x=-.5, y=0, vx=0, vy=0)
        self.assertIsNone(clearance_escape(ranges, (0, 0, 0), [person]))
        ranges[0] = .14
        self.assertIsNone(clearance_escape(ranges, (0, 0, 0), []))
        self.assertIsNone(clearance_escape([float('nan')] * 360, (0, 0, 0), []))

    def test_scan_motion_recovers_translation_and_rotation(self):
        expected = (.08, -.03, .09)
        result = match_scans(scan_points(room_scan(), 3.5), scan_points(room_scan(expected), 3.5))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['delta'], expected, atol=.006)

    def test_stationary_scan_overrules_spinning_wheels(self):
        estimator = PoseEstimator()
        estimator.update_odometry(0, 0)
        estimator.update_scan_motion(room_scan(), 3.5, 0, 0)
        for step in range(1, 6):
            wheel_angle = step * .08 / .033
            estimator.update_odometry(wheel_angle, wheel_angle)
            estimator.update_scan_motion(room_scan(), 3.5, step * .5, step * .08)
        self.assertAlmostEqual(estimator.odometry_pose[0], .4)
        self.assertLess(math.hypot(*estimator.pose[:2]), .005)
        self.assertTrue(estimator.localization_valid)

    def test_parallel_walls_are_not_trusted_as_stationary_evidence(self):
        x = np.linspace(-2, 2, 60)
        pts = np.vstack((np.column_stack((x, np.ones_like(x))),
                         np.column_stack((x, -np.ones_like(x)))))
        normals = np.tile([0., 1.], (len(pts), 1))
        self.assertIsNone(match_scans((pts, normals), (pts, normals)))
        self.assertIsNone(match_scans(scan_points([float('nan')] * 360, 3.5),
                                     scan_points(room_scan(), 3.5)))

    def test_monitor_requires_sustained_independent_disagreement(self):
        monitor = MotionMonitor()
        self.assertFalse(monitor.update(.5, .5, .08, 0, .08, True))
        for now in (1, 1.5):
            self.assertFalse(monitor.update(now, .5, .08, .001, .08, True))
        self.assertTrue(monitor.update(2, .5, .08, .001, .08, True))

    def test_normal_motion_turns_and_invalid_scans_do_not_trigger_slip(self):
        for odom, localized, commanded, valid in ((.08, .079, .08, True),
                                                 (0, .003, 0, True), (.08, 0, .08, False)):
            monitor = MotionMonitor()
            for now in np.arange(.5, 5, .5):
                self.assertFalse(monitor.update(now, .5, odom, localized, commanded, valid))

    def test_motor_stall_is_detected_even_without_encoder_motion(self):
        monitor = MotionMonitor()
        for now in (.5, 1, 1.5):
            monitor.update(now, .5, 0, 0, .08, True)
        self.assertTrue(monitor.update(2, .5, 0, 0, .08, True))

    def test_backup_and_turn_return_to_planning(self):
        recovery = Recovery()
        ranges = [3.] * 360
        self.assertTrue(recovery.start((0, 0, 0), 0))
        self.assertEqual(recovery.command((0, 0, 0), ranges, [], .1), (0, 0))
        left, right = recovery.command((0, 0, 0), ranges, [], .4)
        self.assertLess(left, 0)
        self.assertEqual(left, right)
        left, right = recovery.command((-.18, 0, 0), ranges, [], 3.1)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        recovery.command((-.18, 0, .45), ranges, [], 3.9)
        self.assertFalse(recovery.active)
        self.assertFalse(recovery.start((-.18, 0, .45), 4))

    def test_blocked_rear_and_person_prevent_blind_reverse(self):
        for person in (False, True):
            recovery = Recovery()
            recovery.start((0, 0, 0), 0)
            ranges = np.full(360, 3.)
            people = []
            if person:
                people = [dict(x=-.65, y=0, vx=.2, vy=0)]
            else:
                ranges[0] = .20
            self.assertEqual(recovery.command((0, 0, 0), ranges, people, .4), (0, 0))
            self.assertEqual(recovery.stage, 'BLOCKED')

    def test_failed_backup_does_not_loop_forever(self):
        recovery = Recovery()
        recovery.start((0, 0, 0), 0)
        recovery.command((0, 0, 0), [3.] * 360, [], .4)
        self.assertEqual(recovery.command((0, 0, 0), [3.] * 360, [], 3.5), (0, 0))
        self.assertEqual(recovery.last_reason, 'BACKUP_NO_PROGRESS')

    def test_repeated_wedges_are_bounded(self):
        recovery = Recovery()
        for attempt in range(4):
            recovery.stage = None
            recovery.start((0, 0, 0), 10 * attempt)
        self.assertEqual(recovery.stage, 'BLOCKED')
        self.assertEqual(recovery.command((0, 0, 0), [3.] * 360, [], 31), (0, 0))

    def test_failed_surface_is_temporary_and_does_not_rewrite_map(self):
        mission = MissionController.__new__(MissionController)
        mission.grid = OccupancyGrid(size=60, resolution=.1)
        mission.grid.log_odds[:] = -5
        mission.recovery = Recovery()
        mission.recovery.failed_areas = [(.5, 0, 120)]
        mission.robot = types.SimpleNamespace(getTime=lambda: 0)
        gx, gy = mission.grid.world_to_grid(.5, 0)
        self.assertTrue(mission.navigation_obstacles()[gy, gx])
        self.assertEqual(mission.grid.log_odds[gy, gx], -5)
        mission.robot.getTime = lambda: 121
        self.assertFalse(mission.navigation_obstacles()[gy, gx])

    def test_recovery_progress_cannot_come_from_unverified_encoders(self):
        estimator = PoseEstimator()
        estimator.update_odometry(0, 0)
        estimator.update_scan_motion(room_scan(), 3.5, 0, 0)
        estimator.update_odometry(0, 0)
        estimator.update_scan_motion(room_scan(), 3.5, .5, 0)
        recovery = Recovery()
        recovery.start(estimator.localized_pose, .5)
        recovery.command(estimator.localized_pose, [3.] * 360, [], .9)
        estimator.update_odometry(-.2 / .033, -.2 / .033)
        self.assertLess(estimator.pose[0], -.19)
        recovery.command(estimator.localized_pose, [3.] * 360, [], 1.3)
        self.assertEqual(recovery.stage, 'BACKUP')


if __name__ == '__main__':
    unittest.main()
