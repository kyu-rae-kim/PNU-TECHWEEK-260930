"""Small dependency-light checks for geometry, mapping, and A* behavior."""

import math
import sys
import types
import unittest

import numpy as np

# The algorithm checks run outside Webots, where its `controller` module is absent.
sys.modules.setdefault("controller", types.SimpleNamespace(Robot=object))

from tb3_rescue import MissionController, OccupancyGrid, PoseEstimator, bresenham, wrap_angle


class CoreTests(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)

    def test_bresenham_has_endpoints(self):
        cells = list(bresenham(1, 2, 5, 4))
        self.assertEqual(cells[0], (1, 2))
        self.assertEqual(cells[-1], (5, 4))

    def test_odometry_forward(self):
        estimator = PoseEstimator()
        estimator.update_odometry(0.0, 0.0)
        estimator.update_odometry(10.0, 10.0)
        self.assertAlmostEqual(estimator.pose[0], 0.33, places=3)
        self.assertAlmostEqual(estimator.pose[1], 0.0, places=3)

    def test_mapping_and_astar_around_wall(self):
        grid = OccupancyGrid(size=80, resolution=0.1)
        grid.log_odds[:] = -5
        grid.log_odds[20:60, 40] = 10
        grid.log_odds[38:43, 40] = -5
        inflated = grid.occupied_inflated(radius_m=0.1)
        path = grid.astar((20, 40), (60, 40), inflated)
        self.assertTrue(path)
        self.assertTrue(all(not inflated[y, x] for x, y in path))

    def test_frontier_extraction(self):
        grid = OccupancyGrid(size=60, resolution=0.1)
        grid.log_odds[25:35, 25:35] = -5
        components = grid.frontier_components((30, 30), min_cells=3)
        self.assertGreaterEqual(len(components), 1)

    def test_frontier_returns_multiple_parts_of_large_boundary(self):
        grid = OccupancyGrid(size=80, resolution=0.1)
        grid.log_odds[15:65, 15:65] = -5
        candidates = grid.frontier_components((40, 40), min_cells=3)
        self.assertGreater(len(candidates), 4)
        self.assertTrue(any(x < 25 for x, _, _, _ in candidates))
        self.assertTrue(any(x > 55 for x, _, _, _ in candidates))

    def test_astar_does_not_cut_diagonally_through_corners(self):
        grid = OccupancyGrid(size=20, resolution=0.1)
        grid.log_odds[:] = -5
        inflated = np.zeros_like(grid.log_odds, dtype=bool)
        inflated[5, 6] = True
        inflated[6, 5] = True
        path = grid.astar((5, 5), (6, 6), inflated)
        self.assertTrue(path)
        self.assertNotEqual(path[1], (6, 6))
        self.assertFalse(grid.line_is_free((5, 5), (6, 6), inflated))

    def test_stalled_frontier_changes_exploration_goal(self):
        mission = MissionController.__new__(MissionController)
        mission.grid = OccupancyGrid(size=80, resolution=0.1)
        mission.grid.log_odds[15:65, 15:65] = -5
        mission.visit_counts = np.zeros_like(mission.grid.log_odds, dtype=np.uint16)
        mission.frontier_goal = None
        mission.rejected_frontiers = []
        mission.path = []
        mission.choose_frontier_path((0, 0, 0), 0.0)
        first = mission.frontier_goal
        self.assertIsNotNone(first)
        mission.reject_frontier(1.0)
        mission.choose_frontier_path((0, 0, 0), 1.0)
        self.assertIsNotNone(mission.frontier_goal)
        self.assertGreaterEqual(math.hypot(first[0] - mission.frontier_goal[0],
                                           first[1] - mission.frontier_goal[1]), 7)

    def test_no_progress_triggers_new_goal(self):
        mission = MissionController.__new__(MissionController)
        mission.grid = OccupancyGrid(size=80, resolution=0.1)
        mission.grid.log_odds[15:65, 15:65] = -5
        mission.visit_counts = np.zeros_like(mission.grid.log_odds, dtype=np.uint16)
        mission.apple_detector = types.SimpleNamespace(confirmed=[])
        mission.state = mission.EXPLORE
        mission.frontier_goal = None
        mission.rejected_frontiers = []
        mission.path = []
        mission.no_frontier_count = 0
        mission.plan((0, 0, 0), 0.0)
        first = mission.frontier_goal
        mission.plan((0, 0, 0), 26.0)
        self.assertNotEqual(mission.frontier_goal, first)


if __name__ == "__main__":
    unittest.main()
