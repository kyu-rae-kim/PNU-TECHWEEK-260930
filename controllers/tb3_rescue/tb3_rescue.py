"""Autonomous search-and-return mission for the TECH WEEK apartment.

Pipeline: perception -> localization/mapping -> planning -> control.

The controller intentionally does not use Supervisor, GPS, the apartment layout,
or the apples' world coordinates.  Its local SLAM frame starts at (0, 0, 0).
"""

from __future__ import annotations

import heapq
import json
import math
import os
from collections import deque

import numpy as np
from controller import Robot
from motion_guard import MotionMonitor, Recovery, clearance_escape, match_scans, scan_points
from visual_obstacles import LowObstacleMap, floor_obstacle


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def bresenham(x0, y0, x1, y1):
    """Yield all integer cells on a line, including both endpoints."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def binary_dilate(image, radius=1):
    """Small NumPy-only binary dilation (keeps the controller dependency-light)."""
    padded = np.pad(image.astype(bool), radius, mode="constant")
    result = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def binary_erode(image, radius=1):
    padded = np.pad(image.astype(bool), radius, mode="constant", constant_values=True)
    result = np.ones_like(image, dtype=bool)
    height, width = image.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy:dy + height, dx:dx + width]
    return result


def connected_components(image, connectivity=8):
    """Return components as lists of (x, y); only foreground pixels are visited."""
    foreground = image.astype(bool)
    visited = np.zeros_like(foreground, dtype=bool)
    if connectivity == 8:
        neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0),
                     (-1, 1), (0, 1), (1, 1))
    else:
        neighbors = ((0, -1), (-1, 0), (1, 0), (0, 1))
    height, width = foreground.shape
    components = []
    for y_value, x_value in np.argwhere(foreground):
        x, y = int(x_value), int(y_value)
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = [(x, y)]
        points = []
        while queue:
            cx, cy = queue.pop()
            points.append((cx, cy))
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < width and 0 <= ny < height and
                        foreground[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        components.append(points)
    return components


def draw_disc(image, center, radius, color):
    cx, cy = center
    y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    image[y0:y1, x0:x1][mask] = color


class OccupancyGrid:
    UNKNOWN = 0
    FREE_LIMIT = -3
    OCCUPIED_LIMIT = 4

    def __init__(self, size=360, resolution=0.08, lidar_offset=0.0):
        self.size = size
        self.resolution = resolution
        self.origin = size // 2
        self.lidar_offset = lidar_offset
        self.log_odds = np.zeros((size, size), dtype=np.int16)

    def world_to_grid(self, x, y):
        return (
            int(round(x / self.resolution)) + self.origin,
            self.origin - int(round(y / self.resolution)),
        )

    def grid_to_world(self, gx, gy):
        return (
            (gx - self.origin) * self.resolution,
            (self.origin - gy) * self.resolution,
        )

    def inside(self, gx, gy, margin=1):
        return margin <= gx < self.size - margin and margin <= gy < self.size - margin

    def integrate_scan(self, pose, ranges, max_range, stride=2, dynamic_mask=None):
        x, y, theta = pose
        rx, ry = self.world_to_grid(x, y)
        if self.inside(rx, ry) and self.log_odds[ry, rx] < self.OCCUPIED_LIMIT:
            self.log_odds[ry, rx] = -12
        x += self.lidar_offset * math.cos(theta)
        y += self.lidar_offset * math.sin(theta)
        sx, sy = self.world_to_grid(x, y)
        count = len(ranges)
        for index in range(0, count, stride):
            distance = float(ranges[index])
            if math.isnan(distance) or distance <= 0.03:
                continue
            hit = distance < max_range - 0.06
            distance = min(distance, max_range)
            # LDS-01 indexing in this project: n/2=front, n/4=left.
            relative = math.pi - (2.0 * math.pi * index / count)
            angle = theta + relative
            ex = x + distance * math.cos(angle)
            ey = y + distance * math.sin(angle)
            gx, gy = self.world_to_grid(ex, ey)
            if not self.inside(gx, gy):
                continue
            cells = list(bresenham(sx, sy, gx, gy))
            free_cells = cells[1:-1] if hit else cells[1:]
            for cx, cy in free_cells:
                if self.inside(cx, cy):
                    self.log_odds[cy, cx] = max(-12, self.log_odds[cy, cx] - 1)
            if hit and not (dynamic_mask is not None and dynamic_mask[index]):
                self.log_odds[gy, gx] = min(12, self.log_odds[gy, gx] + 3)

    def scan_score(self, pose, ranges, max_range):
        x, y, theta = pose
        x += self.lidar_offset * math.cos(theta)
        y += self.lidar_offset * math.sin(theta)
        score = 0
        used = 0
        count = len(ranges)
        for index in range(0, count, 6):
            distance = float(ranges[index])
            if not math.isfinite(distance) or distance <= 0.03 or distance >= max_range - 0.08:
                continue
            relative = math.pi - (2.0 * math.pi * index / count)
            gx, gy = self.world_to_grid(
                x + distance * math.cos(theta + relative),
                y + distance * math.sin(theta + relative),
            )
            if self.inside(gx, gy):
                score += int(self.log_odds[gy, gx])
                used += 1
        return score / max(1, used)

    def occupied_inflated(self, radius_m=0.20):
        occupied = self.log_odds >= self.OCCUPIED_LIMIT
        radius = max(1, int(math.ceil(radius_m / self.resolution)))
        return binary_dilate(occupied, radius)

    def frontier_components(self, robot_cell, min_cells=2):
        free = self.log_odds <= self.FREE_LIMIT
        unknown = np.abs(self.log_odds) <= 1
        adjacent_unknown = binary_dilate(unknown, 1)
        frontier = free & adjacent_unknown
        components = []
        rx, ry = robot_cell
        for points in connected_components(frontier, connectivity=8):
            area = len(points)
            if area < min_cells:
                continue
            # One connected frontier can surround an entire room. Keep spatially
            # separated representatives so a doorway is not hidden by its centroid.
            buckets = {}
            for gx, gy in points:
                buckets.setdefault((gx // 8, gy // 8), []).append((gx, gy))
            for members in buckets.values():
                cx = sum(point[0] for point in members) / len(members)
                cy = sum(point[1] for point in members) / len(members)
                gx, gy = min(members, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                distance = math.hypot(gx - rx, gy - ry) * self.resolution
                components.append((gx, gy, min(len(members), 50), distance))
        return components

    def astar(self, start, goal, inflated=None, allow_unknown=False):
        if inflated is None:
            inflated = self.occupied_inflated()
        if not self.inside(*start) or not self.inside(*goal):
            return []

        def traversable(cell):
            gx, gy = cell
            if not self.inside(gx, gy) or inflated[gy, gx]:
                return False
            return allow_unknown or self.log_odds[gy, gx] <= -1

        if not traversable(start):
            inflated = inflated.copy()
            inflated[start[1], start[0]] = False
        if not traversable(goal):
            return []

        open_heap = [(0.0, start)]
        came_from = {}
        cost = {start: 0.0}
        directions = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
        )
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            current_cost = cost[current]
            for dx, dy, step_cost in directions:
                nxt = current[0] + dx, current[1] + dy
                if not traversable(nxt):
                    continue
                if dx and dy and (not traversable((current[0] + dx, current[1])) or
                                  not traversable((current[0], current[1] + dy))):
                    continue
                new_cost = current_cost + step_cost
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                came_from[nxt] = current
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(open_heap, (new_cost + heuristic, nxt))
        return []

    def line_is_free(self, start, end, inflated):
        previous = None
        for gx, gy in bresenham(start[0], start[1], end[0], end[1]):
            if (not self.inside(gx, gy) or inflated[gy, gx]
                    or self.log_odds[gy, gx] > -1):
                return False
            if previous is not None and gx != previous[0] and gy != previous[1]:
                if (inflated[previous[1], gx] or inflated[gy, previous[0]]
                        or self.log_odds[previous[1], gx] > -1
                        or self.log_odds[gy, previous[0]] > -1):
                    return False
            previous = gx, gy
        return True

    def reachable_paths(self, start, inflated):
        """One Dijkstra pass ranks every frontier by actual reachable path length."""
        costs, parents = {start: 0.0}, {}
        queue = [(0.0, start)]
        def free(x, y):
            return self.inside(x, y) and not inflated[y, x] and self.log_odds[y, x] <= -1
        while queue:
            cost, (x, y) = heapq.heappop(queue)
            if cost > costs[(x, y)]:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nxt = x + dx, y + dy
                if not free(*nxt) or (dx and dy and (not free(x + dx, y) or not free(x, y + dy))):
                    continue
                new_cost = cost + math.hypot(dx, dy) * self.resolution
                if new_cost < costs.get(nxt, float('inf')):
                    costs[nxt] = new_cost
                    parents[nxt] = (x, y)
                    heapq.heappush(queue, (new_cost, nxt))
        return costs, parents

    def smooth_path(self, path, inflated):
        if len(path) < 3:
            return path
        result = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1:
                if self.line_is_free(path[anchor], path[candidate], inflated):
                    break
                candidate -= 1
            result.append(path[candidate])
            anchor = candidate
        return result

    def exploration_regions(self, reachable):
        """Approximate rooms by removing narrow doorways, then regrow on free cells."""
        import cv2
        radius = max(1, int(math.ceil(.40 / self.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        # reachable already includes robot clearance; erosion separates doorways.
        core = cv2.erode(reachable.astype(np.uint8), kernel)
        _, labels = cv2.connectedComponents(core, connectivity=4)
        ys, xs = np.nonzero(labels)
        queue = deque(zip(xs.tolist(), ys.tolist()))
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if self.inside(nx, ny) and reachable[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = labels[y, x]
                    queue.append((nx, ny))
        return labels


class PoseEstimator:
    def __init__(self, wheel_radius=0.033, axle_length=0.160):
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length
        self.pose = [0.0, 0.0, 0.0]
        self.previous_encoder = None
        self.distance_travelled = 0.0
        self.odometry_pose = [0.0, 0.0, 0.0]
        self.scan_reference = None
        self.localization_valid = False
        self.localized_pose = [0.0, 0.0, 0.0]
        self.scan_quality = {}

    def update_odometry(self, left, right):
        if self.previous_encoder is None:
            self.previous_encoder = (left, right)
            return self.pose
        dl = (left - self.previous_encoder[0]) * self.wheel_radius
        dr = (right - self.previous_encoder[1]) * self.wheel_radius
        self.previous_encoder = (left, right)
        distance = 0.5 * (dl + dr)
        rotation = (dr - dl) / self.axle_length
        mid_heading = self.pose[2] + 0.5 * rotation
        self.pose[0] += distance * math.cos(mid_heading)
        self.pose[1] += distance * math.sin(mid_heading)
        self.pose[2] = wrap_angle(self.pose[2] + rotation)
        heading = self.odometry_pose[2] + .5 * rotation
        self.odometry_pose[0] += distance * math.cos(heading)
        self.odometry_pose[1] += distance * math.sin(heading)
        self.odometry_pose[2] = wrap_angle(self.odometry_pose[2] + rotation)
        self.distance_travelled += abs(distance)
        return self.pose

    def update_scan_motion(self, ranges, max_range, now, commanded_distance):
        cloud = scan_points(ranges, max_range)
        previous = self.scan_reference
        sample = None
        self.localization_valid = False
        if previous is not None:
            old_cloud, anchor, raw, distance, commanded, stamp = previous
            dx = self.odometry_pose[0] - raw[0]
            dy = self.odometry_pose[1] - raw[1]
            seed = (dx * math.cos(raw[2]) + dy * math.sin(raw[2]),
                    -dx * math.sin(raw[2]) + dy * math.cos(raw[2]),
                    wrap_angle(self.odometry_pose[2] - raw[2]))
            match = match_scans(old_cloud, cloud, seed) if now - stamp <= 1.0 else None
            self.localization_valid = match is not None
            actual = 0.
            if match is not None:
                tx, ty, turn = match['delta']
                self.pose[:] = [anchor[0] + tx * math.cos(anchor[2]) - ty * math.sin(anchor[2]),
                                anchor[1] + tx * math.sin(anchor[2]) + ty * math.cos(anchor[2]),
                                wrap_angle(anchor[2] + turn)]
                self.localized_pose[:] = self.pose
                actual = math.hypot(tx, ty)
                self.scan_quality = dict(rmse_m=round(match['rmse'], 5), overlap=round(match['overlap'], 3))
            else:
                self.scan_quality = {'rejected': True}
            sample = (now, now - stamp, self.distance_travelled - distance, actual,
                      commanded_distance - commanded, self.localization_valid)
        self.scan_reference = (cloud, tuple(self.pose), tuple(self.odometry_pose),
                               self.distance_travelled, commanded_distance, now)
        return sample



class RedAppleDetector:
    VISIT_DISTANCE_M = 0.80
    def __init__(self, camera):
        from ultralytics import YOLO
        import torch

        # Leave CPU time for Webots physics and planning instead of oversubscribing.
        torch.set_num_threads(2)

        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()
        self.focal = self.width / (2.0 * math.tan(camera.getFov() / 2.0))
        self.tracks = []
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(project_root, "models", "YOLO", "yolo11n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model is missing: {model_path}")
        self.model = YOLO(model_path)
        self.last_inference_time = -10.0
        self.boxes = []
        self.people_boxes = []
        self.low_observations = []
        self.low_boxes = []

    def detect(self, image_bytes, pose, now):
        if now - self.last_inference_time < 0.25:
            return []
        self.last_inference_time = now
        self.boxes = []
        self.people_boxes = []
        self.low_observations = []
        self.low_boxes = []
        if not image_bytes:
            return []
        bgra = np.frombuffer(image_bytes, np.uint8).reshape((self.height, self.width, 4))
        bgr = bgra[:, :, :3]
        result = self.model.predict(
            source=bgr,
            conf=0.2,
            iou=0.45,
            classes=[0, 15, 39, 41, 47, 75],  # include vase to resolve person confusion
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]
        candidates = []
        exclusions = []
        if result.boxes is not None:
            detections = [(tuple(float(value) for value in box.xyxy[0].cpu().tolist()),
                           int(box.cls[0].cpu()), float(box.conf[0].cpu()))
                          for box in result.boxes]
            for coords, label, confidence in detections:
                if label == 0:
                    # The apple threshold (.20) is too permissive for person stops.
                    # Compare competing vessel labels before accepting a person.
                    x1, y1, x2, y2 = coords
                    area = max(1., (x2 - x1) * (y2 - y1))
                    vessel = any(other_label in (39, 41, 75) and score >= confidence
                                 and max(0., min(x2, box[2]) - max(x1, box[0])) *
                                 max(0., min(y2, box[3]) - max(y1, box[1])) / area > .5
                                 for box, other_label, score in detections)
                    if confidence < .55 or vessel:
                        continue
                    self.people_boxes.append(coords)
                elif label != 75:
                    self.add_low_obstacle(coords, {15: 'cat', 39: 'container', 41: 'container', 47: 'apple'}[label], pose)
                if label != 47:
                    exclusions.append(coords)
                else:
                    candidates.append((*coords, confidence, "YOLO"))

        # The 10 cm fruit can be too small for the generic COCO model at range.
        # Pure red geometry supplies proposals when YOLO misses it; these are for
        # inspection only. Repeated color sightings NEVER establish apple identity.
        import cv2
        blue, green, red = cv2.split(bgr)
        red_mask = ((red.astype(np.int16) > 75) &
                    (red.astype(np.int16) > 1.7 * green.astype(np.int16)) &
                    (red.astype(np.int16) > 1.7 * blue.astype(np.int16))).astype(np.uint8)
        red_mask[:int(self.height * 0.42), :] = 0
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, 8)
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            # Color alone cannot distinguish a floor texture from a solid object.
            # Keep inspection proposals, but only semantic detections add obstacles.
            if (area < 30 or width < 7 or height < 7 or
                    not 0.62 <= width / height <= 1.55 or
                    not 0.40 <= area / (width * height) <= 0.92):
                continue
            if any(max(0, min(x + width, candidate[2]) - max(x, candidate[0])) *
                   max(0, min(y + height, candidate[3]) - max(y, candidate[1])) >
                   0.25 * area for candidate in candidates):
                continue
            candidates.append((x, y, x + width, y + height, 0.0, "color"))

        observations = []
        for x1, y1, x2, y2, confidence, source in candidates:
            if any(a <= (x1 + x2) / 2 <= c and b <= (y1 + y2) / 2 <= d
                   for a, b, c, d in exclusions):
                continue
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(self.width, int(math.ceil(x2))), min(self.height, int(math.ceil(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            # The centre crop excludes much of the background included by YOLO.
            inset_x, inset_y = max(1, (ix2 - ix1) // 8), max(1, (iy2 - iy1) // 8)
            roi = bgr[iy1 + inset_y:iy2 - inset_y, ix1 + inset_x:ix2 - inset_x].astype(np.int16)
            if roi.size == 0:
                continue
            blue, green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
            maximum = np.maximum(np.maximum(red, green), blue)
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = (maximum - minimum) / np.maximum(maximum, 1)
            red_pixels = ((red == maximum) & (red > 70) & (saturation > 0.42) &
                          (red > 1.7 * green) & (red > 1.7 * blue))
            red_ratio = float(np.count_nonzero(red_pixels)) / red_pixels.size
            if red_ratio < 0.45:
                continue  # YOLO may also label green/purple/orange apples as class 47.
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            box_width, box_height = x2 - x1, y2 - y1
            if not 0.65 <= box_width / max(box_height, 1) <= 1.5:
                continue
            radius = 0.25 * (box_width + box_height)
            if (radius < 3.0 or cy < self.height * 0.42 or
                    ix1 <= 1 or ix2 >= self.width - 1 or iy2 >= self.height - 1):
                continue
            self.boxes.append((ix1, iy1, ix2, iy2, confidence, source))
            bearing = math.atan2((self.width * 0.5 - cx), self.focal)
            distance = (0.10 * self.focal) / (2.0 * radius)
            if not 0.12 <= distance <= 3.8:
                continue
            # extensionSlot is at x=-0.03; camera translation is +0.05.
            camera_x = pose[0] + 0.02 * math.cos(pose[2])
            camera_y = pose[1] + 0.02 * math.sin(pose[2])
            apple_x = camera_x + distance * math.cos(pose[2] + bearing)
            apple_y = camera_y + distance * math.sin(pose[2] + bearing)
            observations.append((
                apple_x, apple_y, confidence, bearing, distance, cx, cy, radius,
                confidence, red_ratio, source,
            ))
        return observations

    def add_low_obstacle(self, coords, label, pose):
        obstacle = floor_obstacle(coords, label, pose, self.focal, self.width, self.height)
        if obstacle is not None:
            self.low_observations.append(obstacle)
            self.low_boxes.append((*map(int, coords), 0., label))

    def update_tracks(self, observations, now):
        newly_confirmed = []
        for x, y, confidence, bearing, observed_distance, cx, cy, radius, _, red_ratio, source in observations:
            nearest = None
            nearest_distance = 0.45
            for track in self.tracks:
                track_distance = math.hypot(x - track["x"], y - track["y"])
                if track_distance < nearest_distance and track.get("matched_at") != now:
                    nearest, nearest_distance = track, track_distance
            if nearest is None:
                nearest = {"x": x, "y": y, "hits": 0, "confirmed": False,
                           "visited": False, "last_seen": now}
                self.tracks.append(nearest)
            if now - nearest["last_seen"] > 1.0:
                nearest["hits"] = nearest["yolo_hits"] = 0
            alpha = 0.22 if not nearest["visited"] else 0.0
            nearest["x"] = (1.0 - alpha) * nearest["x"] + alpha * x
            nearest["y"] = (1.0 - alpha) * nearest["y"] + alpha * y
            nearest["hits"] += 1
            nearest["yolo_hits"] = nearest.get("yolo_hits", 0) + (source == "YOLO")
            if source == "YOLO":
                nearest["last_yolo_seen"] = now
            nearest["last_seen"] = now
            nearest["matched_at"] = now
            nearest["observation"] = {
                "pixel": [round(cx, 1), round(cy, 1)],
                "radius_px": round(radius, 1),
                "bearing_rad": round(bearing, 3),
                "distance_m": round(observed_distance, 3),
                "yolo_confidence": round(confidence, 3),
                "red_ratio": round(red_ratio, 3),
                "source": source,
            }
            if nearest['hits'] >= 3 and nearest['yolo_hits'] >= 3:
                nearest['identified'] = True
            if (nearest.get('identified', False) and not nearest['confirmed']
                    and source == 'YOLO' and observed_distance <= self.VISIT_DISTANCE_M):
                if any(math.hypot(x - old['x'], y - old['y']) < .75 for old in self.visited):
                    nearest['duplicate'] = True
                    continue
                nearest["confirmed"] = True
                nearest['visited'] = True
                nearest['visited_at'] = now
                newly_confirmed.append(nearest)
        self.tracks = [track for track in self.tracks if track["confirmed"] or track.get('identified', False)
                       or now - track["last_seen"] < 4.0 or track.get('retry_after', 0) > now]
        return newly_confirmed

    @property
    def identified(self):
        return [track for track in self.tracks if track.get('identified', track['confirmed'])
                and not track.get('duplicate', False)]

    @property
    def confirmed(self):
        return [track for track in self.tracks if track["confirmed"]]

    @property
    def visited(self):
        return [track for track in self.confirmed if track["visited"]]


class PeopleTracker:
    """Fuse image bearings with LiDAR; retain short-lived, world-frame velocities."""

    def __init__(self):
        self.tracks = []

    def update(self, boxes, focal, width, pose, ranges, now):
        count = len(ranges)
        angles = math.pi - 2 * math.pi * np.arange(count) / count
        mask = np.zeros(count, dtype=bool)
        previous = self.tracks
        tracks = []
        used = set()
        for x1, _, x2, _ in boxes:
            low = math.atan2(width / 2 - x2, focal) - 0.06
            high = math.atan2(width / 2 - x1, focal) + 0.06
            sector = (angles >= low) & (angles <= high)
            valid = sector & np.isfinite(ranges) & (np.asarray(ranges) > 0.1)
            if not np.any(valid):
                continue
            distance = float(np.percentile(np.asarray(ranges)[valid], 25))
            mask |= valid & (np.abs(np.asarray(ranges) - distance) < 0.45)
            bearing = (low + high) / 2 + pose[2]
            x = pose[0] - .03 * math.cos(pose[2]) + distance * math.cos(bearing)
            y = pose[1] - .03 * math.sin(pose[2]) + distance * math.sin(bearing)
            matches = [(math.hypot(x - t['x'], y - t['y']), i, t)
                       for i, t in enumerate(previous) if i not in used]
            vx = vy = 0.0
            if matches:
                delta, index, old = min(matches, key=lambda item: item[0])
                dt = now - old['time']
                if delta < 0.8 and 0.05 < dt < 1.0:
                    used.add(index)
                    vx = clamp((x - old['x']) / dt, -1.2, 1.2)
                    vy = clamp((y - old['y']) / dt, -1.2, 1.2)
            tracks.append(dict(x=x, y=y, vx=vx, vy=vy, time=now))
        tracks.extend(t for i, t in enumerate(previous) if i not in used and now - t['time'] < 0.8)
        self.tracks = tracks
        return mask


class MissionController:
    EXPLORE = "EXPLORE"
    RETURN = "RETURN_HOME"
    COMPLETE = "COMPLETE"

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_encoder = self.left_motor.getPositionSensor()
        self.right_encoder = self.right_motor.getPositionSensor()
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.timestep)
        self.lidar_max = self.lidar.getMaxRange()
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.apple_detector = RedAppleDetector(self.camera)
        self.camera_display = self.robot.getDevice("YOLO camera")
        self.camera_display_image = None
        try:
            self.display = self.robot.getDevice("display")
        except Exception:
            self.display = None

        self.grid = OccupancyGrid(lidar_offset=-0.03)
        self.estimator = PoseEstimator()
        self.state = self.EXPLORE
        self.path = []
        self.path_index = 0
        self.last_plan_time = -10.0
        self.last_path_check_time = -10.0
        self.last_scan_match_time = -10.0
        self.last_display_time = -10.0
        self.last_status_time = -10.0
        self.no_frontier_count = 0
        self.last_wheels = [0.0, 0.0]
        self.display_image = None
        self.start_time = 0.0
        self.apple_goal = None
        self.frontier_goal = None
        self.frontier_best_distance = float("inf")
        self.frontier_progress_time = 0.0
        self.rejected_frontiers = []
        self.visit_counts = np.zeros_like(self.grid.log_odds, dtype=np.uint16)
        self.last_visit_time = -10.0
        self.people = PeopleTracker()
        self.low_obstacles = LowObstacleMap()
        self.dynamic_mask = None
        self.inspection_goal = None
        self.survey_remaining = 2.0 * math.pi
        self.survey_heading = None
        self.camera_seen = np.zeros_like(self.grid.log_odds, dtype=bool)
        self.control_reason = 'STARTING'
        self.stop_reason = None
        self.survey_started = None
        self.survey_timeout = 13.0
        self.survey_direction = 1
        self.last_survey_time = -float('inf')
        self.last_survey_position = (0.0, 0.0)
        self.motion_monitor = MotionMonitor()
        self.recovery = Recovery()
        self.commanded_distance = 0.
        self.last_motion_time = self.robot.getTime()
        self.last_localization_time = self.last_motion_time
        self.close_stop_since = None
        self.escape_started = None
        self.escape_origin = None
        self.escape_anchor = None
        self.escape_attempts = 0

    def navigation_obstacles(self):
        inflated = self.grid.occupied_inflated()
        if hasattr(self, 'low_obstacles'):
            for obj in self.low_obstacles.active(self.robot.getTime()):
                cell = self.grid.world_to_grid(obj['x'], obj['y'])
                if self.grid.inside(*cell):
                    draw_disc(inflated, cell, int(math.ceil((obj['radius'] + .20) / self.grid.resolution)), True)
        recovery = getattr(self, 'recovery', None)
        if recovery is None:
            return inflated
        now = self.robot.getTime()
        recovery.failed_areas = [area for area in recovery.failed_areas if area[2] > now]
        for x, y, _ in recovery.failed_areas:
            gx, gy = self.grid.world_to_grid(x, y)
            radius = int(math.ceil(.22 / self.grid.resolution))
            if self.grid.inside(gx, gy):
                draw_disc(inflated, (gx, gy), radius, True)
        return inflated

    def set_wheels(self, left, right, emergency=False):
        maximum = 6.0
        acceleration_step = 0.7 * self.timestep / 32.0
        scale = max(1.0, abs(left) / maximum, abs(right) / maximum)
        targets = [left / scale, right / scale]
        for i in range(2):
            targets[i] = 0.0 if emergency else clamp(
                targets[i],
                self.last_wheels[i] - acceleration_step,
                self.last_wheels[i] + acceleration_step,
            )
        self.left_motor.setVelocity(targets[0])
        self.right_motor.setVelocity(targets[1])
        self.last_wheels = targets

    def reject_frontier(self, now, reason="stalled", pose=None):
        if reason == "reached" and pose is not None:
            self.request_survey(pose, now)
        if self.frontier_goal is not None:
            self.rejected_frontiers.append((self.frontier_goal, now + 120.0))
            print(f"[PLAN] Skipping {reason} frontier {self.frontier_goal} for 120 s.")
        self.frontier_goal = None
        self.path = []

    def choose_frontier_path(self, pose, now):
        start = self.grid.world_to_grid(pose[0], pose[1])
        inflated = self.navigation_obstacles()
        seen = getattr(self, 'camera_seen', None)
        if seen is None:
            seen = self.visit_counts > 0
        self.rejected_frontiers = [(cell, expiry) for cell, expiry in self.rejected_frontiers
                                   if expiry > now]
        def blocked(cell):
            return any(math.hypot(cell[0] - old[0], cell[1] - old[1]) < 7
                       for old, _ in self.rejected_frontiers)

        if self.frontier_goal is not None and not blocked(self.frontier_goal):
            gx, gy = self.frontier_goal
            already_checked = self.visit_counts[gy, gx] > 0 and seen[gy, gx]
            if not already_checked:
                path = self.grid.astar(start, (gx, gy), inflated)
                if path:
                    return self.grid.smooth_path(path, inflated)
                self.reject_frontier(now)
            else:
                self.frontier_goal = None

        # Prefer new space and reachable nearby boundaries. Avoid repeatedly
        # crossing the same room for a large but inaccessible frontier.
        candidates = [item for item in self.grid.frontier_components(start)
                      if not blocked(item[:2])]
        best = None
        costs, parents = self.grid.reachable_paths(start, inflated)
        reachable = np.zeros_like(inflated, dtype=bool)
        for gx, gy in costs:
            reachable[gy, gx] = True
        rooms = self.grid.exploration_regions(reachable)
        anchor = getattr(self, 'room_anchor', start)
        active_room = int(rooms[anchor[1], anchor[0]])
        if not active_room:
            anchor = start
            active_room = int(rooms[start[1], start[0]])
        if not hasattr(self, 'room_started'):
            self.room_started = now
        # Do not remain indefinitely in a room whose remaining views are blocked.
        room_priority = active_room > 0 and now - self.room_started < 120.
        novelty_cache = {}
        view_radius = int(1.2 / self.grid.resolution)
        view_offsets = [(round(view_radius * math.cos(a)), round(view_radius * math.sin(a)))
                        for a in np.linspace(-math.pi, math.pi, 32, endpoint=False)]
        # Evaluate neighbourhoods, not a single untouched cell beside a route.
        def novelty(gx, gy):
            if (gx, gy) in novelty_cache:
                return novelty_cache[(gx, gy)]
            region = np.s_[max(0, gy - 8):min(self.grid.size, gy + 9),
                           max(0, gx - 8):min(self.grid.size, gx + 9)]
            free = reachable[region] & (rooms[region] == rooms[gy, gx])
            size = max(1, int(np.count_nonzero(free)))
            visited = float(np.count_nonzero(free & (self.visit_counts[region] > 0))) / size
            visible_unseen = set()
            for dx, dy in view_offsets:
                for vx, vy in bresenham(gx, gy, gx + dx, gy + dy):
                    if not self.grid.inside(vx, vy) or self.grid.log_odds[vy, vx] > -1:
                        break
                    if not seen[vy, vx] and math.hypot(vx - gx, vy - gy) * self.grid.resolution >= .20:
                        visible_unseen.add((vx, vy))
            unseen = len(visible_unseen)
            novelty_cache[(gx, gy)] = (visited, unseen)
            return visited, unseen
        # Keep camera viewpoints alongside frontiers, including within this room.
        if hasattr(self, 'camera_seen'):
            visual_candidates = []
            for (gx, gy), distance in costs.items():
                if gx % 8 or gy % 8 or distance < 0.4 or blocked((gx, gy)):
                    continue
                _, gain = novelty(gx, gy)
                if gain >= 20:
                    visual_candidates.append((gx, gy, min(50, gain / 4), distance))
            candidates.extend(visual_candidates)
        for gx, gy, gain, distance in candidates:
            travel = costs.get((gx, gy))
            if travel is None or travel < .4:
                continue
            visited, unseen = novelty(gx, gy)
            # Unvisited areas first; then camera-unseen floors; revisit only as
            # a fallback. Rank within each group by gain and real route cost.
            priority = (2 if unseen >= 20 else
                        1 if self.visit_counts[gy, gx] == 0 and visited < .25 else 0)
            score = (.06 * gain + .015 * min(unseen, 100) - travel - 2.0 * visited
                     - .3 * math.log1p(int(self.visit_counts[gy, gx])))
            preferred_room = (rooms[gy, gx] == active_room if room_priority else
                              active_room > 0 and rooms[gy, gx] != active_room)
            rank = (int(preferred_room and priority > 0), priority, score)
            if best is None or rank > best[0]:
                best = (rank, (gx, gy), math.hypot(gx - start[0], gy - start[1]) * self.grid.resolution)
        if best is not None:
            _, self.frontier_goal, self.frontier_best_distance = best
            chosen_room = int(rooms[self.frontier_goal[1], self.frontier_goal[0]])
            if chosen_room != active_room or not room_priority:
                self.room_started = now
            self.room_anchor = self.frontier_goal
            path = [self.frontier_goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            path.reverse()
            self.frontier_progress_time = now
            return self.grid.smooth_path(path, inflated)
        return []

    def choose_apple_path(self, pose, apple, inspection=False):
        start = self.grid.world_to_grid(pose[0], pose[1])
        centre = self.grid.world_to_grid(apple["x"], apple["y"])
        inflated = self.navigation_obstacles()
        # Approach an accessible free cell near the fruit; the fruit's own cell
        # may be occupied, and the robot must leave collision clearance.
        cells = []
        for dy in range(-7, 8):
            for dx in range(-7, 8):
                gx, gy = centre[0] + dx, centre[1] + dy
                distance = math.hypot(dx, dy) * self.grid.resolution
                minimum, maximum = (0.48, 0.72) if inspection else (0.30, 0.48)
                if (minimum <= distance <= maximum and self.grid.inside(gx, gy)
                        and not inflated[gy, gx] and self.grid.log_odds[gy, gx] <= -1):
                    cells.append((gx, gy, distance))
        cells.sort(key=lambda cell: math.hypot(cell[0] - start[0], cell[1] - start[1]) +
                   8.0 * abs(cell[2] - 0.38))
        for gx, gy, _ in cells[:30]:
            path = self.grid.astar(start, (gx, gy), inflated)
            if path:
                return self.grid.smooth_path(path, inflated)
        return []

    def reactive_exploration_target(self, pose, ranges):
        """Keep discovering space while the next frontier is not yet reachable."""
        count = len(ranges)
        best_score, best_relative, best_clearance = -float("inf"), 0.0, 0.0
        half_window = max(2, count // 36)
        for center in range(0, count, max(1, count // 24)):
            samples = []
            for offset in range(-half_window, half_window + 1):
                value = float(ranges[(center + offset) % count])
                if not math.isnan(value) and value > .03:
                    samples.append(min(value, self.lidar_max))
            if not samples:
                continue
            relative = math.pi - 2.0 * math.pi * center / count
            # Prefer open space, with a small bias against turning fully backward.
            clearance = sum(samples) / len(samples)
            look_ahead = min(clearance * 0.6, 1.0)
            gx, gy = self.grid.world_to_grid(
                pose[0] + look_ahead * math.cos(pose[2] + relative),
                pose[1] + look_ahead * math.sin(pose[2] + relative))
            visits = self.visit_counts[gy, gx] if self.grid.inside(gx, gy) else 100
            unseen = (self.grid.inside(gx, gy) and not self.camera_seen[gy, gx])
            score = (clearance - 0.12 * abs(relative)
                     - .3 * math.log1p(int(visits)) + .8 * unseen + .8 * (visits == 0))
            if score > best_score:
                best_score, best_relative = score, relative
                best_clearance = clearance
        angle = pose[2] + best_relative
        distance = clamp(best_clearance * 0.45, 0.45, 1.0)
        return pose[0] + distance * math.cos(angle), pose[1] + distance * math.sin(angle)

    def choose_home_path(self, pose):
        start = self.grid.world_to_grid(pose[0], pose[1])
        home = self.grid.world_to_grid(0.0, 0.0)
        inflated = self.navigation_obstacles()
        path = self.grid.astar(start, home, inflated)
        return self.grid.smooth_path(path, inflated) if path else []

    def plan(self, pose, now):
        if self.state == self.EXPLORE:
            pending = [track for track in getattr(self.apple_detector, 'identified', self.apple_detector.confirmed)
                       if not track["visited"]]
            pending.sort(key=lambda track: math.hypot(track["x"] - pose[0], track["y"] - pose[1]))
            self.apple_goal = None
            self.path = []
            self.inspection_goal = None
            for apple in pending:
                if now < apple.get('defer_until', 0):
                    continue
                distance = math.hypot(apple['x'] - pose[0], apple['y'] - pose[1])
                if distance < apple.get('best_distance', float('inf')) - .1:
                    apple['best_distance'], apple['progress_time'] = distance, now
                elif now - apple.get('progress_time', now) > 25:
                    apple['defer_until'] = now + 45
                    apple['best_distance'] = float('inf')
                    continue
                self.path = self.choose_apple_path(pose, apple)
                if self.path:
                    self.apple_goal = apple
                    break
            if not self.path:
                for candidate in getattr(self.apple_detector, 'tracks', []):
                    if (candidate['confirmed'] or candidate.get('identified', False) or candidate.get('duplicate', False)
                            or candidate['hits'] < 3
                            or candidate.get('yolo_hits', 0) < 1
                            or now - candidate['last_seen'] > 1.0
                            or now < candidate.get('retry_after', 0)):
                        continue
                    since = candidate.setdefault('inspect_since', now)
                    if now - since > 6.0:
                        candidate['retry_after'] = now + 90.0
                        candidate.pop('inspect_since', None)
                        continue
                    self.path = self.choose_apple_path(pose, candidate, inspection=True)
                    if self.path:
                        self.inspection_goal = candidate
                        break
            if not self.path:
                if self.frontier_goal is not None:
                    goal_x, goal_y = self.grid.grid_to_world(*self.frontier_goal)
                    distance = math.hypot(goal_x - pose[0], goal_y - pose[1])
                    if distance < 0.25:
                        self.reject_frontier(now, "reached", pose)
                    elif distance < self.frontier_best_distance - 0.15:
                        self.frontier_best_distance = distance
                        self.frontier_progress_time = now
                    elif now - self.frontier_progress_time > 25.0:
                        self.reject_frontier(now)
                self.path = self.choose_frontier_path(pose, now)
            else:
                self.frontier_goal = None
            if not self.path:
                self.no_frontier_count += 1
                if self.no_frontier_count in (1, 5, 10, 15):
                    print(f"[PLAN] Frontier temporarily unreachable ({self.no_frontier_count}/20); reactive exploration.")
                if self.no_frontier_count == 20:
                    print("[PLAN] No reachable frontier; continuing reactive search for both apples.")
            else:
                self.no_frontier_count = 0
        elif self.state == self.RETURN:
            self.path = self.choose_home_path(pose)
        self.path_index = 1 if len(self.path) > 1 else 0
        self.last_plan_time = now

    def waypoint(self, pose):
        if not self.path:
            return None
        while self.path_index < len(self.path):
            target = self.grid.grid_to_world(*self.path[self.path_index])
            threshold = 0.18 if self.path_index == len(self.path) - 1 else 0.12
            if math.hypot(target[0] - pose[0], target[1] - pose[1]) > threshold:
                return target
            self.path_index += 1
        return None

    def local_control(self, pose, target, ranges):
        if target is None:
            return 0.0, 0.0
        target_distance = math.hypot(target[0] - pose[0], target[1] - pose[1])
        desired = math.atan2(target[1] - pose[1], target[0] - pose[0])
        heading_error = wrap_angle(desired - pose[2])

        count = len(ranges)
        obstacle_points = []
        if hasattr(self, 'recovery'):
            for x, y, expiry in self.recovery.failed_areas:
                if expiry <= self.robot.getTime():
                    continue
                dx, dy = x - pose[0], y - pose[1]
                obstacle_points.append((dx * math.cos(pose[2]) + dy * math.sin(pose[2]),
                                        -dx * math.sin(pose[2]) + dy * math.cos(pose[2])))
        for index in range(count):
            distance = float(ranges[index])
            if math.isfinite(distance) and distance < 1.1:
                relative = math.pi - 2.0 * math.pi * index / count
                obstacle_points.append((distance * math.cos(relative) - 0.03, distance * math.sin(relative)))

        best = None
        for velocity in (0.0, 0.06, 0.12, 0.18):
            if velocity > 0 and abs(heading_error) > 1.1:
                continue
            for omega in (-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8):
                if velocity == 0.0 and abs(omega) < 0.5:
                    continue
                horizon = 0.85
                trajectory = []
                for elapsed in (0.28, 0.56, horizon):
                    if abs(omega) < 1e-5:
                        px, py = velocity * elapsed, 0.0
                    else:
                        radius = velocity / omega
                        px = radius * math.sin(omega * elapsed)
                        py = radius * (1.0 - math.cos(omega * elapsed))
                    trajectory.append((px, py))
                ptheta = omega * horizon
                clearance = min(
                    (math.hypot(ox - px, oy - py)
                     for px, py in trajectory for ox, oy in obstacle_points),
                    default=1.1,
                )
                if clearance < (0.18 if velocity == 0 else 0.24):
                    continue
                people_clearance = float('inf')
                for person in getattr(self.people, 'tracks', []):
                    for elapsed, (px, py) in zip((0.28, 0.56, horizon), trajectory):
                        wx = pose[0] + px * math.cos(pose[2]) - py * math.sin(pose[2])
                        wy = pose[1] + px * math.sin(pose[2]) + py * math.cos(pose[2])
                        separation = math.hypot(person['x'] + person['vx'] * elapsed - wx,
                                                person['y'] + person['vy'] * elapsed - wy)
                        people_clearance = min(people_clearance, separation)
                if people_clearance < 0.70:
                    continue
                predicted_error = abs(wrap_angle(heading_error - ptheta))
                score = (
                    2.2 * velocity
                    - 0.65 * predicted_error
                    + 0.42 * min(clearance, 0.8)
                    - 0.05 * abs(omega)
                )
                if target_distance < 0.45:
                    score -= velocity * 0.8
                if best is None or score > best[0]:
                    best = (score, velocity, omega)
        if best is None:
            return 0.0, 0.0
        _, velocity, omega = best
        wheel_radius, axle = 0.033, 0.160
        return (
            (velocity - omega * axle * 0.5) / wheel_radius,
            (velocity + omega * axle * 0.5) / wheel_radius,
        )

    def update_mission(self, pose, now):
        """Perception atomically counts a nearby confirmed apple as visited."""
        if self.state == self.EXPLORE and len(self.apple_detector.visited) >= 2:
            self.state = self.RETURN
            self.path = []
            print('[MISSION] Both red apples approached; planning return to start.')
        if self.state == self.RETURN and math.hypot(pose[0], pose[1]) < 0.22:
            self.state = self.COMPLETE

    def emergency_required(self, ranges, pose):
        self.stop_reason = None
        values = np.asarray(ranges)
        valid = np.isfinite(values) & (values > 0.03)
        if np.count_nonzero(valid | np.isposinf(values)) < len(values) * 0.75:
            self.stop_reason = 'INVALID_LIDAR'
            return True
        angles = math.pi - 2 * math.pi * np.arange(len(values)) / len(values)
        distances = np.hypot(values[valid] * np.cos(angles[valid]) - .03,
                             values[valid] * np.sin(angles[valid]))
        self.nearest_obstacle_m = float(np.min(distances)) if len(distances) else None
        if len(distances) and distances.min() < .18:
            self.stop_reason = 'OBSTACLE_TOO_CLOSE'
            return True
        if any(math.hypot(t['x'] - pose[0], t['y'] - pose[1]) < 0.65
               for t in self.people.tracks):
            self.stop_reason = 'PERSON_TOO_CLOSE'
            return True
        return False

    def request_survey(self, pose, now):
        """Only inspect a substantially new, camera-unseen area; never rearm a turn."""
        if getattr(self, 'survey_remaining', 0) > 0:
            return False
        if now - self.last_survey_time < 22.0:
            return False
        if math.hypot(pose[0] - self.last_survey_position[0],
                      pose[1] - self.last_survey_position[1]) < 1.0:
            return False
        start = self.grid.world_to_grid(*pose[:2])
        visible = set()
        for relative in np.linspace(-math.pi, math.pi, 64, endpoint=False):
            bearing = pose[2] + relative
            end = self.grid.world_to_grid(pose[0] + 2 * math.cos(bearing),
                                          pose[1] + 2 * math.sin(bearing))
            for gx, gy in bresenham(*start, *end):
                if not self.grid.inside(gx, gy) or self.grid.log_odds[gy, gx] > -1:
                    break
                visible.add((gx, gy))
        unseen = [(gx, gy) for gx, gy in visible if not self.camera_seen[gy, gx]]
        if len(unseen) < 16 or len(unseen) < .20 * len(visible):
            return False
        # Choose the half-turn toward the greater amount of unseen floor.
        left = right = 0
        for gx, gy in unseen:
            x, y = self.grid.grid_to_world(gx, gy)
            if wrap_angle(math.atan2(y - pose[1], x - pose[0]) - pose[2]) >= 0:
                left += 1
            else:
                right += 1
        self.survey_direction = 1 if left >= right else -1
        self.survey_remaining = math.pi
        self.survey_timeout = 7.0
        self.survey_heading = self.survey_started = None
        self.last_survey_time = now
        self.last_survey_position = tuple(pose[:2])
        return True

    def proximity_escape(self, pose, ranges, now):
        """Bound each escape, but never permanently lock out at one location."""
        person_stop = self.control_reason == 'PERSON_TOO_CLOSE'
        blocked = self.control_reason in ('OBSTACLE_TOO_CLOSE', 'NO_SAFE_TRAJECTORY', 'PERSON_TOO_CLOSE')
        if (not blocked or self.recovery.active or not self.estimator.localization_valid):
            if self.escape_started is not None:
                self.path = []
                self.last_plan_time = -10
            self.close_stop_since = self.escape_started = None
            return None
        if self.close_stop_since is None:
            self.close_stop_since = now
        if now - self.close_stop_since < 1.0:
            return None
        if self.escape_anchor is None or math.dist(pose[:2], self.escape_anchor) > .3:
            self.escape_anchor = tuple(pose[:2])
            self.escape_attempts = 0
        if self.escape_started is not None:
            if (now - self.escape_started >= (7.5 if person_stop else 4.0)
                    or math.dist(pose[:2], self.escape_origin) >= (.30 if person_stop else .20)):
                moved = math.dist(pose[:2], self.escape_origin)
                print(f'[RECOVERY] Clearance attempt ended: localized progress={moved:.3f}m.')
                self.escape_started = None
                self.close_stop_since = now
                self.path = []
                self.last_plan_time = -10
                self.motion_monitor.reset()
                return None
        elif self.escape_attempts >= 2 and now - self.close_stop_since < 5.0:
            # Pause between retries; the old unconditional return latched forever.
            return None
        command = clearance_escape(ranges, pose, self.people.tracks,
                                   curved=self.escape_attempts >= 1 and not person_stop)
        if command is not None and self.escape_started is None:
            self.escape_started = now
            self.escape_origin = tuple(pose[:2])
            self.escape_attempts += 1
            self.survey_remaining = 0
            print('[RECOVERY] Slowly increasing clearance from nearby obstacle.')
        return command

    def survey_control(self, pose, now):
        """Bound the initial look-around even when rotation cannot make progress."""
        if self.survey_remaining <= 0:
            self.survey_heading = self.survey_started = None
            return None
        if self.survey_started is None:
            self.survey_started = now
            self.last_survey_position = tuple(pose[:2])
        if self.survey_heading is not None:
            self.survey_remaining -= abs(wrap_angle(pose[2] - self.survey_heading))
        self.survey_heading = pose[2]
        if (self.survey_remaining <= 0
                or now - self.survey_started >= getattr(self, 'survey_timeout', 13.0)
                or getattr(self, 'apple_goal', None) is not None
                or getattr(self, 'inspection_goal', None) is not None):
            self.survey_remaining = 0
            self.survey_heading = self.survey_started = None
            self.last_survey_time = now
            return None
        direction = getattr(self, 'survey_direction', 1)
        return -1.4 * direction, 1.4 * direction

    def record_camera_coverage(self, pose):
        start = self.grid.world_to_grid(pose[0], pose[1])
        # A small apple glimpsed at 2m is not a thorough inspection. Require a
        # closer view and do not count floor below the camera's vertical field.
        near = .073 * 2 * self.apple_detector.focal / self.camera.getHeight()
        for relative in np.linspace(-self.camera.getFov() / 2, self.camera.getFov() / 2, 45):
            angle = pose[2] + relative
            end = self.grid.world_to_grid(pose[0] + 1.2 * math.cos(angle),
                                          pose[1] + 1.2 * math.sin(angle))
            for gx, gy in bresenham(*start, *end):
                if not self.grid.inside(gx, gy) or self.grid.log_odds[gy, gx] > -1:
                    break
                distance = math.hypot(gx - start[0], gy - start[1]) * self.grid.resolution
                if distance >= near:
                    self.camera_seen[gy, gx] = True

    def update_display(self, pose, now):
        if self.display is None or now - self.last_display_time < 0.5:
            return
        self.last_display_time = now
        try:
            image = np.full((self.grid.size, self.grid.size, 3), 127, np.uint8)
            image[self.grid.log_odds <= self.grid.FREE_LIMIT] = (245, 245, 245)
            image[self.grid.log_odds >= self.grid.OCCUPIED_LIMIT] = (20, 20, 20)
            for cell in self.path:
                draw_disc(image, cell, 1, (255, 120, 0))
            hx, hy = self.grid.world_to_grid(0.0, 0.0)
            draw_disc(image, (hx, hy), 4, (0, 210, 0))
            for apple in self.apple_detector.identified:
                ax, ay = self.grid.world_to_grid(apple["x"], apple["y"])
                draw_disc(image, (ax, ay), 5, (0, 180, 0) if apple["visited"] else (255, 0, 0))
            rx, ry = self.grid.world_to_grid(pose[0], pose[1])
            draw_disc(image, (rx, ry), 4, (0, 90, 255))
            heading_end = (int(rx + 10 * math.cos(pose[2])), int(ry - 10 * math.sin(pose[2])))
            for px, py in bresenham(rx, ry, heading_end[0], heading_end[1]):
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    image[py, px] = (0, 90, 255)
            if self.display_image is not None:
                self.display.imageDelete(self.display_image)
            self.display_image = self.display.imageNew(
                image.tobytes(), self.display.RGB, self.grid.size, self.grid.size
            )
            self.display.imagePaste(self.display_image, 0, 0, False)
            self.display.setColor(0xFFFFFF)
            self.display.setFont("Arial", 18, True)
            self.display.drawText(
                f"{self.state} | visited {len(self.apple_detector.visited)}/2",
                8,
                8,
            )
        except Exception as error:
            print(f"[DISPLAY] disabled: {error}")
            self.display = None

    def update_camera_display(self, image_bytes):
        if not image_bytes:
            return
        try:
            frame = np.frombuffer(image_bytes, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4))[:, :, 2::-1].copy()
            height, width = frame.shape[:2]
            overlay_boxes = list(self.apple_detector.boxes)
            overlay_boxes.extend(self.apple_detector.low_boxes)
            overlay_boxes.extend((*map(int, coords), 0.0, 'person')
                                 for coords in self.apple_detector.people_boxes)
            for x1, y1, x2, y2, confidence, source in overlay_boxes:
                x1, y1 = clamp(x1, 0, width - 1), clamp(y1, 0, height - 1)
                x2, y2 = clamp(x2, x1 + 1, width), clamp(y2, y1 + 1, height)
                color = (255, 40, 40) if source == "YOLO" else (255, 190, 0)
                if source == 'person':
                    color = (0, 220, 255)
                elif source not in ('YOLO', 'color'):
                    color = (190, 80, 255)
                frame[y1:y1 + 3, x1:x2] = color
                frame[y2 - 3:y2, x1:x2] = color
                frame[y1:y2, x1:x1 + 3] = color
                frame[y1:y2, x2 - 3:x2] = color
            if self.camera_display_image is not None:
                self.camera_display.imageDelete(self.camera_display_image)
            self.camera_display_image = self.camera_display.imageNew(
                frame.tobytes(), self.camera_display.RGB, width, height)
            self.camera_display.imagePaste(self.camera_display_image, 0, 0, False)
            self.camera_display.setColor(0xFF3030)
            self.camera_display.setFont("Arial", 18, True)
            for x1, y1, _, _, confidence, source in overlay_boxes:
                label = f"YOLO red apple {confidence:.2f}" if source == "YOLO" else "red candidate"
                if source == 'person':
                    label = 'person - yield'
                elif source not in ('YOLO', 'color'):
                    label = source + ' - obstacle'
                self.camera_display.drawText(label, x1, max(0, y1 - 22))
        except Exception as error:
            print(f"[CAMERA DISPLAY] {error}")
            self.camera_display = None

    def save_results(self, pose):
        output_dir = os.environ.get('TB3_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), "mission_output"))
        os.makedirs(output_dir, exist_ok=True)
        image = np.full((self.grid.size, self.grid.size), 127, np.uint8)
        image[self.grid.log_odds <= self.grid.FREE_LIMIT] = 255
        image[self.grid.log_odds >= self.grid.OCCUPIED_LIMIT] = 0
        with open(os.path.join(output_dir, "occupancy_map.pgm"), "wb") as stream:
            stream.write(f"P5\n{self.grid.size} {self.grid.size}\n255\n".encode("ascii"))
            stream.write(image.tobytes())
        result = {
            "state": self.state,
            "odometry_pose_m_rad": [round(v, 4) for v in self.estimator.odometry_pose],
            "motion_evidence": self.motion_monitor.metrics,
            "scan_quality": self.estimator.scan_quality,
            "recovery": {"stage": self.recovery.stage, "attempts": self.recovery.attempts,
                         "reason": self.recovery.last_reason},
            "control_reason": self.control_reason,
            "nearest_obstacle_m": getattr(self, 'nearest_obstacle_m', None),
            "person_boxes": len(self.apple_detector.people_boxes),
            "tracked_people": len(self.people.tracks),
            "low_obstacles": self.low_obstacles.active(self.robot.getTime()),
            "stop_reason": self.stop_reason,
            "survey_remaining_rad": round(self.survey_remaining, 3),
            "wheel_command_rad_s": self.last_wheels,
            "inspection_goal": None if self.inspection_goal is None else {
                k: self.inspection_goal.get(k) for k in ('x', 'y', 'yolo_hits', 'inspect_since')},
            "final_pose_local_m_rad": [round(value, 4) for value in pose],
            "red_apples_found": len(self.apple_detector.confirmed),
            "red_apples_visited": len(self.apple_detector.visited),
            "success": self.state == self.COMPLETE and len(self.apple_detector.visited) >= 2,
            "home_error_m": round(math.hypot(pose[0], pose[1]), 3),
            "elapsed_simulation_s": round(self.robot.getTime() - self.start_time, 2),
            "targets": [{key: track[key] for key in
                         ('x', 'y', 'visited', 'last_seen', 'observation', 'visited_at') if key in track}
                        for track in self.apple_detector.confirmed],
            "red_apple_positions_local_m": [
                [round(track["x"], 3), round(track["y"], 3)]
                for track in self.apple_detector.confirmed
            ],
        }
        with open(os.path.join(output_dir, "mission_result.json"), "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)

    def run(self):
        print("[MISSION] Autonomous SLAM search started. Local start pose=(0, 0, 0).")
        self.start_time = self.robot.getTime()
        while self.robot.step(self.timestep) != -1:
            now = self.robot.getTime()
            dt = now - self.last_motion_time
            self.commanded_distance += abs(.5 * sum(self.last_wheels) * .033) * dt
            self.last_motion_time = now
            pose = self.estimator.update_odometry(
                self.left_encoder.getValue(), self.right_encoder.getValue()
            )
            ranges = self.lidar.getRangeImage()

            matching_ranges = np.array(ranges, dtype=float)
            if self.dynamic_mask is not None:
                matching_ranges[self.dynamic_mask] = np.nan
            scan_updated = now - self.last_scan_match_time >= .45
            if scan_updated:
                sample = self.estimator.update_scan_motion(
                    matching_ranges, self.lidar_max, now, self.commanded_distance)
                pose = self.estimator.pose
                self.last_scan_match_time = now
                if self.estimator.localization_valid:
                    self.last_localization_time = now
                if sample is not None and not self.recovery.active:
                    if self.escape_started is not None:
                        # Clearance escape has its own measured progress/timeout.
                        # Ordinary recovery rejects the very soft margin we are
                        # leaving and would latch into BLOCKED at this distance.
                        self.motion_monitor.reset()
                        stuck = False
                    elif abs(.5 * sum(self.last_wheels) * .033) > .025:
                        stuck = self.motion_monitor.update(*sample)
                    else:
                        self.motion_monitor.reset()
                        stuck = False
                    if stuck and self.recovery.start(pose, now):
                        self.path = []
                        self.survey_remaining = 0
                        self.last_survey_time = now
                        self.last_survey_position = tuple(pose[:2])
                        self.motion_monitor.reset()
                        print(f'[RECOVERY] Wheel motion disagrees with LiDAR: {self.motion_monitor.metrics}')
                if self.estimator.localization_valid and not self.recovery.active:
                    self.grid.integrate_scan(pose, ranges, self.lidar_max, dynamic_mask=self.dynamic_mask)

            camera_image = self.camera.getImage()
            inference_due = now - self.apple_detector.last_inference_time >= 0.25
            observations = self.apple_detector.detect(camera_image, pose, now)
            if inference_due:
                self.dynamic_mask = self.people.update(
                    self.apple_detector.people_boxes, self.apple_detector.focal,
                    self.apple_detector.width, pose, ranges, now)
                if self.estimator.localization_valid and not self.recovery.active and camera_image:
                    self.low_obstacles.update(self.apple_detector.low_observations, now)
                    self.record_camera_coverage(pose)
            if not self.estimator.localization_valid or self.recovery.active:
                observations = []
            if now - self.last_visit_time >= 0.5:
                gx, gy = self.grid.world_to_grid(pose[0], pose[1])
                if self.grid.inside(gx, gy):
                    radius = 4  # penalize the surrounding 0.3 m, not one grid cell
                    x0, x1 = max(0, gx - radius), min(self.grid.size, gx + radius + 1)
                    y0, y1 = max(0, gy - radius), min(self.grid.size, gy + radius + 1)
                    patch = self.visit_counts[y0:y1, x0:x1]
                    np.minimum(patch.astype(np.uint32) + 1, 65535, out=patch, casting="unsafe")
                self.last_visit_time = now

            if self.camera_display is not None and inference_due:
                self.update_camera_display(camera_image)
            for track in self.apple_detector.update_tracks(observations, now):
                self.path = []
                print(
                    f"[MISSION] Red apple confirmed and visited "
                    f"(apples={len(self.apple_detector.confirmed)}, visited={len(self.apple_detector.visited)}): "
                    f"({track['x']:.2f}, {track['y']:.2f}) m; {track['observation']}"
                )

            if scan_updated and self.estimator.localization_valid and not self.recovery.active:
                self.update_mission(pose, now)
            # Only downstream action uses virtual floor-obstacle returns.
            # Pose estimation and mapping above always use the real LiDAR scan.
            ranges = self.low_obstacles.control_ranges(ranges, pose, now)
            home_distance = math.hypot(pose[0], pose[1])
            if self.state == self.COMPLETE:
                self.set_wheels(0.0, 0.0, emergency=True)
                self.save_results(pose)
                print(
                    f"[MISSION] Complete: returned home, "
                    f"red apples visited={len(self.apple_detector.visited)}/2."
                )
                while self.robot.step(self.timestep) != -1:
                    self.set_wheels(0.0, 0.0)
                    self.update_display(pose, self.robot.getTime())
                return

            if self.path and self.path_index < len(self.path) and now - self.last_path_check_time >= .5:
                self.last_path_check_time = now
                current = self.grid.world_to_grid(pose[0], pose[1])
                if not self.grid.line_is_free(current, self.path[self.path_index], self.navigation_obstacles()):
                    self.path = []
                    self.last_plan_time = -10.0
            needs_plan = (
                now - self.last_plan_time >= 1.5
                and (not self.path or self.path_index >= len(self.path) or now - self.last_plan_time >= 4.0)
            )
            if self.state != self.COMPLETE and needs_plan and not self.recovery.active:
                self.plan(pose, now)

            target = self.waypoint(pose)
            self.control_reason = 'FOLLOW_PATH' if target is not None else 'WAIT_FOR_PATH'
            if target is None and self.state == self.EXPLORE and not (self.apple_goal or self.inspection_goal):
                target = self.reactive_exploration_target(pose, ranges)
                self.control_reason = 'REACTIVE_EXPLORE'
            if target is None and self.state == self.EXPLORE:
                focus = self.apple_goal or self.inspection_goal
                if focus is not None:
                    self.control_reason = 'VERIFY_APPLE'
                    angle = wrap_angle(math.atan2(focus['y'] - pose[1], focus['x'] - pose[0]) - pose[2])
                    left, right = -clamp(angle * 2, -1.5, 1.5), clamp(angle * 2, -1.5, 1.5)
                else:
                    left, right = self.local_control(pose, target, ranges)
            else:
                left, right = self.local_control(pose, target, ranges)
            if self.state == self.EXPLORE:
                survey = self.survey_control(pose, now)
                if survey is not None:
                    self.control_reason = 'SURVEY'
                    left, right = survey
            if self.recovery.active:
                was_stage = self.recovery.stage
                left, right = self.recovery.command(
                    self.estimator.localized_pose, ranges, self.people.tracks, now, self.estimator.localization_valid)
                self.control_reason = 'RECOVERY_' + (self.recovery.stage or 'REPLAN')
                if was_stage != self.recovery.stage:
                    print(f'[RECOVERY] {was_stage} -> {self.recovery.stage or "REPLAN"}; {self.recovery.last_reason}')
                if not self.recovery.active:
                    self.path = []
                    self.frontier_goal = None
                    self.apple_goal = self.inspection_goal = None
                    self.last_plan_time = -10
                    self.motion_monitor.reset()
            if now - self.last_localization_time > 2.0:
                left = right = 0.
                self.control_reason = 'LOCALIZATION_UNCERTAIN'
            emergency = self.emergency_required(ranges, pose) or (left == right == 0.0)
            if emergency:
                self.control_reason = self.stop_reason or (self.control_reason if
                    self.control_reason.startswith('RECOVERY_') or self.control_reason in ('VERIFY_APPLE', 'LOCALIZATION_UNCERTAIN')
                    else 'NO_SAFE_TRAJECTORY')
            if now - self.last_localization_time <= .75:
                escape = self.proximity_escape(pose, ranges, now)
                if escape is not None:
                    left, right = escape
                    emergency = False
                    self.control_reason = 'CLEARANCE_ESCAPE'
            self.set_wheels(left, right, emergency=emergency)
            self.update_display(pose, now)
            if now - self.last_status_time >= 5.0:
                known = int(np.count_nonzero(np.abs(self.grid.log_odds) >= 2))
                coverage = 100.0 * known / self.grid.log_odds.size
                print(
                    f"[STATUS] t={now - self.start_time:.0f}s state={self.state} "
                    f"pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) "
                    f"map={coverage:.1f}% apples={len(self.apple_detector.confirmed)}/2 "
                    f"visited={len(self.apple_detector.visited)}/2 "
                    f"people={len(self.people.tracks)} control={self.control_reason}"
                )
                self.save_results(pose)
                self.last_status_time = now


if __name__ == "__main__":
    mission = MissionController()
    try:
        mission.run()
    finally:
        mission.set_wheels(0, 0, emergency=True)
