"""Short-lived floor obstacles from the existing monocular camera.

Synthetic ranges are for collision control only, never LiDAR localization/SLAM.
"""
import math
import numpy as np


def floor_obstacle(box, label, pose, focal, width, height):
    x1, y1, x2, y2 = box
    if (x2 <= x1 or y2 <= y1 or y2 <= height / 2 + 6
            or y2 >= height - 2 or x1 <= 1 or x2 >= width - 1):
        return None
    # A clipped box does not expose the floor contact point. Retain an earlier
    # valid track instead of inventing an extremely close obstacle from the crop.
    # Existing camera: extension z=.153 plus local z=-.08 => .073m.
    forward = .073 * focal / (y2 - height / 2)
    if not .10 <= forward <= 2.5:
        return None
    lateral = (width / 2 - (x1 + x2) / 2) * forward / focal
    radius = min(.35, max(.05, (x2 - x1) * forward / focal / 2)) + .03
    if label == 'cat':
        radius = max(radius, .18)
    return dict(x=pose[0] + (forward + .02) * math.cos(pose[2]) - lateral * math.sin(pose[2]),
                y=pose[1] + (forward + .02) * math.sin(pose[2]) + lateral * math.cos(pose[2]),
                radius=radius, label=label)


class LowObstacleMap:
    def __init__(self):
        self.tracks = []

    def active(self, now):
        self.tracks = [t for t in self.tracks if now - t['seen'] < (1.5 if t['label'] == 'cat' else 20.)]
        return self.tracks

    def update(self, observations, now):
        self.active(now)
        used = set()
        for obs in observations:
            matches = [(math.hypot(obs['x'] - t['x'], obs['y'] - t['y']), i, t)
                       for i, t in enumerate(self.tracks) if i not in used and t['label'] == obs['label']]
            best = min(matches, default=None, key=lambda value: value[0])
            if best is not None and best[0] < .35:
                _, index, track = best
                used.add(index)
                track.update(obs, seen=now)
            else:
                self.tracks.append(dict(obs, seen=now))
                used.add(len(self.tracks) - 1)

    def control_ranges(self, ranges, pose, now):
        result = np.array(ranges, dtype=float)
        # Do not turn sensor failures into apparently valid measurements.
        usable = np.isposinf(result) | (np.isfinite(result) & (result > .03))
        bearings = math.pi - 2 * math.pi * np.arange(len(result)) / len(result)
        for obj in self.active(now):
            dx, dy = obj['x'] - pose[0], obj['y'] - pose[1]
            x = dx * math.cos(pose[2]) + dy * math.sin(pose[2]) + .03
            y = -dx * math.sin(pose[2]) + dy * math.cos(pose[2])
            radius = obj['radius']
            if x * x + y * y <= radius * radius:
                result[usable] = np.minimum(result[usable], .04)
                continue
            projection = x * np.cos(bearings) + y * np.sin(bearings)
            discriminant = radius * radius - (x * x + y * y - projection * projection)
            hits = (discriminant >= 0) & (projection > 0) & usable
            distances = projection[hits] - np.sqrt(discriminant[hits])
            result[hits] = np.minimum(result[hits], np.maximum(.04, distances))
        return result
