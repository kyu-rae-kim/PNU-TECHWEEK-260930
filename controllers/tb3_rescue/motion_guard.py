"""Independent LiDAR motion evidence and bounded, sensor-checked unwedging."""
import math
from collections import deque

import numpy as np


def angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clearance_escape(ranges, pose, people):
    """Slow straight retreat/advance only when every nearby return moves away.

    This handles the stop-threshold deadlock; never authorizes motion inside the
    15cm hard footprint or toward a person. Validate the next two seconds anew
    on every control step.
    """
    values = np.asarray(ranges, dtype=float)
    valid = np.isfinite(values) & (values > .03)
    if np.mean(valid | np.isposinf(values)) < .75 or not np.any(valid):
        return None
    bearings = math.pi - 2 * math.pi * np.arange(len(values))[valid] / len(values)
    points = np.column_stack((values[valid] * np.cos(bearings) - .03,
                              values[valid] * np.sin(bearings)))
    initial = np.linalg.norm(points, axis=1)
    if initial.min() < .15:
        return None
    near = initial < .25
    if not np.any(near):
        return None
    best = None
    for velocity in (-.04, .04):
        previous = initial
        safe = True
        for elapsed in np.linspace(.1, 2., 20):
            distances = np.linalg.norm(points - [velocity * elapsed, 0], axis=1)
            if distances.min() < .15 or np.any(distances[near] < previous[near] - .00001):
                safe = False
                break
            wx = pose[0] + velocity * elapsed * math.cos(pose[2])
            wy = pose[1] + velocity * elapsed * math.sin(pose[2])
            if any(math.hypot(p['x'] + p['vx'] * elapsed - wx,
                              p['y'] + p['vy'] * elapsed - wy) < .7 for p in people):
                safe = False
                break
            previous = distances
        gain = float(previous.min() - initial.min())
        if safe and gain > .01 and (best is None or gain > best[0]):
            best = gain, velocity
    return None if best is None else (best[1] / .033, best[1] / .033)


def scan_points(ranges, max_range):
    values = np.asarray(ranges, dtype=float)[::3]
    bearings = math.pi - 2 * math.pi * np.arange(0, len(ranges), 3) / len(ranges)
    valid = np.isfinite(values) & (values > .12) & (values < max_range - .08)
    usable = np.where(valid, values, 0.)
    points = np.column_stack((usable * np.cos(bearings) - .03, usable * np.sin(bearings)))
    # Normals must not span missing returns or depth discontinuities.
    previous, following = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    with np.errstate(invalid='ignore'):
        tangent = following - previous
        length = np.linalg.norm(tangent, axis=1)
        valid &= np.roll(valid, 1) & np.roll(valid, -1)
        valid &= (np.linalg.norm(points - previous, axis=1) < .45)
        valid &= (np.linalg.norm(points - following, axis=1) < .45) & (length > .015)
    normals = np.column_stack((-tangent[valid, 1], tangent[valid, 0])) / length[valid, None]
    return points[valid], normals


def match_scans(reference, current, seed=(0., 0., 0.)):
    """Point-to-line ICP: current base frame -> reference base frame.

    Reject poor overlap and unobservable geometry instead of claiming zero motion.
    Both stationary and odometry seeds are tried so spinning wheels cannot force
    the answer toward the encoder displacement.
    """
    target, normals = reference
    source, _ = current
    if min(len(source), len(target)) < 30:
        return None
    solutions = []
    for initial in ((0., 0., 0.), seed):
        transform = np.array(initial, dtype=float)
        for _ in range(12):
            c, s = math.cos(transform[2]), math.sin(transform[2])
            rotated = source @ np.array([[c, s], [-s, c]])
            moved = rotated + transform[:2]
            squared = np.sum((moved[:, None, :] - target[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(squared, axis=1)
            distances = squared[np.arange(len(source)), nearest]
            n = normals[nearest]
            residual = np.sum((moved - target[nearest]) * n, axis=1)
            keep = (distances < .28 ** 2) & (np.abs(residual) < .16)
            if np.count_nonzero(keep) < 30:
                break
            cutoff = max(.012, float(np.quantile(np.abs(residual[keep]), .8)))
            keep &= np.abs(residual) <= cutoff
            jacobian = np.column_stack((n, -n[:, 0] * rotated[:, 1] + n[:, 1] * rotated[:, 0]))[keep]
            # Parallel walls alone cannot constrain displacement along a corridor.
            information = jacobian.T @ jacobian / len(jacobian)
            scale = np.sqrt(np.maximum(np.diag(information), 1e-9))
            eigenvalues = np.linalg.eigvalsh(information / scale[:, None] / scale[None, :])
            normal_eigenvalues = np.linalg.eigvalsh(n[keep].T @ n[keep] / np.count_nonzero(keep))
            if eigenvalues[0] < .03 or normal_eigenvalues[0] < .035:
                break
            increment, _, _, _ = np.linalg.lstsq(jacobian, -residual[keep], rcond=None)
            if np.linalg.norm(increment[:2]) > .18 or abs(increment[2]) > .20:
                break
            transform += increment
            if np.linalg.norm(increment) < .0005:
                overlap = float(np.mean(keep))
                rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
                if overlap >= .55 and rmse < .035:
                    solutions.append((rmse, transform.copy(), overlap))
                break
    if not solutions:
        return None
    rmse, transform, overlap = min(solutions, key=lambda result: result[0])
    return dict(delta=transform, rmse=rmse, overlap=overlap)


class MotionMonitor:
    """Compare a two-second history of independent motion and applied commands."""
    def __init__(self):
        self.samples = deque()
        self.metrics = {}

    def reset(self):
        self.samples.clear()

    def update(self, now, dt, odometry, localized, commanded, valid):
        if not valid or dt <= 0 or dt > 1:
            self.reset()
            self.metrics = {'localization_valid': False, 'time_s': round(now, 3)}
            return False
        self.samples.append((now, dt, odometry, localized, commanded))
        while self.samples and now - self.samples[0][0] > 2.5:
            self.samples.popleft()
        duration = sum(row[1] for row in self.samples)
        odom, actual, command = (sum(row[i] for row in self.samples) for i in (2, 3, 4))
        self.metrics = dict(localization_valid=True, time_s=round(now, 3), window_s=round(duration, 3),
                            odometry_m=round(odom, 4), localized_m=round(actual, 4),
                            commanded_m=round(command, 4), error_m=round(odom - actual, 4))
        spinning = odom > .12 and odom - actual > .10 and actual < .3 * odom
        stalled = command > .16 and actual < .025 and odom < .04
        return duration >= 1.8 and command > .10 and (spinning or stalled)


class Recovery:
    """Brake -> back up <= 18cm -> turn ~25deg -> replan; at most three attempts."""
    def __init__(self):
        self.stage = None
        self.attempts = 0
        self.origin = None
        self.stage_time = 0.
        self.backup_pose = None
        self.turn_heading = 0.
        self.turn_sign = 1
        self.failed_areas = []
        self.cooldown_until = 0.
        self.last_reason = None

    @property
    def active(self):
        return self.stage is not None

    def start(self, pose, now):
        if self.active or now < self.cooldown_until:
            return False
        if self.origin is None or math.hypot(pose[0] - self.origin[0], pose[1] - self.origin[1]) > .65:
            self.attempts = 0
        exhausted = self.attempts >= 3
        if not exhausted:
            self.attempts += 1
        self.origin = tuple(pose)
        self.backup_pose = tuple(pose)
        self.stage_time = now
        self.stage = 'BLOCKED' if exhausted else 'BRAKE'
        self.last_reason = 'RECOVERY_ATTEMPT_LIMIT' if exhausted else 'ODOMETRY_LOCALIZATION_MISMATCH'
        self.failed_areas.append((pose[0] + .28 * math.cos(pose[2]),
                                  pose[1] + .28 * math.sin(pose[2]), now + 120))
        return True

    @staticmethod
    def safe(ranges, pose, people, velocity, omega):
        values = np.asarray(ranges, dtype=float)
        valid = np.isfinite(values) & (values > .03)
        if np.mean(valid | np.isposinf(values)) < .75:
            return False
        bearings = math.pi - 2 * math.pi * np.arange(len(values))[valid] / len(values)
        points = np.column_stack((values[valid] * np.cos(bearings) - .03,
                                  values[valid] * np.sin(bearings)))
        for elapsed in np.linspace(0, 1.2, 9):
            # Recovery commands are either straight retreat or in-place rotation.
            centre = np.array([velocity * elapsed, 0])
            if len(points) and np.min(np.linalg.norm(points - centre, axis=1)) < .18:
                return False
            wx = pose[0] + centre[0] * math.cos(pose[2])
            wy = pose[1] + centre[0] * math.sin(pose[2])
            for person in people:
                if math.hypot(person['x'] + person['vx'] * elapsed - wx,
                              person['y'] + person['vy'] * elapsed - wy) < .70:
                    return False
        return True

    def command(self, pose, ranges, people, now, localized_valid=True):
        if self.stage is None:
            return None
        if self.stage == 'BLOCKED':
            return (0., 0.)
        if not localized_valid:
            if now - self.stage_time > 3:
                self.stage = 'BLOCKED'
                self.last_reason = 'LOCALIZATION_UNCERTAIN'
            return (0., 0.)
        if self.stage == 'BRAKE':
            if now - self.stage_time < .3:
                return (0., 0.)
            self.stage, self.stage_time = 'BACKUP', now
            self.backup_pose = tuple(pose)
        if self.stage == 'BACKUP':
            travelled = math.hypot(pose[0] - self.backup_pose[0], pose[1] - self.backup_pose[1])
            if travelled < .18 and now - self.stage_time < 3.0:
                if self.safe(ranges, pose, people, -.06, 0):
                    return (-.06 / .033, -.06 / .033)
                self.last_reason = 'REAR_PATH_BLOCKED'
                self.stage = 'BLOCKED'
                return (0., 0.)
            if travelled < .04:
                self.stage = 'BLOCKED'
                self.last_reason = 'BACKUP_NO_PROGRESS'
                return (0., 0.)
            self.stage, self.stage_time = 'TURN', now
            self.turn_heading = pose[2]
            values = np.asarray(ranges, dtype=float)
            n = len(values)
            left = np.nanmedian(values[n // 8:3 * n // 8])
            right = np.nanmedian(values[5 * n // 8:7 * n // 8])
            self.turn_sign = 1 if left >= right else -1
        if self.stage == 'TURN':
            turned = abs(angle(pose[2] - self.turn_heading))
            if turned >= .44:
                self.stage = None
                self.cooldown_until = now + 3
                return (0., 0.)
            if now - self.stage_time > 2.0:
                self.stage = 'BLOCKED'
                self.last_reason = 'TURN_NO_PROGRESS'
                return (0., 0.)
            if self.safe(ranges, pose, people, 0, .6 * self.turn_sign):
                return (-1.45 * self.turn_sign, 1.45 * self.turn_sign)
            self.stage = 'BLOCKED'
            self.last_reason = 'TURN_PATH_BLOCKED'
        return (0., 0.)
