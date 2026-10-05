"""Ground plane with a gravity check. The cube is constrained to yaw only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ self.normal - self.offset

    def project(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist = self.signed_distance(pts)
        return pts - dist[:, None] * self.normal

    def origin(self) -> np.ndarray:
        return self.normal * self.offset

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        up = self.normal
        helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_x = np.cross(helper, up)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(up, axis_x)
        return axis_x, axis_y


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = float(normal @ points[0])
    return normal, offset


def _orient_up(normal: np.ndarray, offset: float, gravity: np.ndarray) -> tuple[np.ndarray, float, float]:
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    if normal @ up < 0.0:
        normal = -normal
        offset = -offset
    cosine = float(np.clip(normal @ up, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return normal, offset, angle


def _refit(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = float(normal @ center)
    return normal, offset


def fit_ground(
    points: np.ndarray,
    gravity: np.ndarray,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
    iterations: int = 200,
    seed: int = 0,
) -> Plane | None:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    rng = np.random.default_rng(seed)
    n = len(pts)
    candidates: list[tuple[float, np.ndarray, float, np.ndarray]] = []
    for _ in range(iterations):
        choice = rng.choice(n, size=3, replace=False)
        fitted = _plane_from_three(pts[choice])
        if fitted is None:
            continue
        normal, offset = fitted
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle > max_angle_deg:
            continue
        inliers = np.abs(pts @ normal - offset) < thresh
        count = int(inliers.sum())
        if count < max(30, int(0.05 * n)):
            continue
        height = float(np.median(pts[inliers] @ up))
        candidates.append((height, inliers))
    if not candidates:
        return None
    lowest = min(item[0] for item in candidates)
    near_floor = [item for item in candidates if item[0] <= lowest + 0.05]
    near_floor.sort(key=lambda item: int(item[1].sum()), reverse=True)
    for _height, inliers in near_floor:
        normal, offset = _refit(pts[inliers])
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle <= max_angle_deg:
            return Plane(normal=normal, offset=offset)
    return None


def sample_points(depth: np.ndarray, intr, stride: int = 8) -> np.ndarray:
    from geom.transform import deproject_pixels

    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us]
    good = np.isfinite(z)
    if not np.any(good):
        return np.zeros((0, 3))
    pixels = np.column_stack([us[good].ravel(), vs[good].ravel()])
    return deproject_pixels(pixels, z[good].ravel(), intr)
