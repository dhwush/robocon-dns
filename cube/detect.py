"""Visible-face cube pose. Position is the bottom-face center, not the cloud centroid."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from geom.transform import deproject_pixels, fold_yaw_quarter, yaw_about_z
from ground.plane import Plane


@dataclass
class CubePose:
    bottom_center: np.ndarray
    yaw: float
    edge_direction: np.ndarray
    footprint_fraction: float
    front_color: bool


def color_mask(color_bgr: np.ndarray, colors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    front = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for spec in colors:
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        part = cv2.inRange(hsv, low, high)
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, kernel)
        mask = cv2.bitwise_or(mask, part)
        if spec.get("front"):
            front = cv2.bitwise_or(front, part)
    return mask, front


def _mask_points(color_bgr, depth, intr, colors):
    mask, front = color_mask(color_bgr, colors)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    z = depth[ys, xs]
    good = np.isfinite(z)
    if int(good.sum()) < 10:
        return None
    pixels = np.column_stack([xs[good], ys[good]])
    points = deproject_pixels(pixels, z[good], intr)
    front_flag = front[ys[good], xs[good]] > 0
    return points, front_flag, int(good.sum())


def _footprint(points: np.ndarray, plane: Plane) -> tuple[np.ndarray, np.ndarray, tuple]:
    projected = plane.project(points)
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    coords = np.column_stack(
        [
            (projected - origin) @ axis_x,
            (projected - origin) @ axis_y,
        ]
    )
    rect = cv2.minAreaRect(coords.astype(np.float32))
    return coords, projected, rect


def detect_visible_cube(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.6,
    min_pixels: int = 400,
    min_footprint_fraction: float = 0.45,
    size_tolerance: float = 0.18,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> CubePose | None:
    packed = _mask_points(color_bgr, depth, intr, colors)
    if packed is None:
        return None
    points, front_flag, count = packed
    if count < min_pixels:
        return None
    height = plane.signed_distance(points)
    # Up is +normal, so points above the ground have positive distance.
    on_cube = (height > -0.05) & (height < cube_size + 0.08)
    if int(on_cube.sum()) < min_pixels:
        return None
    points = points[on_cube]
    front_flag = front_flag[on_cube]
    peak = float(np.percentile(plane.signed_distance(points), 90))
    if abs(peak - cube_size) > size_tolerance + 0.05 and peak < cube_size * 0.5:
        return None
    coords, _projected, rect = _footprint(points, plane)
    (center_u, center_v), (width, height_rect), angle_deg = rect
    long_side = max(width, height_rect)
    short_side = min(width, height_rect)
    area = float(width * height_rect)
    full_area = cube_size * cube_size
    fraction = area / full_area
    if fraction < min_footprint_fraction:
        return None
    if abs(long_side - cube_size) > size_tolerance:
        return None
    if short_side < cube_size * 0.45:
        return None
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    bottom = origin + center_u * axis_x + center_v * axis_y
    angle = np.deg2rad(angle_deg)
    edge_2d = np.array([np.cos(angle), np.sin(angle)])
    front_ratio = float(front_flag.mean())
    front_seen = 0.08 < front_ratio < 0.85
    if front_seen:
        shift = coords[front_flag].mean(axis=0) - coords.mean(axis=0)
        if np.linalg.norm(shift) > 1e-4:
            edge_2d = shift / np.linalg.norm(shift)
    edge = edge_2d[0] * axis_x + edge_2d[1] * axis_y
    edge = edge / np.linalg.norm(edge)
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    edge_world = rotation.T @ edge
    yaw = yaw_about_z(edge_world)
    if not front_seen:
        yaw = fold_yaw_quarter(yaw)
    return CubePose(
        bottom_center=bottom,
        yaw=yaw,
        edge_direction=edge,
        footprint_fraction=fraction,
        front_color=front_seen,
    )
