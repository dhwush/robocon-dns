"""Fit a ground-aligned cube when the face cannot be recognized.

This module is not used by the basic detector. Call it only after the
visible-face path has failed.
"""

from __future__ import annotations

import numpy as np

from cube.detect import CubePose, color_mask
from geom.transform import deproject_pixels, fold_yaw_quarter, yaw_about_z
from ground.plane import Plane


def _box_distance(
    points: np.ndarray,
    center: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    up: np.ndarray,
    size: float,
) -> np.ndarray:
    half = size * 0.5
    delta = points - center
    local = np.column_stack([delta @ axis_x, delta @ axis_y, delta @ up])
    outside = np.maximum(np.abs(local) - half, 0.0)
    return np.linalg.norm(outside, axis=1)


def fit_when_face_missing(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.6,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> CubePose | None:
    mask, _front = color_mask(color_bgr, colors)
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    z = depth[ys, xs]
    good = np.isfinite(z)
    if int(good.sum()) < 30:
        return None
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
    height = plane.signed_distance(points)
    keep = (height > -0.08) & (height < cube_size + 0.1)
    if int(keep.sum()) < 30:
        return None
    points = points[keep]
    up = plane.normal
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal - (normal @ up) * up
    if np.linalg.norm(normal) < 1e-6:
        return None
    normal = normal / np.linalg.norm(normal)
    face_center = points.mean(axis=0)
    if normal @ face_center > 0.0:
        normal = -normal
    tangent = np.cross(up, normal)
    tangent = tangent / np.linalg.norm(tangent)
    half = cube_size * 0.5
    geometric = face_center - normal * half
    bottom0 = plane.project(geometric)[0]

    best_score = -1.0
    best_bottom = bottom0
    best_tangent = tangent
    shifts = np.linspace(-0.04, 0.04, 5)
    for sign in (1.0, -1.0):
        base = face_center + sign * normal * half
        base = plane.project(base)[0]
        edge = tangent if sign < 0 else -tangent
        for along in shifts:
            for across in shifts:
                bottom = base + along * edge + across * normal
                center = bottom + up * half
                dist = _box_distance(points, center, edge, np.cross(up, edge), up, cube_size)
                score = float((dist < 0.04).mean())
                if score > best_score:
                    best_score = score
                    best_bottom = bottom
                    best_tangent = edge / np.linalg.norm(edge)
    if best_score < 0.35:
        return None
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    yaw = fold_yaw_quarter(yaw_about_z(rotation.T @ best_tangent))
    return CubePose(
        bottom_center=best_bottom,
        yaw=yaw,
        edge_direction=best_tangent,
        footprint_fraction=best_score,
        front_color=False,
    )
