"""Rasterize a yaw-only cube and a rectangular field into color and depth."""

from __future__ import annotations

import numpy as np

from geom.transform import Intrinsics, project_points, world_to_camera


def intrinsics_for(width: int, height: int, model: str = "inverse_brown_conrady") -> Intrinsics:
    fx = (width / 2.0) / np.tan(np.deg2rad(87.0 / 2.0))
    fy = (height / 2.0) / np.tan(np.deg2rad(58.0 / 2.0))
    return Intrinsics(width, height, float(fx), float(fy), width / 2.0, height / 2.0, 0, 0, 0, 0, 0, model)


def _paint(points: np.ndarray, colors: np.ndarray, intr: Intrinsics, depth: np.ndarray, image: np.ndarray) -> None:
    if len(points) == 0:
        return
    pixels = project_points(points, intr)
    z = points[:, 2]
    ui = np.rint(pixels[:, 0]).astype(int)
    vi = np.rint(pixels[:, 1]).astype(int)
    inside = (z > 0.05) & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
    if not np.any(inside):
        return
    ui = ui[inside]
    vi = vi[inside]
    z = z[inside]
    cols = colors[inside]
    pix = vi * intr.width + ui
    order = np.lexsort((z, pix))
    _, first = np.unique(pix[order], return_index=True)
    chosen = order[first]
    flat = vi[chosen] * intr.width + ui[chosen]
    depth.ravel()[flat] = z[chosen]
    image.reshape(-1, 3)[flat] = cols[chosen]


def _face(origin, axis_a, axis_b, color, samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    aa = np.linspace(-0.5, 0.5, samples)
    bb = np.linspace(-0.5, 0.5, samples)
    grid_a, grid_b = np.meshgrid(aa, bb)
    pts = origin + grid_a.ravel()[:, None] * axis_a + grid_b.ravel()[:, None] * axis_b
    cols = np.repeat(np.array(color, dtype=np.uint8)[None, :], len(pts), axis=0)
    return pts, cols


def render_cube(
    intr: Intrinsics,
    bottom_center: np.ndarray,
    yaw: float,
    size: float = 0.6,
    ground_z: float = 2.0,
    top_bgr: tuple[int, int, int] = (255, 80, 40),
    front_bgr: tuple[int, int, int] | None = None,
    noise_m: float = 0.0,
    seed: int = 0,
    faces: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down camera: +Z points at the ground, world frame is the camera frame."""
    image = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    ground = []
    xs = np.linspace(-1.4, 1.4, 90)
    ys = np.linspace(-1.0, 1.0, 70)
    gx, gy = np.meshgrid(xs, ys)
    ground_pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, ground_z)])
    ground_col = np.full((len(ground_pts), 3), 40, dtype=np.uint8)
    _paint(ground_pts, ground_col, intr, depth, image)

    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = np.array([c, s, 0.0]) * size
    axis_y = np.array([-s, c, 0.0]) * size
    axis_z = np.array([0.0, 0.0, -size])
    center = np.asarray(bottom_center, dtype=float).copy()
    center[2] = ground_z
    top_origin = center + np.array([0.0, 0.0, -size])
    parts = []
    if faces in ("all", "top"):
        parts.append(_face(top_origin, axis_x, axis_y, top_bgr))
    side_color = front_bgr if front_bgr is not None else top_bgr
    if faces in ("all", "front"):
        parts.append(_face(center + axis_x * 0.5, axis_y, axis_z, side_color))
    if faces == "all":
        parts.append(_face(center - axis_x * 0.5, axis_y, axis_z, top_bgr))
        parts.append(_face(center + axis_y * 0.5, axis_x, axis_z, top_bgr))
        parts.append(_face(center - axis_y * 0.5, axis_x, axis_z, top_bgr))
    for pts, cols in parts:
        _paint(pts, cols, intr, depth, image)
    if noise_m > 0.0:
        rng = np.random.default_rng(seed)
        valid = np.isfinite(depth)
        depth[valid] += rng.normal(0.0, noise_m, size=int(valid.sum()))
        holes = rng.random(depth.shape) < 0.02
        depth[holes] = np.nan
    return image, depth


def render_field(
    intr: Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
    field_length: float,
    field_width: float,
    ground_samples: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((intr.height, intr.width, 3), 180, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    xs = np.linspace(-0.3, field_length + 0.3, ground_samples)
    ys = np.linspace(-0.3, field_width + 0.3, ground_samples)
    gx, gy = np.meshgrid(xs, ys)
    world = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cam = world_to_camera(world, rotation, translation)
    cols = np.full((len(cam), 3), 180, dtype=np.uint8)
    _paint(cam[cam[:, 2] > 0.05], cols[cam[:, 2] > 0.05], intr, depth, image)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [field_length, 0.0, 0.0],
            [field_length, field_width, 0.0],
            [0.0, field_width, 0.0],
        ]
    )
    cam_corners = world_to_camera(corners, rotation, translation)
    import cv2

    for i in range(4):
        pair = project_points(cam_corners[[i, (i + 1) % 4]], intr)
        p0 = tuple(np.rint(pair[0]).astype(int))
        p1 = tuple(np.rint(pair[1]).astype(int))
        cv2.line(image, p0, p1, (0, 0, 0), 4)
    return image, depth
