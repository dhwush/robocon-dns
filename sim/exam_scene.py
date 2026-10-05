"""Solid-color exam scenes rendered with the published D435i color model.

Color frames are 8-bit BGR PNGs. Depth frames are 16-bit PNGs in millimeters;
a value of 0 means the pixel has no depth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from geom.transform import deproject_pixels, project_points, world_to_camera

ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "tests" / "output" / "scenes"
VIS_DIR = ROOT / "tests" / "output" / "vis"
INTRINSICS_PATH = ROOT / "config" / "d435i_1280x720.yaml"

GROUND_BGR = (170, 170, 170)
BLUE_BGR = (255, 40, 40)
YELLOW_BGR = (0, 220, 220)
RED_BGR = (40, 40, 220)
GREEN_BGR = (40, 160, 40)
WHITE_BGR = (235, 235, 235)


def clip_near(poly: np.ndarray, zmin: float = 0.25) -> np.ndarray:
    """Clip a convex polygon to the camera half-space z >= zmin."""
    points = np.asarray(poly, dtype=float)
    output: list[np.ndarray] = []
    count = len(points)
    for index in range(count):
        current = points[index]
        nxt = points[(index + 1) % count]
        current_in = current[2] >= zmin
        next_in = nxt[2] >= zmin
        if current_in and next_in:
            output.append(nxt)
        elif current_in and not next_in:
            output.append(_cross_z(current, nxt, zmin))
        elif not current_in and next_in:
            output.append(_cross_z(current, nxt, zmin))
            output.append(nxt)
    if len(output) < 3:
        return np.zeros((0, 3))
    return np.vstack(output)


def _cross_z(start: np.ndarray, end: np.ndarray, zmin: float) -> np.ndarray:
    delta = end[2] - start[2]
    scale = 0.0 if abs(delta) < 1e-9 else (zmin - start[2]) / delta
    return start + scale * (end - start)


def paint_polygon(
    color: np.ndarray,
    depth: np.ndarray,
    poly_cam: np.ndarray,
    bgr: tuple[int, int, int],
    intr,
) -> None:
    poly = clip_near(poly_cam)
    if len(poly) < 3 or np.any(poly[:, 2] <= 0.05):
        return
    pixels = np.rint(project_points(poly, intr)).astype(np.int32)
    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pixels, 255)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    rays = deproject_pixels(np.column_stack([xs, ys]), np.ones(len(xs)), intr)
    normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    denom = rays @ normal
    valid = np.abs(denom) > 1e-8
    scale = np.zeros(len(xs))
    scale[valid] = (float(poly[0] @ normal)) / denom[valid]
    current = depth[ys, xs]
    closer = valid & (scale > 0.05) & (~np.isfinite(current) | (scale < current))
    if not np.any(closer):
        return
    depth[ys[closer], xs[closer]] = scale[closer]
    color[ys[closer], xs[closer]] = bgr


def _quad(origin: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            origin - axis_a - axis_b,
            origin + axis_a - axis_b,
            origin + axis_a + axis_b,
            origin - axis_a + axis_b,
        ]
    )


def box_polygons(
    bottom: np.ndarray,
    yaw: float,
    size: float,
    height: float,
    top_bgr: tuple[int, int, int],
    front_bgr: tuple[int, int, int] | None = None,
    z_sign: float = 1.0,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Five faces. The front face is the +local-X side. z_sign > 0 points up."""
    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = np.array([c, s, 0.0]) * size
    axis_y = np.array([-s, c, 0.0]) * size
    axis_z = np.array([0.0, 0.0, z_sign * height])
    base = np.asarray(bottom, dtype=float)
    side = front_bgr if front_bgr is not None else top_bgr
    half_x = axis_x * 0.5
    half_y = axis_y * 0.5
    half_z = axis_z * 0.5
    faces = [
        (_quad(base + half_z, half_x, half_y), top_bgr),
        (_quad(base + half_x + half_z, half_y, half_z), side),
        (_quad(base - half_x + half_z, half_y, half_z), top_bgr),
        (_quad(base + half_y + half_z, half_x, half_z), top_bgr),
        (_quad(base - half_y + half_z, half_x, half_z), top_bgr),
    ]
    return faces


def blank(intr) -> tuple[np.ndarray, np.ndarray]:
    color = np.full((intr.height, intr.width, 3), 30, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    return color, depth


def add_depth_noise(depth: np.ndarray, noise_m: float, hole_fraction: float, seed: int) -> None:
    if noise_m <= 0.0 and hole_fraction <= 0.0:
        return
    rng = np.random.default_rng(seed)
    valid = np.isfinite(depth)
    if noise_m > 0.0:
        depth[valid] += rng.normal(0.0, noise_m, size=int(valid.sum()))
    if hole_fraction > 0.0:
        holes = valid & (rng.random(depth.shape) < hole_fraction)
        depth[holes] = np.nan


def draw_polyline(color: np.ndarray, intr, points_cam: np.ndarray, bgr: tuple[int, int, int], thickness: int) -> None:
    count = len(points_cam)
    for index in range(count):
        segment = _clip_edge(points_cam[index], points_cam[(index + 1) % count])
        if segment is None:
            continue
        pixels = np.rint(project_points(segment, intr)).astype(int)
        cv2.line(color, tuple(pixels[0]), tuple(pixels[1]), bgr, thickness, cv2.LINE_AA)


def _clip_edge(start: np.ndarray, end: np.ndarray, zmin: float = 0.25) -> np.ndarray | None:
    a_in = start[2] >= zmin
    b_in = end[2] >= zmin
    if a_in and b_in:
        return np.vstack([start, end])
    if not a_in and not b_in:
        return None
    hit = _cross_z(start, end, zmin)
    if a_in:
        return np.vstack([start, hit])
    return np.vstack([hit, end])


def look_at(camera_world: np.ndarray, target_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera axes: x right, y down, z forward. World z is up."""
    forward = np.asarray(target_world, dtype=float) - np.asarray(camera_world, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ np.asarray(camera_world, dtype=float)
    return rotation, translation


def render_nadir(
    intr,
    cubes: list[dict],
    distractors: list[dict],
    ground_z: float = 1.65,
    noise_m: float = 0.004,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down camera. The camera frame is the world frame, +Z toward the ground."""
    color, depth = blank(intr)
    ground = _quad(
        np.array([0.0, 0.0, ground_z]),
        np.array([1.15, 0.0, 0.0]),
        np.array([0.0, 0.85, 0.0]),
    )
    paint_polygon(color, depth, ground, GROUND_BGR, intr)
    for item in distractors:
        for poly, bgr in box_polygons(
            np.array([item["x"], item["y"], ground_z]),
            item["yaw"],
            item["size"],
            item["height"],
            item["color"],
            z_sign=-1.0,
        ):
            paint_polygon(color, depth, poly, bgr, intr)
    for item in cubes:
        for poly, bgr in box_polygons(
            np.array([item["x"], item["y"], ground_z]),
            item["yaw"],
            item["size"],
            item["height"],
            item["top"],
            item.get("front"),
            z_sign=-1.0,
        ):
            paint_polygon(color, depth, poly, bgr, intr)
    add_depth_noise(depth, noise_m, 0.01, seed)
    return color, depth


def render_field(
    intr,
    rotation: np.ndarray,
    translation: np.ndarray,
    cubes: list[dict],
    distractors: list[dict],
    field_length: float,
    field_width: float,
    noise_m: float = 0.004,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    color, depth = blank(intr)
    color[:] = GROUND_BGR
    ground = _quad(np.array([1.0, 0.6, 0.0]), np.array([2.4, 0.0, 0.0]), np.array([0.0, 1.8, 0.0]))
    paint_polygon(color, depth, world_to_camera(ground, rotation, translation), GROUND_BGR, intr)
    for item in distractors:
        _paint_box(color, depth, intr, rotation, translation, item, z_sign=1.0)
    for item in cubes:
        _paint_box(color, depth, intr, rotation, translation, item, z_sign=1.0)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [field_length, 0.0, 0.0],
            [field_length, field_width, 0.0],
            [0.0, field_width, 0.0],
        ]
    )
    draw_polyline(color, intr, world_to_camera(corners, rotation, translation), (0, 0, 0), 8)
    add_depth_noise(depth, noise_m, 0.01, seed)
    return color, depth


def _paint_box(color, depth, intr, rotation, translation, item: dict, z_sign: float) -> None:
    bottom = np.array([item["x"], item["y"], item.get("z", 0.0)], dtype=float)
    for poly, bgr in box_polygons(
        bottom,
        item["yaw"],
        item["size"],
        item["height"],
        item["top"],
        item.get("front"),
        z_sign=z_sign,
    ):
        paint_polygon(color, depth, world_to_camera(poly, rotation, translation), bgr, intr)


def save_rgbd(directory: Path, name: str, color: np.ndarray, depth: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"{name}_color.png"), color)
    millimeters = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    millimeters[valid] = np.clip(np.rint(depth[valid] * 1000.0), 1, 65535).astype(np.uint16)
    cv2.imwrite(str(directory / f"{name}_depth.png"), millimeters)


def load_rgbd(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    color = cv2.imread(str(directory / f"{name}_color.png"), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(directory / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
    if color is None or raw is None:
        raise FileNotFoundError(name)
    depth = raw.astype(float) / 1000.0
    depth[raw == 0] = np.nan
    return color, depth
