"""Camera extrinsic from the rectangular field border."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from geom.transform import deproject_pixels
from ground.plane import Plane


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray

    @property
    def direction(self) -> np.ndarray:
        delta = self.p1 - self.p0
        norm = np.linalg.norm(delta)
        if norm < 1e-8:
            return np.zeros(3)
        return delta / norm

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)


def identity_extrinsic() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3), np.zeros(3)


def _intersect(seg_a: Segment, seg_b: Segment) -> np.ndarray | None:
    direction_a = seg_a.direction
    direction_b = seg_b.direction
    if np.linalg.norm(direction_a) < 1e-8 or np.linalg.norm(direction_b) < 1e-8:
        return None
    design = np.column_stack([direction_a, -direction_b])
    sol, _, _, _ = np.linalg.lstsq(design, seg_b.p0 - seg_a.p0, rcond=None)
    point = seg_a.p0 + sol[0] * direction_a
    other = seg_b.p0 + sol[1] * direction_b
    if np.linalg.norm(point - other) > 0.08:
        return None
    return point


def _away_from(corner: np.ndarray, segment: Segment) -> np.ndarray:
    direction = segment.direction
    if (segment.midpoint() - corner) @ direction < 0.0:
        direction = -direction
    return direction


def pose_from_corner(
    edge_a: Segment,
    edge_b: Segment,
    plane: Plane,
    field_length: float,
    field_width: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    corner = _intersect(edge_a, edge_b)
    if corner is None:
        return None
    dir_a = _away_from(corner, edge_a)
    dir_b = _away_from(corner, edge_b)
    cosine = float(np.clip(abs(dir_a @ dir_b), 0.0, 1.0))
    if np.degrees(np.arccos(cosine)) < 70.0:
        return None
    err_a = abs(edge_a.length - field_length) + abs(edge_b.length - field_width)
    err_b = abs(edge_b.length - field_length) + abs(edge_a.length - field_width)
    if err_a <= err_b:
        axis_x, axis_y = dir_a, dir_b
    else:
        axis_x, axis_y = dir_b, dir_a
    axis_z = np.cross(axis_x, axis_y)
    norm = np.linalg.norm(axis_z)
    if norm < 1e-8:
        return None
    axis_z = axis_z / norm
    if axis_z @ plane.normal < 0.0:
        axis_y = -axis_y
        axis_z = -axis_z
    rotation = np.column_stack([axis_x, axis_y, axis_z])
    return rotation, corner


def estimate_from_lines(
    segments: list[Segment],
    plane: Plane,
    field_length: float,
    field_width: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the field corner closest to the camera.

    A rectangle's opposite corner gives the same edge lengths with both axes
    reversed. The configured world origin is the near corner of the long and
    short edges, which stays put as long as calibration is done from that side.
    """
    candidates: list[tuple[float, tuple[np.ndarray, np.ndarray]]] = []
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            pose = pose_from_corner(first, second, plane, field_length, field_width)
            if pose is None:
                continue
            _rotation, translation = pose
            if translation[2] <= 0.05:
                continue
            length_err = min(
                abs(first.length - field_length) + abs(second.length - field_width),
                abs(second.length - field_length) + abs(first.length - field_width),
            )
            if length_err > 0.45 * (field_length + field_width):
                continue
            candidates.append((float(translation[2]), pose))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _pixel_on_plane(u: float, v: float, intr, plane: Plane) -> np.ndarray | None:
    ray = deproject_pixels(np.array([[u, v]]), np.array([1.0]), intr)[0]
    denom = float(plane.normal @ ray)
    if abs(denom) < 1e-8:
        return None
    scale = plane.offset / denom
    if scale <= 0.0:
        return None
    return ray * scale


def segments_from_image(color_bgr: np.ndarray, intr, plane: Plane) -> list[Segment]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=30, maxLineGap=12)
    segments: list[Segment] = []
    if lines is None:
        return segments
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        p0 = _pixel_on_plane(float(x1), float(y1), intr, plane)
        p1 = _pixel_on_plane(float(x2), float(y2), intr, plane)
        if p0 is None or p1 is None:
            continue
        seg = Segment(p0, p1)
        if seg.length < 0.15:
            continue
        segments.append(seg)
    return _merge_collinear(segments)


def _merge_collinear(segments: list[Segment], angle_deg: float = 12.0, gap: float = 0.15) -> list[Segment]:
    unused = segments[:]
    merged: list[Segment] = []
    while unused:
        seed = unused.pop()
        direction = seed.direction
        group = [seed]
        rest = []
        for other in unused:
            if abs(direction @ other.direction) < np.cos(np.deg2rad(angle_deg)):
                rest.append(other)
                continue
            # Distance from other's midpoint to the seed line.
            delta = other.midpoint() - seed.midpoint()
            lateral = np.linalg.norm(delta - (delta @ direction) * direction)
            if lateral > gap:
                rest.append(other)
                continue
            group.append(other)
        unused = rest
        points = np.vstack([np.stack([seg.p0, seg.p1]) for seg in group])
        along = points @ direction
        merged.append(Segment(points[np.argmin(along)], points[np.argmax(along)]))
    return merged


def calibrate_from_image(
    color_bgr: np.ndarray,
    intr,
    plane: Plane,
    field_length: float,
    field_width: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    segments = segments_from_image(color_bgr, intr, plane)
    return estimate_from_lines(segments, plane, field_length, field_width)


def save_extrinsic(path: str | Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    payload = {"R": np.asarray(rotation, dtype=float).tolist(), "t": np.asarray(translation, dtype=float).tolist()}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)


def load_extrinsic(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return np.array(payload["R"], dtype=float), np.array(payload["t"], dtype=float)
