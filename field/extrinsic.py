"""用矩形场地的边求相机外参。

世界系原点是靠近相机的那个角，X 沿长边，Y 沿短边，Z 向上。
基础要求的相机在世界原点，不走这里，直接用单位旋转和零平移。
提高要求只要求挪走相机后仍给出世界系位姿。安装方向和文档 2.3 的水平向前不一致时，
继续用这里求出的外参，不换成那组固定旋转，也不改用 AprilTag。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from geom.transform import deproject_pixels
from ground.plane import Plane


# --- 地面上的三维线段 ---------------------------------------------------------

@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray

    @property
    def direction(self) -> np.ndarray:
        # 无向线段先给出一个单位方向，朝向在求角时再按“离开交点”翻转。
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


# --- 由一个角求外参 -----------------------------------------------------------
# 两条边的交点是原点在相机系中的位置 t。
# 较长的边对齐世界 X，较短的边对齐世界 Y；Z 由叉乘得到，并翻到与地面法向同侧。

def identity_extrinsic() -> tuple[np.ndarray, np.ndarray]:
    # 基础要求：相机固定在世界原点。单位阵表示世界系就是相机系，不是文档 2.3 的 R_WC。
    return np.eye(3), np.zeros(3)


def _intersect(seg_a: Segment, seg_b: Segment) -> np.ndarray | None:
    # 解 p_a + a d_a = p_b + b d_b。两条线不共面时两端点对不上，超过 8 cm 就不是同一个角。
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
    # 线段中点在角的哪一侧，方向就指向哪一侧，也就是指向场地内部。
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
    # 矩形邻边应接近 90°。夹角小于 70° 的是同一条边的碎段，不是角。
    cosine = float(np.clip(abs(dir_a @ dir_b), 0.0, 1.0))
    if np.degrees(np.arccos(cosine)) < 70.0:
        return None
    # 用可见长度判断哪条是长边。边被裁切时长度偏短，后面还会用长度误差再筛一次。
    err_a = abs(edge_a.length - field_length) + abs(edge_b.length - field_width)
    err_b = abs(edge_b.length - field_length) + abs(edge_a.length - field_width)
    if err_a <= err_b:
        axis_x, axis_y = dir_a, dir_b
    else:
        axis_x, axis_y = dir_b, dir_a
    # R 的三列是世界轴在相机系中的方向，t 是角点。X_c = R X_w + t。
    axis_z = np.cross(axis_x, axis_y)
    norm = np.linalg.norm(axis_z)
    if norm < 1e-8:
        return None
    axis_z = axis_z / norm
    # 叉乘可能朝下。翻 Y 和 Z，让 Z 与地面向上的法向一致，X 仍沿长边。
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
    """取离相机更近的那个场地角。

    矩形对角点会得到两套差 180° 的坐标。世界原点约定在长边和短边靠近相机的一角，
    标定始终从这一侧做，原点才不会跳到对角。
    """
    # 矩形对边等长，对角点会得到两套相差 180° 的坐标。取相机系里更近的那个角当原点。
    candidates: list[tuple[float, tuple[np.ndarray, np.ndarray]]] = []
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            pose = pose_from_corner(first, second, plane, field_length, field_width)
            if pose is None:
                continue
            _rotation, translation = pose
            # 角点在相机后方说明射线交点不可用。
            if translation[2] <= 0.05:
                continue
            length_err = min(
                abs(first.length - field_length) + abs(second.length - field_width),
                abs(second.length - field_length) + abs(first.length - field_width),
            )
            # 长度差太大就不是这对已知边长，可能是地面上的别的直线。
            if length_err > 0.45 * (field_length + field_width):
                continue
            candidates.append((float(translation[2]), pose))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


# --- 从图像提取场地边 ---------------------------------------------------------
# Canny + 霍夫得到像素线段，再用已知内参把端点射线交到地面上。
# 共线的碎段合并，这样被画面裁断的边仍能参与求角。

def _pixel_on_plane(u: float, v: float, intr, plane: Plane) -> np.ndarray | None:
    # 用深度 1 取出射线方向。与平面 normal·(s·ray) = offset 相交，s 必须为正。
    ray = deproject_pixels(np.array([[u, v]]), np.array([1.0]), intr)[0]
    denom = float(plane.normal @ ray)
    if abs(denom) < 1e-8:
        return None
    scale = plane.offset / denom
    if scale <= 0.0:
        return None
    return ray * scale


def segments_from_image(color_bgr: np.ndarray, intr, plane: Plane) -> list[Segment]:
    # 场地边在图像上仍是直线。先提边缘，再把端点落到已经拟合好的地面上。
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
        # 短于 15 cm 的是噪声边缘，不是场地边。
        if seg.length < 0.15:
            continue
        segments.append(seg)
    return _merge_collinear(segments)


def _merge_collinear(segments: list[Segment], angle_deg: float = 12.0, gap: float = 0.15) -> list[Segment]:
    # 霍夫常把一条边切成几段。方向差小于 12°、横向距离小于 15 cm 的并成一条。
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
            # 中点到种子直线的横向距离。平行但相距一条场地宽的边不能合并。
            delta = other.midpoint() - seed.midpoint()
            lateral = np.linalg.norm(delta - (delta @ direction) * direction)
            if lateral > gap:
                rest.append(other)
                continue
            group.append(other)
        unused = rest
        # 合并后的端点取沿直线投影的最小值和最大值，恢复被裁断的长度。
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


# --- 外参缓存 -----------------------------------------------------------------
# 相机不动、只推平台时复用这份 R、t，不必每帧都看见整圈场地边。

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
