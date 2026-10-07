"""地面平面。法向必须接近重力的反方向，从而把方块姿态收成只有 yaw。

方块点投影到这块地面上求足迹，不另拟合上表面。上表面 RMS、顶面与地面的 12° 夹角、
正方形残差都依赖顶面平面，这里不算。足迹的长边、短边和面积在方块检测里丢帧。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- 平面表示 -----------------------------------------------------------------
# 单位法向 normal 指向上，平面方程为 normal · x = offset。

@dataclass
class Plane:
    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        # 正值表示点在法向一侧。法向朝上时，方块上的点距离为正。
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ self.normal - self.offset

    def project(self, points: np.ndarray) -> np.ndarray:
        # 沿法向落到平面上，用来把可见点收成地面足迹。
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist = self.signed_distance(pts)
        return pts - dist[:, None] * self.normal

    def origin(self) -> np.ndarray:
        # 法向是单位向量时，normal * offset 正好落在平面上，当作平面坐标原点。
        return self.normal * self.offset

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        # 在平面内造一对正交基，供最小外接矩形使用。helper 避开与法向平行的轴。
        up = self.normal
        helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_x = np.cross(helper, up)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(up, axis_x)
        return axis_x, axis_y


# --- RANSAC 拟合 --------------------------------------------------------------
# 三点定一个平面，丢掉和重力夹角过大的假设。
# 方块顶面和地面平行，所以在通过重力检查的候选里取最低的一层，避免把顶面当成地面。

def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    # 三点叉乘得到法向。三点共线时叉乘接近 0，这个假设作废。
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
    # 重力向下，up 取反。法向如果朝下，连同 offset 一起翻过来，保证距离符号稳定。
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    if normal @ up < 0.0:
        normal = -normal
        offset = -offset
    cosine = float(np.clip(normal @ up, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return normal, offset, angle


def _refit(points: np.ndarray) -> tuple[np.ndarray, float]:
    # 内点再用 SVD 重拟合。最小奇异向量是法向，比三点叉乘稳。
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
    # 近似水平的候选里取最低一层。方块顶面和地面都水平，取最低的才不会把顶面当成地面。
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
        # 墙面、方块侧面和重力夹角大，直接丢掉。通过的只剩近似水平的面。
        if angle > max_angle_deg:
            continue
        inliers = np.abs(pts @ normal - offset) < thresh
        count = int(inliers.sum())
        if count < max(30, int(0.05 * n)):
            continue
        # 用内点高度而不是平面锚点。倾斜的假平面会把锚点抛到很远。
        height = float(np.median(pts[inliers] @ up))
        candidates.append((height, inliers))
    if not candidates:
        return None
    # 顶面和地面都水平。取最低的一层，5 cm 以内的候选里再选内点最多的。
    lowest = min(item[0] for item in candidates)
    near_floor = [item for item in candidates if item[0] <= lowest + 0.05]
    near_floor.sort(key=lambda item: int(item[1].sum()), reverse=True)
    for _height, inliers in near_floor:
        normal, offset = _refit(pts[inliers])
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle <= max_angle_deg:
            return Plane(normal=normal, offset=offset)
    return None


def plane_holds(
    points: np.ndarray,
    plane: Plane,
    gravity: np.ndarray,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
) -> bool:
    # 上一帧平面还贴着这块点云、法向仍朝上时，直接沿用，不再抽点和 SVD。
    # 夹角和内点比例与 fit_ground 相同。只看内点数会把已经拧斜的平面粘住。
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return False
    _normal, _offset, angle = _orient_up(np.asarray(plane.normal, dtype=float), float(plane.offset), gravity)
    if angle > max_angle_deg:
        return False
    count = int(np.count_nonzero(np.abs(pts @ plane.normal - plane.offset) < thresh))
    return count >= max(30, int(0.05 * len(pts)))


# --- 深度图抽样 ---------------------------------------------------------------
# 按已知内参把像素反投影成相机系点，供地面拟合使用。stride 用来减点。

def sample_points(depth: np.ndarray, intr, stride: int = 8) -> np.ndarray:
    from geom.transform import deproject_pixels

    # 隔 stride 个像素取一个，地面拟合不需要全分辨率。
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us]
    good = np.isfinite(z)
    if not np.any(good):
        return np.zeros((0, 3))
    pixels = np.column_stack([us[good].ravel(), vs[good].ravel()])
    return deproject_pixels(pixels, z[good].ravel(), intr)
