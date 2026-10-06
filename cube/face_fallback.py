"""看不清面时的补救。基本检测失败之后才调用，不改坐标链和内参。

掠射或只剩一条棱时，彩色区域认不出是哪一面。这里用贴地的立方体去贴可见点云，
yaw 仍只绕地面法向。贴不住就继续拒绝。这不是上表面正方形拟合，默认关闭。
"""

from __future__ import annotations

import numpy as np

from cube.detect import CubePose, color_mask
from geom.transform import deproject_pixels, fold_yaw_quarter, yaw_about_z
from ground.plane import Plane


# --- 点到立方体的距离 ---------------------------------------------------------
# 几何中心在方块中心，三轴分别是棱方向、另一条水平棱和地面法向。

def _box_distance(
    points: np.ndarray,
    center: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    up: np.ndarray,
    size: float,
) -> np.ndarray:
    # 变到方块自身坐标后，落在半边长盒子内的距离是 0，外面的是到表面的欧氏距离。
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
    cube_size: float = 0.2,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> CubePose | None:
    # 1. 取出贴地的彩色点，拟合可见面的水平法向。法向指向相机。
    # 这里仍用颜色块本身。主检测才会做轮廓筛选和腐蚀。
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
    # 比基本检测略宽：掠射时深度更吵，但仍丢掉明显钻到地下或高过方块的点。
    keep = (height > -0.08) & (height < cube_size + 0.1)
    if int(keep.sum()) < 30:
        return None
    points = points[keep]
    up = plane.normal
    # SVD 最小方向是点云最薄的方向，竖直面的法向大致如此。再去掉竖直分量，只留水平法向。
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal - (normal @ up) * up
    if np.linalg.norm(normal) < 1e-6:
        return None
    normal = normal / np.linalg.norm(normal)
    face_center = points.mean(axis=0)
    # 相机在原点。法向与面心同号说明它指向远离相机的一侧，翻过来。
    if normal @ face_center > 0.0:
        normal = -normal
    tangent = np.cross(up, normal)
    tangent = tangent / np.linalg.norm(tangent)
    half = cube_size * 0.5
    # 可见面在立方体外侧，中心要从面心沿法向退回半个边长，再压到地面。
    geometric = face_center - normal * half
    bottom0 = plane.project(geometric)[0]

    # 2. 立方体中心在面心内侧半个边长处。两个朝向、以及沿棱和法向的小幅滑动都试一遍，
    #    取得分最高且足够多的点落在盒子附近的那个。yaw 按 90° 对称折叠。
    best_score = -1.0
    best_bottom = bottom0
    best_tangent = tangent
    # 面心有偏，所以在 ±4 cm 内滑动。两个符号对应“体在法向哪一侧”。
    shifts = np.linspace(-0.04, 0.04, 5)
    for sign in (1.0, -1.0):
        base = face_center + sign * normal * half
        base = plane.project(base)[0]
        edge = tangent if sign < 0 else -tangent
        for along in shifts:
            for across in shifts:
                bottom = base + along * edge + across * normal
                # 贴合时用几何中心量点到盒子的距离。返回的仍是底面中心，半高由入口统一加。
                center = bottom + up * half
                dist = _box_distance(points, center, edge, np.cross(up, edge), up, cube_size)
                # 4 cm 内算贴住。比例太低说明这不是当前边长的贴地方块。
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
