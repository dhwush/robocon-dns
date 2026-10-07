"""面能看清时的方块位姿。

位置取地面足迹，不拟合上表面。上表面 RMS、12° 法向和 6 mm 正方形残差在这里没有平面可算。
丢帧只看足迹：面积比例、长边相对边长 12%、短边是否还像一条棱。
看不清一整块面时返回 None，不在这里做贴地模型补救。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from geom.transform import deproject_pixels, fold_yaw_quarter, yaw_about_z
from ground.plane import Plane


# --- 输出 ---------------------------------------------------------------------
# bottom_center 在相机系，yaw 已变到世界系并只保留绕 Z 的角。

@dataclass
class CubePose:
    bottom_center: np.ndarray
    yaw: float
    edge_direction: np.ndarray
    footprint_fraction: float
    front_color: bool
    inliers: int = 0
    corners: np.ndarray | None = None


# --- 颜色分割 -----------------------------------------------------------------
# 开运算去掉孤立色点，闭运算补小洞。红色在色相环两端，配置里写成两段再并起来。
# 实机方块上下两半是红和蓝，哪一面在上都可以，两色并进同一块掩膜。水平分界绕四周，不标成正面，yaw 仍按 90° 对称。
# 地面漆成红或蓝时，贴地 3.5 cm 以内的颜色点稍后按高度丢掉，不靠把色相收窄来区分。

# 贴地同色时，离地不足这条的点视为地面。侧面最底下会少一条，顶面和其余侧面还在。
GROUND_CLEARANCE_M = 0.035

# 足迹只收实测顶面往下这一条以内的点。侧面还在掩膜里，但不参与矩形。
TOP_LAYER_BELOW_PEAK_M = 0.04


def _scale_for_dim_light(hsv: np.ndarray) -> float:
    # 整幅中值亮度低于约 110 时，按比例放宽饱和度和亮度下限，暗处的红蓝仍能进掩膜。
    # 不再低于颜色函数里的保底，灰地面饱和度接近 0，不会因此被收进来。
    sample = hsv[::8, ::8, 2]
    if sample.size == 0:
        return 1.0
    median_v = float(np.median(sample))
    return float(np.clip(median_v / 110.0, 0.45, 1.0))


def color_mask(color_bgr: np.ndarray, colors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    scale = _scale_for_dim_light(hsv)
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    front = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for spec in colors:
        low = np.array(spec["hsv_low"], dtype=float)
        low[1] = np.clip(low[1] * scale, 30, 255)
        low[2] = np.clip(low[2] * scale, 20, 255)
        low = low.astype(np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        part = cv2.inRange(hsv, low, high)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, kernel)
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, kernel)
        mask = cv2.bitwise_or(mask, part)
        if spec.get("front"):
            front = cv2.bitwise_or(front, part)
    # 红蓝接缝大约十个像素。哪一面在上都能并成一块，缝留在内部，不把轮廓拉进面里。
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge)
    front = cv2.bitwise_and(front, mask)
    return mask, front


def _valid_depth(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # 工作距离内的有限深度才参与选块。NaN 和超距点不能拿来判断这块是不是方块。
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def _keep_above_ground(
    mask: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    # 明显高于地面的颜色点留下。贴地的底边按图像上的邻域留在方块上，
    # 不按地面投影去裁，否则侧面深度偏近时底边会从轮廓上消失。
    # 铺开的同色地面离这块面远，进不了这一圈。红面或蓝面在上都不看色相。
    ys, xs = np.nonzero(mask)
    kept = np.zeros(mask.shape, dtype=np.uint8)
    if len(xs) == 0:
        return kept
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < 10:
        return kept
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
    height = plane.signed_distance(points)
    elevated = (height > GROUND_CLEARANCE_M) & (height < cube_size + 0.08)
    if int(elevated.sum()) < 10:
        return kept
    body = np.zeros(mask.shape, dtype=np.uint8)
    body[ys[good][elevated], xs[good][elevated]] = 255
    z_med = float(np.median(z[good][elevated]))
    radius = int(np.ceil(1.5 * GROUND_CLEARANCE_M * float(intr.fx) / max(z_med, 1e-3)))
    radius = int(np.clip(radius, 6, 64))
    near = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    keep = elevated.copy()
    contact = (height > -0.02) & (height <= GROUND_CLEARANCE_M)
    keep[contact & (near[ys[good], xs[good]] > 0)] = True
    kept[ys[good][keep], xs[good][keep]] = 255
    # 面上没有深度的颜色点也留下，洞才不会把轮廓拉进去。
    color_near = cv2.bitwise_and(mask, near)
    has_depth = np.zeros(mask.shape, dtype=bool)
    has_depth[ys[good], xs[good]] = True
    kept[(color_near > 0) & ~has_depth] = 255
    return kept


def _span_long_side(component: np.ndarray, depth: np.ndarray, intr, plane: Plane) -> float | None:
    # 抽样量这块颜色在地面上的长边，用来丢掉比方块大一圈的背景。
    ys, xs = np.nonzero(component)
    if len(xs) < 30:
        return None
    step = max(len(xs) // 600, 1)
    z = depth[ys[::step], xs[::step]]
    good = np.isfinite(z)
    if int(good.sum()) < 20:
        return None
    points = deproject_pixels(
        np.column_stack([xs[::step][good], ys[::step][good]]),
        z[good],
        intr,
    )
    _coords, _projected, rect = _footprint(points, plane)
    (_center, (width, height_rect), _angle) = rect
    return float(max(width, height_rect))


def _select_component(
    mask: np.ndarray,
    front: np.ndarray,
    depth: np.ndarray,
    zmin: float,
    zmax: float,
    plane: Plane | None = None,
    intr=None,
    cube_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray] | None:
    """只留一块轮廓：面积在画面的 0.4%–40%，且深度中值落在工作距离内。

    斜视时顶面和正面之间会空出大约十个像素。先膨胀再取轮廓，把同一方块的面连起来。
    轮廓内部仍只用原来的颜色像素。有地面时，在长边没有大过方块太多的轮廓里，
    选长边最接近边长的一块，避免背景里另一块红蓝占满掩膜。
    """
    valid = _valid_depth(depth, zmin, zmax)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    linked = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_err = np.inf
    for contour in ranked:
        area = float(cv2.contourArea(contour))
        if area < 0.004 * image_area or area > 0.40 * image_area:
            continue
        filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        component = cv2.bitwise_and(mask, filled)
        # 深度只用来判断这块是不是方块。面上的深度洞不从掩膜里挖掉，轮廓才留在外沿。
        measured = cv2.bitwise_and(component, valid.astype(np.uint8) * 255)
        ys, xs = np.nonzero(measured)
        if len(xs) == 0:
            continue
        samples = depth[ys, xs]
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0 or not (zmin <= float(np.median(samples)) <= zmax):
            continue
        chosen = component, cv2.bitwise_and(front, component)
        # 没有地面时维持旧行为：取最大的一块。有地面时改选长边最接近方块的那块。
        if plane is None or intr is None:
            return chosen
        long_side = _span_long_side(component, depth, intr, plane)
        if long_side is None or long_side > cube_size * 2.5:
            continue
        err = abs(long_side - cube_size)
        if err < best_err:
            best_err = err
            best = chosen
    return best


def _erode_radius(distance_m: float) -> int:
    # 0.5 m 用 5 px，1.5 m 用 3 px。更远仍保留 3 px，避免把方块蚀穿。
    radius = int(round(5.0 - (distance_m - 0.5) * 2.0))
    return int(np.clip(radius, 3, 5))


def _drop_edge_jumps(depth: np.ndarray, mask: np.ndarray, thresh_m: float = 0.015) -> np.ndarray:
    """只去掉外轮廓附近深度跳变超过 15 mm 的点。面的内部保持连续。整幅图的 80 mm 飞点过滤在这之前已经做过。"""
    from camera.source import remove_flying_pixels

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    # 只在掩膜外接框里看邻域。框外的深度本来就不会拿来比较。
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, mask.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    # 外轮廓向内约 3 像素。红蓝接缝和斜侧面在内部，上面的跳变留下。
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rim = cv2.subtract(sub, cv2.erode(sub, rim_kernel))
    local = np.where(sub > 0, depth[y0:y1, x0:x1], np.nan)
    cleaned = remove_flying_pixels(local, thresh_m)
    updated = sub.copy()
    on_rim = rim > 0
    updated[on_rim] = np.where(np.isfinite(cleaned[on_rim]), 255, 0).astype(np.uint8)
    kept = mask.copy()
    kept[y0:y1, x0:x1] = updated
    return kept


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    # 封在面内部的洞补上。外轮廓本身不向外扩。
    if int(np.count_nonzero(mask)) == 0:
        return mask
    flood = mask.copy()
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def prepare_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    zmin: float = 0.3,
    zmax: float = 3.0,
    plane: Plane | None = None,
    intr=None,
    cube_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray] | None:
    # 先收成一块轮廓。深度跳变只去掉外轮廓那一圈，面的内部保持连续。
    # 有地面时先去掉铺开的贴地颜色，方块底边留在轮廓上。红面或蓝面在上用的是同一条路径。
    mask, front = color_mask(color_bgr, colors)
    if plane is not None and intr is not None:
        above = _keep_above_ground(mask, depth, intr, plane, cube_size, zmin, zmax)
        mask = cv2.bitwise_and(mask, above)
        front = cv2.bitwise_and(front, above)
    selected = _select_component(
        mask, front, depth, zmin, zmax, plane=plane, intr=intr, cube_size=cube_size
    )
    if selected is None:
        return None
    mask, front = selected
    ys, xs = np.nonzero(mask)
    finite = depth[ys, xs]
    finite = finite[np.isfinite(finite)]
    if len(finite) < 10:
        return None
    # 已经按高度去掉地面时不再腐蚀。腐蚀会吃掉落地边，足迹比真实方块小一圈。
    if plane is None or intr is None:
        distance = float(np.median(finite))
        radius = _erode_radius(distance)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.erode(mask, kernel)
        front = cv2.bitwise_and(front, mask)
    mask = _drop_edge_jumps(depth, mask, 0.015)
    mask = _fill_enclosed(mask)
    front = cv2.bitwise_and(front, mask)
    if int(np.count_nonzero(mask)) < 10:
        return None
    return mask, front


def _circular_median_hue(hues: np.ndarray) -> float:
    # OpenCV 色相是 0–180 的环。取环形中值，避免 0 和 179 被当成差很远。
    values = np.asarray(hues, dtype=float)
    if len(values) > 2000:
        values = values[:: max(len(values) // 2000, 1)]
    best = float(values[0])
    best_cost = np.inf
    for candidate in values:
        delta = (values - candidate + 90.0) % 180.0 - 90.0
        cost = float(np.median(np.abs(delta)))
        if cost < best_cost:
            best_cost = cost
            best = float(candidate)
    return best


def calibrate_hsv(
    frames: list[tuple[np.ndarray, np.ndarray]],
    colors: list[dict],
    zmin: float = 0.3,
    zmax: float = 3.0,
) -> list[dict]:
    """用约 30 帧里已经能分开的方块像素收紧色相。

    OpenCV 色相是 0–180，对应 0°–360°。8° 色相在这里是 4 个单位。
    饱和度、明度下限取这批像素的 5% 分位，并限制在不会把灰白地面收进来的范围。
    """
    updated: list[dict] = []
    for spec in colors:
        hues: list[np.ndarray] = []
        sats: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        for color_bgr, depth in frames:
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            part = cv2.inRange(hsv, low, high)
            part = cv2.bitwise_and(part, _valid_depth(depth, zmin, zmax).astype(np.uint8) * 255)
            ys, xs = np.nonzero(part)
            if len(xs) < 30:
                continue
            pix = hsv[ys, xs]
            hues.append(pix[:, 0].astype(float))
            sats.append(pix[:, 1].astype(float))
            vals.append(pix[:, 2].astype(float))
        if not hues:
            updated.append(dict(spec))
            continue
        hue = np.concatenate(hues)
        sat = np.concatenate(sats)
        val = np.concatenate(vals)
        center = _circular_median_hue(hue)
        delta = (hue - center + 90.0) % 180.0 - 90.0
        half = max(3.0 * float(np.std(delta)), 4.0)
        hue_low = int(np.clip(np.floor(center - half), 0, 179))
        hue_high = int(np.clip(np.ceil(center + half), hue_low + 1, 180))
        sat_low = int(np.clip(np.percentile(sat, 5), 40, 200))
        val_low = int(np.clip(np.percentile(val, 5), 30, 220))
        cloned = dict(spec)
        cloned["hsv_low"] = [hue_low, sat_low, val_low]
        cloned["hsv_high"] = [hue_high, 255, 255]
        updated.append(cloned)
    return updated


# --- 掩膜反投影 ---------------------------------------------------------------
# 只用腐蚀后仍有深度的彩色像素，按已知内参变到相机系。

def _mask_points(
    color_bgr,
    depth,
    intr,
    colors,
    zmin: float,
    zmax: float,
    plane: Plane,
    cube_size: float,
    prepared=None,
):
    if prepared is None:
        prepared = prepare_mask(
            color_bgr,
            depth,
            colors,
            zmin,
            zmax,
            plane=plane,
            intr=intr,
            cube_size=cube_size,
        )
    if prepared is None:
        return None
    mask, front = prepared
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    good = np.isfinite(z)
    if int(good.sum()) < 10:
        return None
    pixels = np.column_stack([xs[good], ys[good]])
    points = deproject_pixels(pixels, z[good], intr)
    front_flag = front[ys[good], xs[good]] > 0
    return points, front_flag, int(good.sum()), mask


# --- 地面足迹 -----------------------------------------------------------------
# 点沿地面法向压到平面上，再用最小外接矩形量水平跨度。
# 矩形只用接近顶面的一层。侧面深度偏近时会在靠近相机的一侧铺开，不拿来撑矩形。
# 不拟合上表面，因此没有上表面 RMS、12° 法向和 6 mm 正方形残差。
# 长边、短边、面积三道门限仍然把不合格的足迹丢掉。

def _footprint(points: np.ndarray, plane: Plane) -> tuple[np.ndarray, np.ndarray, tuple]:
    # 投影到地面后，顶面仍是约 0.2 m 的正方形；只看见一个竖直面时会塌成一条窄带。
    projected = plane.project(points)
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    coords = np.column_stack(
        [
            (projected - origin) @ axis_x,
            (projected - origin) @ axis_y,
        ]
    )
    # minAreaRect 给出中心、长宽和一条边的角度，单位与坐标相同（米、度）。
    rect = cv2.minAreaRect(coords.astype(np.float32))
    return coords, projected, rect


def detect_visible_cube(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.2,
    min_pixels: int = 400,
    min_footprint_fraction: float = 0.45,
    size_tolerance: float = 0.06,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 3.0,
    rejects: dict | None = None,
    prepared=None,
) -> CubePose | None:
    # 颜色足迹给出底面中心和 yaw。面积、长边或短边不合格就丢帧，不拟合上表面。
    def fail(reason: str) -> None:
        # 记一次拒绝原因。返回值是 None，调用方据此丢掉这一帧。
        if rejects is not None:
            rejects[reason] = int(rejects.get(reason, 0)) + 1
        return None

    # 1. 颜色点是否够多，并且贴在地面以上、高度接近方块边长。
    # 足迹只投接近顶面的一层。侧面留在掩膜里，不参与矩形。
    packed = _mask_points(
        color_bgr, depth, intr, colors, zmin, zmax, plane, cube_size, prepared=prepared
    )
    if packed is None:
        return fail("mask")
    points, front_flag, count, _mask = packed
    if count < min_pixels:
        return fail("inliers")
    height = plane.signed_distance(points)
    # 掩膜已经去掉了方块占地以外的地面。这里再丢掉钻到地下和高过方块的点，落地边保留。
    on_cube = (height > -0.02) & (height < cube_size + 0.08)
    if int(on_cube.sum()) < min_pixels:
        return fail("inliers")
    points = points[on_cube]
    front_flag = front_flag[on_cube]
    # 用 90 分位而不是最大值，避免一颗飞点把高度拉高。明显矮于半个边长就不是这个方块。
    height = plane.signed_distance(points)
    peak = float(np.percentile(height, 90))
    if abs(peak - cube_size) > size_tolerance + 0.05 and peak < cube_size * 0.5:
        return fail("height")
    # 顶面决定矩形。再往下的侧面不投到地面，偏近的深度就不会把框拉向相机。
    top = height >= peak - TOP_LAYER_BELOW_PEAK_M
    if int(top.sum()) < 3:
        return fail("height")
    # 2. 足迹要接近 0.2 m × 0.2 m。太扁说明只剩一条棱，这一帧拒绝。
    coords, _projected, rect = _footprint(points[top], plane)
    front_flag = front_flag[top]
    (center_u, center_v), (width, height_rect), angle_deg = rect
    long_side = max(width, height_rect)
    short_side = min(width, height_rect)
    area = float(width * height_rect)
    full_area = cube_size * cube_size
    fraction = area / full_area
    if fraction < min_footprint_fraction:
        return fail("footprint_area")
    # 长边相对标称边长超过 12% 就丢帧。size_tolerance 更小时再收紧，不会把 12% 放宽。
    side_limit = min(float(size_tolerance), 0.12 * cube_size)
    if abs(long_side - cube_size) > side_limit:
        return fail("footprint_long")
    # 短边太窄说明只剩一条棱，这一帧丢掉。
    if short_side < cube_size * 0.45:
        return fail("footprint_short")
    # 3. 底面中心给位置。能分出正面颜色时，用颜色重心相对足迹中心的方向当 yaw；
    #    否则用矩形棱的方向，并折进 90° 周期。
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    # 矩形中心已经在平面坐标里，抬回三维就是底面中心，不是点云重心。
    bottom = origin + center_u * axis_x + center_v * axis_y
    box = cv2.boxPoints(rect)
    corners = origin + box[:, 0:1] * axis_x + box[:, 1:2] * axis_y
    angle = np.deg2rad(angle_deg)
    edge_2d = np.array([np.cos(angle), np.sin(angle)])
    front_ratio = float(front_flag.mean())
    # 正面像素太少或几乎占满（整块都是同一颜色）时，分不出哪条棱朝前。
    front_seen = 0.08 < front_ratio < 0.85
    if front_seen:
        # 正面颜色的重心相对足迹中心的偏移，就是该面法向在地面上的投影。
        shift = coords[front_flag].mean(axis=0) - coords.mean(axis=0)
        if np.linalg.norm(shift) > 1e-4:
            edge_2d = shift / np.linalg.norm(shift)
    edge = edge_2d[0] * axis_x + edge_2d[1] * axis_y
    edge = edge / np.linalg.norm(edge)
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    # 棱方向只旋转、不平移。yaw 取它在世界 XY 上相对 X 轴的角。
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
        inliers=count,
        corners=corners,
    )
