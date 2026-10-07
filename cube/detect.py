"""面能看清时的方块位姿。

位置取地面足迹，不拟合上表面。上表面 RMS、12° 法向和 6 mm 正方形残差在这里没有平面可算。
丢帧只看足迹：面积比例、长边相对边长 12%、短边是否还像一条棱。
看不清一整块面时返回 None，不在这里做贴地模型补救。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
# 红色在色相环两端，配置里写成两段再并起来。方块上下两半是红和蓝，哪一面在上都可以。
# 绿线是这块颜色的凸外沿，搜索范围按方块在该距离上的像素大小封住。
# 地面漆成红或蓝时，贴地 3.5 cm 以内的颜色点按高度丢掉，不靠把色相收窄来区分。

# 贴地同色时，离地不足这条的点视为地面。侧面最底下贴着方块的一条留下。
GROUND_CLEARANCE_M = 0.035

# 足迹只收实测顶面往下这一条以内的点。再宽就会带上侧面顶边，矩形被拉长。
TOP_LAYER_BELOW_PEAK_M = 0.025


def _scale_for_dim_light(hsv: np.ndarray) -> float:
    # 整幅中值亮度低于约 110 时，按比例放宽饱和度和亮度下限，暗处的红蓝仍能进掩膜。
    # 不再低于颜色函数里的保底，灰地面饱和度接近 0，不会因此被收进来。
    sample = hsv[::8, ::8, 2]
    if sample.size == 0:
        return 1.0
    median_v = float(np.median(sample))
    return float(np.clip(median_v / 110.0, 0.45, 1.0))


def _hue_range(hsv: np.ndarray, spec: dict, s_low: int, v_low: int, v_high: int = 255) -> np.ndarray:
    # 色相沿用配置。饱和度和亮度单独放宽，用来补被光冲淡或压暗的那一半。
    low = np.array([spec["hsv_low"][0], s_low, v_low], dtype=np.uint8)
    high = np.array([spec["hsv_high"][0], 255, v_high], dtype=np.uint8)
    return cv2.inRange(hsv, low, high)


def _scaled_odd(full_px: int, scale: float) -> int:
    # 半分辨率上核按比例缩小，并保持奇数。5→3，15→7。再小就固定 3，1 像素核不起作用。
    if scale >= 0.99:
        return full_px
    px = max(3, int(full_px * scale))
    if px % 2 == 0:
        px += 1
    return px


def _color_layers(
    color_bgr: np.ndarray, colors: list[dict], pixel_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 严格颜色用来认出方块在哪。放宽的只放松饱和度和亮度，色相仍是配置里的红和蓝。
    # 不在这里做大范围闭运算，否则旁边另一块颜色会被撑进同一条轮廓。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    dim = _scale_for_dim_light(hsv)
    strict = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    loose = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    front = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_scaled_odd(3, pixel_scale),) * 2)
    for spec in colors:
        low = np.array(spec["hsv_low"], dtype=float)
        low[1] = np.clip(low[1] * dim, 30, 255)
        low[2] = np.clip(low[2] * dim, 20, 255)
        low = low.astype(np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        part = cv2.inRange(hsv, low, high)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, speck)
        strict = cv2.bitwise_or(strict, part)
        loose = cv2.bitwise_or(loose, _hue_range(hsv, spec, 25, 25))
        if spec.get("front"):
            front = cv2.bitwise_or(front, part)
    loose = cv2.bitwise_or(loose, strict)
    return strict, loose, front


def color_mask(color_bgr: np.ndarray, colors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    strict, _loose, front = _color_layers(color_bgr, colors)
    return strict, cv2.bitwise_and(front, strict)


def _valid_depth(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # 工作距离内的有限深度才参与选块。NaN 和超距点不能拿来判断这块是不是方块。
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def _cube_silhouette(
    strict: np.ndarray,
    loose: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float = 1.0,
) -> np.ndarray | None:
    # 绿线是方块在图像上的外沿。立方体的投影是凸的，所以最后填凸包，边是直的。
    # 种子只用离地的严格颜色。面上深度稀的地方仍靠颜色补全，已知贴地的像素删掉，
    # 搜索范围按边长在这一距离上占多少像素封住，轮廓就不会铺满画面。
    valid = _valid_depth(depth, zmin, zmax)
    ys, xs = np.nonzero(strict)
    if len(xs) < 3:
        return None
    z = depth[ys, xs]
    good = valid[ys, xs]
    if int(good.sum()) < 3:
        return None
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
    height = plane.signed_distance(points)
    elevated = (height > GROUND_CLEARANCE_M) & (height < cube_size + 0.08)
    if int(elevated.sum()) < 3:
        return None
    elev = np.zeros(strict.shape, dtype=np.uint8)
    elev[ys[good][elevated], xs[good][elevated]] = 255
    # 同一面上的几个深度点并成一块。地面不是离地点，不会因此和方块并到一起。
    link_px = _scaled_odd(15, pixel_scale)
    blob = cv2.dilate(elev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link_px, link_px)))
    count, labels = cv2.connectedComponents(blob, connectivity=8)
    best_rank: tuple[float, float, float] | None = None
    best_seed: np.ndarray | None = None
    for lab in range(1, count):
        seed = (labels == lab) & (elev > 0)
        n_seed = int(seed.sum())
        if n_seed < 3:
            continue
        long_side = _span_long_side(seed.astype(np.uint8) * 255, depth, intr, plane)
        if long_side is not None and long_side > cube_size * 2.5:
            continue
        if long_side is None:
            rank = (1.0, 0.0, -float(n_seed))
        else:
            rank = (0.0, abs(long_side - cube_size), -float(n_seed))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_seed = seed
    if best_seed is None:
        return None
    z_med = float(np.median(depth[best_seed]))
    if not np.isfinite(z_med) or z_med <= 1e-3:
        return None
    reach = float(np.clip(cube_size * float(intr.fx) / z_med, 12.0 * pixel_scale, 420.0 * pixel_scale))
    inv = np.where(best_seed, np.uint8(0), np.uint8(255))
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    window = dist <= (2.2 * reach)
    color = (loose > 0) & window
    # 红蓝接缝大约十个像素。只用来把两半连成同一块，凸包的顶点仍是原来的颜色像素。
    link = cv2.dilate(
        color.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link_px, link_px)),
    )
    link = (link > 0) & window
    n_link, link_labels = cv2.connectedComponents(link.astype(np.uint8), connectivity=8)
    touch = np.unique(link_labels[best_seed])
    keep_label = np.zeros(n_link, dtype=bool)
    keep_label[touch] = True
    keep_label[0] = False
    chosen = keep_label[link_labels] & color
    ys2, xs2 = np.nonzero(chosen)
    if len(xs2) < 3:
        return None
    z2 = depth[ys2, xs2]
    good2 = valid[ys2, xs2]
    measured = np.full(len(xs2), np.nan)
    if int(good2.sum()) >= 3:
        pts = deproject_pixels(np.column_stack([xs2[good2], ys2[good2]]), z2[good2], intr)
        measured[good2] = plane.signed_distance(pts)
    body_h = (measured > GROUND_CLEARANCE_M) & (measured < cube_size + 0.08)
    if int(body_h.sum()) < 3:
        return None
    body = np.zeros(strict.shape, dtype=np.uint8)
    body[ys2[body_h], xs2[body_h]] = 255
    radius = int(np.ceil(GROUND_CLEARANCE_M * float(intr.fx) / z_med))
    radius = int(np.clip(radius, max(2, int(round(4 * pixel_scale))), max(4, int(round(20 * pixel_scale)))))
    near = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    contact = good2 & (measured > -0.02) & (measured <= GROUND_CLEARANCE_M) & (near[ys2, xs2] > 0)
    ground = np.zeros(strict.shape, dtype=np.uint8)
    on_ground = good2 & (measured <= GROUND_CLEARANCE_M) & ~contact
    if int(on_ground.sum()) > 0:
        ground[ys2[on_ground], xs2[on_ground]] = 255
    ground_px = _scaled_odd(5, pixel_scale)
    ground_near = cv2.dilate(ground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ground_px, ground_px)))
    unknown = ~good2
    # 没有深度的面留在颜色块里。紧挨已知地面的空洞是地面本身，不拿来撑外沿。
    keep_px = body_h | contact | (unknown & (ground_near[ys2, xs2] == 0))
    survive = np.zeros(strict.shape, dtype=np.uint8)
    survive[ys2[keep_px], xs2[keep_px]] = 255
    n_sur, sur_labels = cv2.connectedComponents(survive, connectivity=8)
    body_labs = np.unique(sur_labels[body > 0])
    keep_sur = np.zeros(n_sur, dtype=bool)
    keep_sur[body_labs] = True
    keep_sur[0] = False
    survive = np.where(keep_sur[sur_labels], np.uint8(255), np.uint8(0))
    sy, sx = np.nonzero(body)
    cx = float(np.median(sx))
    cy = float(np.median(sy))
    yy, xx = np.nonzero(survive)
    if len(xx) < 3:
        return None
    cap = 1.9 * reach
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= cap * cap
    if int(inside.sum()) < 3:
        return None
    pts_px = np.column_stack([xx[inside], yy[inside]]).astype(np.int32)
    hull = cv2.convexHull(pts_px)
    peri = float(cv2.arcLength(hull, True))
    approx = cv2.approxPolyDP(hull, max(2.0 * pixel_scale, 0.012 * peri), True)
    if len(approx) >= 3:
        hull = cv2.convexHull(approx)
    mask = np.zeros(strict.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


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
    link_px: int = 15,
) -> tuple[np.ndarray, np.ndarray] | None:
    """只留一块轮廓：面积在画面的 0.4%–40%，且深度中值落在工作距离内。

    斜视时顶面和正面之间会空出大约十个像素。先膨胀再取轮廓，把同一方块的面连起来。
    轮廓内部仍只用原来的颜色像素。有地面时，在长边没有大过方块太多的轮廓里，
    选长边最接近边长的一块，避免背景里另一块红蓝占满掩膜。
    """
    valid = _valid_depth(depth, zmin, zmax)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link_px, link_px))
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
    # 封在面内部的洞补上。种子必须落在背景上，掩膜碰到画面边缘时不能把整幅填满。
    if int(np.count_nonzero(mask)) == 0:
        return mask
    flood = mask.copy()
    height, width = mask.shape
    seed = None
    for x in range(width):
        if flood[0, x] == 0:
            seed = (x, 0)
            break
    if seed is None:
        for y in range(height):
            if flood[y, 0] == 0:
                seed = (0, y)
                break
    if seed is None:
        return mask
    pad = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, pad, seed, 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def _half_view(color_bgr: np.ndarray, depth: np.ndarray, intr):
    # 彩色用面积平均缩小，深度用最近邻，空洞先写成 0 再还原成无效，避免 NaN 被插值抹掉。
    height, width = color_bgr.shape[:2]
    small_w = max(1, width // 2)
    small_h = max(1, height // 2)
    color = cv2.resize(color_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    filled = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    small_depth = cv2.resize(filled, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    small_depth[small_depth <= 0] = np.nan
    sx = small_w / float(width)
    sy = small_h / float(height)
    small_intr = replace(
        intr,
        width=small_w,
        height=small_h,
        fx=float(intr.fx) * sx,
        fy=float(intr.fy) * sy,
        cx=float(intr.cx) * sx,
        cy=float(intr.cy) * sy,
    )
    return color, small_depth, small_intr, sx


def _mask_from_plane(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    plane: Plane,
    intr,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float,
):
    # 半分辨率上先分出凸外沿，再最近邻放大回深度图。像素计数仍按放大后的掩膜。
    if pixel_scale < 0.99:
        color, depth_s, intr_s, pixel_scale = _half_view(color_bgr, depth, intr)
    else:
        color, depth_s, intr_s = color_bgr, depth, intr
    strict, loose, front = _color_layers(color, colors, pixel_scale=pixel_scale)
    mask = _cube_silhouette(
        strict, loose, depth_s, intr_s, plane, cube_size, zmin, zmax, pixel_scale=pixel_scale
    )
    if mask is None:
        return None
    if mask.shape[:2] != color_bgr.shape[:2]:
        full = (color_bgr.shape[1], color_bgr.shape[0])
        mask = cv2.resize(mask, full, interpolation=cv2.INTER_NEAREST)
        front = cv2.resize(front, full, interpolation=cv2.INTER_NEAREST)
    front = cv2.bitwise_and(front, mask)
    if int(np.count_nonzero(mask)) < 10:
        return None
    return mask, front


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
    # 有地面时，绿线取方块颜色区域的凸外沿。红面或蓝面在上走同一条路径。
    # 先在半分辨率上找轮廓。找不到再回到全分辨率，平台挪动后的第一帧仍能找回方块。
    if plane is not None and intr is not None:
        if min(color_bgr.shape[:2]) >= 8:
            found = _mask_from_plane(
                color_bgr, depth, colors, plane, intr, cube_size, zmin, zmax, pixel_scale=0.5
            )
            if found is not None:
                return found
        return _mask_from_plane(
            color_bgr, depth, colors, plane, intr, cube_size, zmin, zmax, pixel_scale=1.0
        )
    strict, _loose, front = _color_layers(color_bgr, colors)
    mask = strict
    front = cv2.bitwise_and(front, mask)
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
