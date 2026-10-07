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
# 绿线只描成片的严格红蓝，阈值按配置走，不随整幅亮度放宽。放宽颜色只补大约 10 px 的接缝。
# 没有深度的点不拿来撑外沿，凸包也不把方块和旁边木板之间的地面填进掩膜。
# 地面漆成红或蓝时，贴地 3.5 cm 以内的颜色点按高度丢掉，不靠把色相收窄来区分。

# 贴地同色时，离地不足这条的点视为地面。侧面最底下贴着方块的一条留下。
GROUND_CLEARANCE_M = 0.035

# 足迹先收实测顶面往下 1 cm。这一层面积不够时，再放到 2.5 cm，不把上半截整面投进去。
TOP_LAYER_BELOW_PEAK_M = 0.010
FOOTPRINT_FALLBACK_BELOW_PEAK_M = 0.025


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
    # 严格颜色按配置里的 HSV 走，不再按整幅亮度把饱和度和明度放宽。
    # 曝光用 --light 锁定，红蓝应落在这组阈值里。放宽只比严格低一档，用来补接缝。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    strict = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    loose = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    front = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_scaled_odd(3, pixel_scale),) * 2)
    for spec in colors:
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        part = cv2.inRange(hsv, low, high)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, speck)
        strict = cv2.bitwise_or(strict, part)
        loose_s = max(50, int(low[1]) - 40)
        loose_v = 25
        loose = cv2.bitwise_or(loose, _hue_range(hsv, spec, loose_s, loose_v))
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
    seam_full_px: int = 11,
) -> np.ndarray | None:
    # 绿线沿严格红蓝走。长边和高度只做门限，通过的里面取像素最多的一块。
    # 远处深度把长边晃离 0.2 m 时，杂物不会因为碰巧更接近边长而盖过方块。
    # 放宽颜色只填红蓝接缝那大约 10 px。再往外一个边长是上限，木板和机器人进不来。
    packed = _elevated_maps(strict, depth, intr, plane, cube_size, zmin, zmax)
    if packed is None:
        return None
    elev, height_img = packed
    # 定位用大约 11 px 把红蓝接上。全分辨率重描时用 3 px，避免把棱向外胀开。
    seam_px = _scaled_odd(seam_full_px, pixel_scale)
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_px, seam_px))
    blob = cv2.dilate(elev, seam_kernel)
    seeds = _top_seeds(blob, elev, depth, height_img, intr, plane, cube_size, limit=1)
    if not seeds:
        return None
    best_seed = seeds[0]
    z_med = float(np.median(depth[best_seed]))
    if not np.isfinite(z_med) or z_med <= 1e-3:
        return None
    # 搜索半径是一个边长，不再乘 2.2。接缝核比这个更紧，实际只往外补大约 10 px。
    reach = float(np.clip(cube_size * float(intr.fx) / z_med, 12.0 * pixel_scale, 420.0 * pixel_scale))
    inv = np.where(best_seed, np.uint8(0), np.uint8(255))
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    window = dist <= reach
    radius = int(np.ceil(GROUND_CLEARANCE_M * float(intr.fx) / z_med))
    radius = int(np.clip(radius, max(2, int(round(4 * pixel_scale))), max(4, int(round(20 * pixel_scale)))))
    near = cv2.dilate(
        best_seed.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    contact = (
        (strict > 0)
        & np.isfinite(height_img)
        & (height_img > -0.02)
        & (height_img <= GROUND_CLEARANCE_M)
        & (near > 0)
        & window
    )
    seam_band = cv2.dilate(best_seed.astype(np.uint8) * 255, seam_kernel)
    valid = _valid_depth(depth, zmin, zmax)
    bridge_px = (loose > 0) & (seam_band > 0) & window & valid & ~best_seed
    bridge = np.zeros(strict.shape, dtype=bool)
    by, bx = np.nonzero(bridge_px)
    if len(bx) > 0:
        bz = depth[by, bx]
        bpts = deproject_pixels(np.column_stack([bx, by]), bz, intr)
        bheight = plane.signed_distance(bpts)
        keep = (bheight > GROUND_CLEARANCE_M) & (bheight < cube_size + 0.08)
        bridge[by[keep], bx[keep]] = True
    mask = np.zeros(strict.shape, dtype=np.uint8)
    mask[best_seed | contact | bridge] = 255
    # 接缝上没进放宽颜色的缝用闭运算补上。补上的像素仍限制在一个边长以内。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, seam_kernel)
    mask = cv2.bitwise_and(mask, window.astype(np.uint8) * 255)
    # 凸包的顶点只来自这块方块。斜视时顶面内部被补上，旁边的木板不是顶点，进不了掩膜。
    yy, xx = np.nonzero(mask)
    if len(xx) >= 3:
        hull = cv2.convexHull(np.column_stack([xx, yy]).astype(np.int32))
        filled = np.zeros(strict.shape, dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull, 255)
        mask = filled
    return mask


def _elevated_maps(strict, depth, intr, plane, cube_size, zmin, zmax):
    # 离地 3.5 cm 到边长加 8 cm 的严格颜色。地面和太高的杂物不进种子。
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
    height_img = np.full(strict.shape, np.nan, dtype=float)
    height_img[ys[good], xs[good]] = height
    return elev, height_img


def _top_seeds(
    blob: np.ndarray,
    elev: np.ndarray,
    depth: np.ndarray,
    height_img: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    limit: int = 3,
) -> list[np.ndarray]:
    # 先留长边离边长不超过 max(4 cm, 0.02z) 的。一块都没有时，再留下短到 0.65 倍边长的。
    # 偏长的仍然丢掉。两组里都按像素从多到少排。
    count, labels = cv2.connectedComponents(blob, connectivity=8)
    tight: list[tuple[int, np.ndarray]] = []
    short: list[tuple[int, np.ndarray]] = []
    for lab in range(1, count):
        seed = (labels == lab) & (elev > 0)
        n_seed = int(seed.sum())
        if n_seed < 3:
            continue
        z_med = float(np.median(depth[seed]))
        if not np.isfinite(z_med) or z_med <= 1e-3:
            continue
        edge_px = cube_size * float(intr.fx) / z_med
        min_solid = max(12, int(0.02 * edge_px * edge_px))
        if n_seed < min_solid:
            continue
        sy, sx = np.nonzero(seed)
        box = (int(sx.max()) - int(sx.min()) + 1) * (int(sy.max()) - int(sy.min()) + 1)
        if n_seed < 0.12 * box:
            continue
        long_side = _span_long_side(seed.astype(np.uint8) * 255, depth, intr, plane)
        if long_side is None or long_side > cube_size * 2.5:
            continue
        band = max(0.04, 0.02 * z_med)
        if long_side > cube_size + band or long_side < 0.65 * cube_size:
            continue
        heights = height_img[seed]
        heights = heights[np.isfinite(heights)]
        if len(heights) < 3:
            continue
        peak = float(np.percentile(heights, 90))
        if peak < 0.5 * cube_size or peak > cube_size + 0.08:
            continue
        if long_side >= cube_size - band:
            tight.append((n_seed, seed))
        else:
            short.append((n_seed, seed))
    passed = tight or short
    passed.sort(key=lambda item: item[0], reverse=True)
    return [seed for _n, seed in passed[:limit]]


def _span_long_side(component: np.ndarray, depth: np.ndarray, intr, plane: Plane) -> float | None:
    # 抽样量这块颜色在地面上的长边，用来丢掉比方块大一圈的背景。
    ys, xs = np.nonzero(component)
    if len(xs) < 12:
        return None
    step = max(len(xs) // 600, 1)
    z = depth[ys[::step], xs[::step]]
    good = np.isfinite(z)
    if int(good.sum()) < 8:
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


def _crop_view(color_bgr: np.ndarray, depth: np.ndarray, intr, roi: np.ndarray):
    # 只把方块附近裁出来做全分辨率。主点跟着裁切原点移，焦距不变。
    ys, xs = np.nonzero(roi)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = replace(
        intr,
        width=x1 - x0,
        height=y1 - y0,
        cx=float(intr.cx) - x0,
        cy=float(intr.cy) - y0,
    )
    return color_bgr[y0:y1, x0:x1], depth[y0:y1, x0:x1], cropped, x0, y0


def _paste(shape: tuple[int, int], tile: np.ndarray, x0: int, y0: int) -> np.ndarray:
    full = np.zeros(shape, dtype=np.uint8)
    full[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    return full


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
    seam_full_px: int = 11,
    roi: np.ndarray | None = None,
):
    # 半分辨率只定位。全分辨率在方块附近用小接缝核重描，足迹用这一次的掩膜。
    full_hw = color_bgr.shape[:2]
    origin = (0, 0)
    cropped = False
    view_color, view_depth, view_intr = color_bgr, depth, intr
    if roi is not None and pixel_scale >= 0.99:
        view_color, view_depth, view_intr, x0, y0 = _crop_view(color_bgr, depth, intr, roi)
        origin = (x0, y0)
        cropped = True
    if pixel_scale < 0.99:
        color, depth_s, intr_s, pixel_scale = _half_view(view_color, view_depth, view_intr)
    else:
        color, depth_s, intr_s = view_color, view_depth, view_intr
    strict, loose, front = _color_layers(color, colors, pixel_scale=pixel_scale)
    mask = _cube_silhouette(
        strict,
        loose,
        depth_s,
        intr_s,
        plane,
        cube_size,
        zmin,
        zmax,
        pixel_scale=pixel_scale,
        seam_full_px=seam_full_px,
    )
    if mask is None:
        return None
    if mask.shape[:2] != view_color.shape[:2]:
        full = (view_color.shape[1], view_color.shape[0])
        mask = cv2.resize(mask, full, interpolation=cv2.INTER_NEAREST)
        front = cv2.resize(front, full, interpolation=cv2.INTER_NEAREST)
    front = cv2.bitwise_and(front, mask)
    if cropped:
        mask = _paste(full_hw, mask, origin[0], origin[1])
        front = _paste(full_hw, front, origin[0], origin[1])
    if int(np.count_nonzero(mask)) < 10:
        return None
    return mask, front


def _locate_rois(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    plane: Plane,
    intr,
    cube_size: float,
    zmin: float,
    zmax: float,
    pad: int = 24,
    limit: int = 3,
) -> list[tuple[np.ndarray, int]]:
    # 半分辨率列出最多三块候选，外接框映回全图再外扩。全分辨率对每一块重描。
    color, depth_s, intr_s, scale = _half_view(color_bgr, depth, intr)
    strict, _loose, _front = _color_layers(color, colors, pixel_scale=scale)
    packed = _elevated_maps(strict, depth_s, intr_s, plane, cube_size, zmin, zmax)
    if packed is None:
        return []
    elev, height_img = packed
    seam_px = _scaled_odd(11, scale)
    blob = cv2.dilate(elev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_px, seam_px)))
    seeds = _top_seeds(blob, elev, depth_s, height_img, intr_s, plane, cube_size, limit=limit)
    height, width = color_bgr.shape[:2]
    rois: list[tuple[np.ndarray, int]] = []
    for seed in seeds:
        ys, xs = np.nonzero(seed)
        if len(xs) == 0:
            continue
        x0 = max(int(np.floor(float(xs.min()) / scale)) - pad, 0)
        x1 = min(int(np.ceil(float(xs.max() + 1) / scale)) + pad, width)
        y0 = max(int(np.floor(float(ys.min()) / scale)) - pad, 0)
        y1 = min(int(np.ceil(float(ys.max() + 1) / scale)) + pad, height)
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[y0:y1, x0:x1] = 255
        expected = int(round(float(seed.sum()) / (scale * scale)))
        rois.append((roi, expected))
    return rois


def _refine_roi(
    color_bgr,
    depth,
    colors,
    plane,
    intr,
    cube_size,
    zmin,
    zmax,
    roi,
    expected_n: int = 0,
):
    # 先用 3 px 接缝。掩膜不到定位像素的一半时，红蓝没接上，再用 11 px。
    refined = _mask_from_plane(
        color_bgr, depth, colors, plane, intr, cube_size, zmin, zmax, pixel_scale=1.0, seam_full_px=3, roi=roi
    )
    small = refined is None or (
        expected_n > 0 and int(np.count_nonzero(refined[0])) * 2 < expected_n
    )
    if small:
        wider = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=11,
            roi=roi,
        )
        if wider is not None:
            refined = wider
    return refined


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
    # 有地面时，半分辨率给出最多三块候选。全分辨率逐块重描，留下像素最多的一块。
    # 一块都没定位到时，整幅用全分辨率再找一次。
    if plane is not None and intr is not None:
        found: list[tuple[np.ndarray | None, int]] = []
        if min(color_bgr.shape[:2]) >= 8:
            found = _locate_rois(
                color_bgr, depth, colors, plane, intr, cube_size, zmin, zmax
            )
        if not found:
            found = [(None, 0)]
        best = None
        best_n = -1
        for roi, expected_n in found:
            refined = _refine_roi(
                color_bgr, depth, colors, plane, intr, cube_size, zmin, zmax, roi, expected_n
            )
            if refined is None:
                continue
            n = int(np.count_nonzero(refined[0]))
            if n > best_n:
                best = refined
                best_n = n
        return best
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


def _drop_near_camera(points: np.ndarray, plane: Plane, fraction: float = 0.05) -> np.ndarray:
    # 投影到地面后，丢掉离相机落点最近的大约 5%。偏近的深度飞点就不会把长边撑出绿线。
    projected = plane.project(points)
    cam = plane.project(np.zeros(3))
    dist = np.linalg.norm(projected - cam, axis=1)
    drop = int(np.floor(len(points) * fraction))
    if drop < 1 or len(points) - drop < 3:
        return np.ones(len(points), dtype=bool)
    cutoff = np.partition(dist, drop - 1)[drop - 1]
    keep = dist > cutoff
    if int(keep.sum()) < 3:
        keep = dist >= cutoff
    return keep


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
    # 足迹只投掩膜里接近顶面的那一层。侧面和落地边留在掩膜里，不参与矩形。
    # 掩膜之外的木台即使高度差不多，也不会被投到这个矩形上。
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
    # 落地边和凸包里贴近地面的点不参与 90 分位。峰值只在离地超过半个边长的点上取。
    height = plane.signed_distance(points)
    body = height > max(GROUND_CLEARANCE_M, 0.5 * cube_size)
    if int(body.sum()) < 3:
        return fail("height")
    peak = float(np.percentile(height[body], 90))
    if abs(peak - cube_size) > size_tolerance + 0.05 and peak < cube_size * 0.5:
        return fail("height")
    # 顶面往下 1 cm。这一层面积够时，正面上沿留在矩形外面。
    # 斜视时 1 cm 只剩近处一条，面积不够就改用顶面往下 2.5 cm 的点。
    def measured(layer: np.ndarray):
        if int(layer.sum()) < 3:
            return None
        pts = points[layer]
        flags = front_flag[layer]
        keep = _drop_near_camera(pts, plane, 0.05)
        pts = pts[keep]
        flags = flags[keep]
        if len(pts) < 3:
            return None
        _coords, _projected, rect = _footprint(pts, plane)
        (_center_u, _center_v), (width, height_rect), angle_deg = rect
        long_side = max(width, height_rect)
        short_side = min(width, height_rect)
        fraction = float(width * height_rect) / (cube_size * cube_size)
        return pts, flags, rect, fraction, long_side, short_side, angle_deg

    top = body & (height >= peak - TOP_LAYER_BELOW_PEAK_M)
    chosen = measured(top)
    if chosen is None or chosen[3] < min_footprint_fraction:
        wider = measured(body & (height >= peak - FOOTPRINT_FALLBACK_BELOW_PEAK_M))
        if wider is not None:
            chosen = wider
    if chosen is None:
        return fail("height")
    points_top, front_flag, rect, fraction, long_side, short_side, angle_deg = chosen
    (center_u, center_v), _size, _angle = rect
    # 2. 足迹要接近 0.2 m × 0.2 m。太扁说明只剩一条棱，这一帧拒绝。
    if fraction < min_footprint_fraction:
        return fail("footprint_area")
    # 近处长边仍按 24 mm。更远处按 0.02*z 放宽，2 m 约 4 cm，3 m 约 6 cm，并且不超过 size_tolerance。
    z_ref = float(np.median(points_top[:, 2]))
    if not np.isfinite(z_ref) or z_ref <= 0.0:
        z_ref = 1.0
    side_limit = min(float(size_tolerance), max(0.12 * cube_size, 0.02 * z_ref))
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
