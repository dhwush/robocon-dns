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
    source: str = "depth"
    long_side: float = 0.0
    short_side: float = 0.0
    peak: float = 0.0
    mask: np.ndarray | None = None


# --- 颜色分割 -----------------------------------------------------------------
# 红色在色相环两端，配置里写成两段再并起来。方块上下两半是红和蓝，哪一面在上都可以。
# 绿线只描成片的严格红蓝。放宽颜色只补大约 10 px 的接缝，搜索不超过一个边长。
# 没有深度的点不拿来撑外沿，凸包也不把方块和旁边木板之间的地面填进掩膜。
# 地面漆成红或蓝时，贴地 3.5 cm 以内的颜色点按高度丢掉，不靠把色相收窄来区分。

# 贴地同色时，离地不足这条的点视为地面。侧面最底下贴着方块的一条留下。
GROUND_CLEARANCE_M = 0.035

# 足迹只收实测顶面往下这一条以内的点。再宽就会带上正面上沿，矩形被拉向相机。
TOP_LAYER_BELOW_PEAK_M = 0.010

# 方块四周这一圈里，中位高度和散布都在这以内，才认为地面就是方块脚下的那一层。
_GROUND_RING_M = 0.06
_GROUND_BIAS_M = 0.008
_GROUND_SPREAD_M = 0.015

# 射线交点与测到的深度差过这条，像素就在侧面而不是顶面。位置仍用交点，不用测到的深度。
_RAY_DEPTH_AGREE_M = 0.02
# 中位深度差超过这条，才认为顶面深度确实偏了，值得改用射线。
_RAY_BIAS_TRUST_M = 0.016
# 顶面掩膜只剥掉最外 1 个像素，避开棱上的混合深度。再往里改在平面坐标里裁，避免斜视时把矩形裁歪。

# 新旧足迹中心差过这条，就不折中，按地面是否贴脚下来取舍。
_SCHEME_SPLIT_M = 0.02


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


def _spec_is_red(spec: dict) -> bool:
    # 红在色相环两端。蓝落在 100–130。名字优先，没有名字时用色相段判断。
    name = str(spec.get("name", ""))
    if name.startswith("red"):
        return True
    if name.startswith("blue"):
        return False
    low = int(spec["hsv_low"][0])
    high = int(spec["hsv_high"][0])
    return low <= 15 or high >= 165


def _hsv_to_bgr(hue: int, sat: int, val: int) -> tuple[int, int, int]:
    patch = np.array([[[hue, sat, val]]], dtype=np.uint8)
    bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def standard_color_chips(colors: list[dict]) -> list[dict]:
    # 图像识别里的标准红、标准蓝是满饱和、满明度的原色。
    # 色相取配置门槛里包住的那一个：红是 0，蓝是 120。门槛里没有时用该段中点。
    grouped: dict[str, list[dict]] = {"red": [], "blue": []}
    for spec in colors:
        grouped["red" if _spec_is_red(spec) else "blue"].append(spec)
    chips = []
    for name, preferred in (("red", 0), ("blue", 120)):
        specs = grouped[name]
        if not specs:
            continue
        covered = any(int(spec["hsv_low"][0]) <= preferred <= int(spec["hsv_high"][0]) for spec in specs)
        if covered:
            hue = preferred
        else:
            spec = specs[0]
            hue = int(round(0.5 * (int(spec["hsv_low"][0]) + int(spec["hsv_high"][0]))))
        sat_low = min(int(spec["hsv_low"][1]) for spec in specs)
        val_low = min(int(spec["hsv_low"][2]) for spec in specs)
        chips.append(
            {
                "name": name,
                "hsv": (hue, 255, 255),
                "bgr": _hsv_to_bgr(hue, 255, 255),
                "gate_bgr": _hsv_to_bgr(hue, sat_low, val_low),
                "sat_low": sat_low,
                "val_low": val_low,
                "ranges": [(int(spec["hsv_low"][0]), int(spec["hsv_high"][0])) for spec in specs],
            }
        )
    return chips


def measure_scene_hsv(color_bgr: np.ndarray, colors: list[dict]) -> dict[str, tuple[int, int, int] | None]:
    # 只按色相圈出红和蓝，饱和度和明度留着跟门槛比。进不了严格掩膜时仍能看出偏暗还是发白。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    found: dict[str, tuple[int, int, int] | None] = {}
    for chip in standard_color_chips(colors):
        band = np.zeros(hue.shape, dtype=bool)
        for low, high in chip["ranges"]:
            if low <= high:
                band |= (hue >= low) & (hue <= high)
            else:
                band |= (hue >= low) | (hue <= high)
        count = int(band.sum())
        if count < 30:
            found[chip["name"]] = None
            continue
        pix = hsv[band]
        values = pix[:, 0].astype(np.int16)
        if chip["name"] == "red":
            values = np.where(values > 90, values - 180, values)
        median_h = int(np.median(values)) % 180
        found[chip["name"]] = (median_h, int(np.median(pix[:, 1])), int(np.median(pix[:, 2])))
    return found


def _light_step(light_name: str, levels: dict | None, measured: tuple[int, int, int] | None, chip: dict) -> str:
    # 曝光从 bright 到 dark 递增。发白就往曝光更低的一档走，偏暗就往更高的一档走。
    names = list(levels) if levels else []
    index = names.index(light_name) if light_name in names else -1
    if measured is None:
        return "no pixels"
    _hue, sat, val = measured
    if val < chip["val_low"]:
        if 0 <= index < len(names) - 1:
            return f"too dark -> --light {names[index + 1]}"
        return "too dark"
    if sat < chip["sat_low"]:
        if index > 0:
            return f"washed out -> --light {names[index - 1]}"
        return "washed out"
    return "in range"


def render_color_standard(
    colors: list[dict],
    light_name: str = "daylight",
    light_levels: dict | None = None,
    measured: dict | None = None,
) -> np.ndarray:
    # 左块是标准色，右块是当前画面里同色相的中值。底下一条是还能进掩膜的最暗颜色。
    chips = standard_color_chips(colors)
    width, height = 880, 300
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    if not chips:
        return image
    slot = width // len(chips)

    def _text(origin: tuple[int, int], line: str) -> None:
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for index, chip in enumerate(chips):
        x0 = index * slot + 24
        hue, sat, val = chip["hsv"]
        cv2.rectangle(image, (x0, 48), (x0 + 150, 148), chip["bgr"], -1)
        cv2.rectangle(image, (x0, 156), (x0 + 150, 184), chip["gate_bgr"], -1)
        _text((x0, 32), f"{chip['name']}  H{hue} S{sat} V{val}")
        _text((x0, 206), f"gate S>={chip['sat_low']} V>={chip['val_low']}")
        scene = None if measured is None else measured.get(chip["name"])
        if scene is None:
            _text((x0, 240), "scene: none")
        else:
            swatch = _hsv_to_bgr(*scene)
            cv2.rectangle(image, (x0 + 168, 48), (x0 + 250, 148), swatch, -1)
            _text((x0, 240), f"scene H{scene[0]} S{scene[1]} V{scene[2]}")
        _text((x0, 272), _light_step(light_name, light_levels, scene, chip))
    return image


def show_color_standard(image: np.ndarray, pump: bool = True) -> bool:
    # 没有显示器时不打断取流。调用方继续跑，只是看不到这块色板。
    # pump 为假时不取键盘，留给已经在等按键的那个窗口。
    try:
        cv2.imshow("color standard", image)
        if pump:
            cv2.waitKey(1)
    except cv2.error:
        return False
    return True


def _valid_depth(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # 工作距离内的有限深度才参与选块。NaN 和超距点不能拿来判断这块是不是方块。
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def _cube_silhouettes(
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
) -> list[np.ndarray]:
    # 绿线沿严格红蓝走。种子先要成片，再比地面长边谁更接近边长。
    # 长边超过 2.5 倍边长的块丢掉之后，稀疏的线材补不上这个门槛，不能接替方块。
    # 放宽颜色只填红蓝接缝那大约 10 px。再往外一个边长是上限，木板和机器人进不来。
    valid = _valid_depth(depth, zmin, zmax)
    ys, xs = np.nonzero(strict)
    if len(xs) < 3:
        return []
    z = depth[ys, xs]
    good = valid[ys, xs]
    if int(good.sum()) < 3:
        return []
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
    height = plane.signed_distance(points)
    elevated = (height > GROUND_CLEARANCE_M) & (height < cube_size + 0.08)
    if int(elevated.sum()) < 3:
        return []
    elev = np.zeros(strict.shape, dtype=np.uint8)
    elev[ys[good][elevated], xs[good][elevated]] = 255
    height_img = np.full(strict.shape, np.nan, dtype=float)
    height_img[ys[good], xs[good]] = height
    # 定位用大约 11 px 把红蓝接上。全分辨率重描时用 3 px，避免把棱向外胀开。
    seam_px = _scaled_odd(seam_full_px, pixel_scale)
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_px, seam_px))
    blob = cv2.dilate(elev, seam_kernel)
    seeds = _ranked_seeds(blob, elev, depth, intr, plane, cube_size, limit=3)
    masks: list[np.ndarray] = []
    for seed in seeds:
        mask = _mask_from_seed(
            seed,
            strict,
            loose,
            depth,
            intr,
            plane,
            cube_size,
            height_img,
            pixel_scale,
            seam_kernel,
            zmin,
            zmax,
        )
        if mask is not None and int(np.count_nonzero(mask)) >= 10:
            masks.append(mask)
    return masks


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
    masks = _cube_silhouettes(
        strict, loose, depth, intr, plane, cube_size, zmin, zmax, pixel_scale, seam_full_px
    )
    if not masks:
        return None
    return masks[0]


def _ranked_seeds(
    blob: np.ndarray,
    elev: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    limit: int = 3,
) -> list[np.ndarray]:
    # 通过成片、外接框和 2.5 倍边长的块都留下，最多 3 块。长边更接近边长的排在前面。
    count, labels = cv2.connectedComponents(blob, connectivity=8)
    ranked: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for lab in range(1, count):
        seed = (labels == lab) & (elev > 0)
        n_seed = int(seed.sum())
        if n_seed < 3:
            continue
        z_med = float(np.median(depth[seed]))
        if not np.isfinite(z_med) or z_med <= 1e-3:
            continue
        edge_px = cube_size * float(intr.fx) / z_med
        # 门槛按这一距离上一个面的 2% 走。远处半分辨率只剩三十来个像素时，不再用近处的 8%。
        min_solid = max(12, int(0.02 * edge_px * edge_px))
        if n_seed < min_solid:
            continue
        sy, sx = np.nonzero(seed)
        box = (int(sx.max()) - int(sx.min()) + 1) * (int(sy.max()) - int(sy.min()) + 1)
        # 线材铺开时长边也能接近边长，像素占不满外接框，先丢掉。
        if n_seed < 0.12 * box:
            continue
        long_side = _span_long_side(seed.astype(np.uint8) * 255, depth, intr, plane)
        if long_side is not None and long_side > cube_size * 2.5:
            continue
        # 量不出长边的成片块排在后面。远处深度稀时仍能留下，不让它盖过已经量准的方块。
        if long_side is None:
            rank = (1.0, 0.0, -float(n_seed))
        else:
            rank = (0.0, abs(long_side - cube_size), -float(n_seed))
        ranked.append((rank, seed))
    ranked.sort(key=lambda item: item[0])
    return [seed for _rank, seed in ranked[: max(int(limit), 1)]]


def _mask_from_seed(
    seed: np.ndarray,
    strict: np.ndarray,
    loose: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    height_img: np.ndarray,
    pixel_scale: float,
    seam_kernel: np.ndarray,
    zmin: float,
    zmax: float,
) -> np.ndarray | None:
    valid = _valid_depth(depth, zmin, zmax)
    z_med = float(np.median(depth[seed]))
    if not np.isfinite(z_med) or z_med <= 1e-3:
        return None
    # 搜索半径是一个边长，不再乘 2.2。接缝核比这个更紧，实际只往外补大约 10 px。
    reach = float(np.clip(cube_size * float(intr.fx) / z_med, 12.0 * pixel_scale, 420.0 * pixel_scale))
    inv = np.where(seed, np.uint8(0), np.uint8(255))
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    window = dist <= reach
    radius = int(np.ceil(GROUND_CLEARANCE_M * float(intr.fx) / z_med))
    radius = int(np.clip(radius, max(2, int(round(4 * pixel_scale))), max(4, int(round(20 * pixel_scale)))))
    near = cv2.dilate(
        seed.astype(np.uint8) * 255,
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
    seam_band = cv2.dilate(seed.astype(np.uint8) * 255, seam_kernel)
    bridge_px = (loose > 0) & (seam_band > 0) & window & valid & ~seed
    bridge = np.zeros(strict.shape, dtype=bool)
    by, bx = np.nonzero(bridge_px)
    if len(bx) > 0:
        bz = depth[by, bx]
        bpts = deproject_pixels(np.column_stack([bx, by]), bz, intr)
        bheight = plane.signed_distance(bpts)
        keep = (bheight > GROUND_CLEARANCE_M) & (bheight < cube_size + 0.08)
        bridge[by[keep], bx[keep]] = True
    mask = np.zeros(strict.shape, dtype=np.uint8)
    mask[seed | contact | bridge] = 255
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


def _expand_roi(mask: np.ndarray, pad: int) -> np.ndarray | None:
    # 半分辨率只给出方块大概在哪。外接框向外留一圈，全分辨率重描时棱不会被裁掉。
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    roi = np.zeros(mask.shape, dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255
    return roi


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


def _masks_from_plane(
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
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 整幅上最多留 3 块绿线。每一块单独成掩膜，后面按地面中心比较。
    if pixel_scale < 0.99:
        color, depth_s, intr_s, pixel_scale = _half_view(color_bgr, depth, intr)
    else:
        color, depth_s, intr_s = color_bgr, depth, intr
    strict, loose, front = _color_layers(color, colors, pixel_scale=pixel_scale)
    masks = _cube_silhouettes(
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
    found: list[tuple[np.ndarray, np.ndarray]] = []
    full = (color_bgr.shape[1], color_bgr.shape[0])
    for mask in masks:
        if mask.shape[:2] != color_bgr.shape[:2]:
            mask = cv2.resize(mask, full, interpolation=cv2.INTER_NEAREST)
            front_full = cv2.resize(front, full, interpolation=cv2.INTER_NEAREST)
        else:
            front_full = front
        front_full = cv2.bitwise_and(front_full, mask)
        if int(np.count_nonzero(mask)) < 10:
            continue
        found.append((mask, front_full))
    return found


def prepare_candidate_masks(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    zmin: float,
    zmax: float,
    plane: Plane,
    intr,
    cube_size: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 半分辨率找出最多 3 块，再各自在全分辨率上重描。一块重描失败就留定位掩膜。
    located: list[tuple[np.ndarray, np.ndarray]] = []
    if min(color_bgr.shape[:2]) >= 8:
        located = _masks_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=0.5,
            seam_full_px=11,
        )
    if not located:
        one = _mask_from_plane(
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
        )
        return [] if one is None else [one]
    refined: list[tuple[np.ndarray, np.ndarray]] = []
    for mask, _front in located[:3]:
        roi = _expand_roi(mask, 24)
        one = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=3,
            roi=roi,
        )
        if one is not None and int(np.count_nonzero(one[0])) * 2 < int(np.count_nonzero(mask)):
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
                one = wider
        refined.append(one if one is not None else (mask, _front))
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
    # 有地面时，半分辨率只找方块在哪。找到后在全分辨率上用 3 px 接缝重描，足迹用这一次。
    # 半分辨率没找到时，整幅用全分辨率再找一次。重描失败才退回定位掩膜。
    if plane is not None and intr is not None:
        located = None
        if min(color_bgr.shape[:2]) >= 8:
            located = _mask_from_plane(
                color_bgr,
                depth,
                colors,
                plane,
                intr,
                cube_size,
                zmin,
                zmax,
                pixel_scale=0.5,
                seam_full_px=11,
            )
        roi = _expand_roi(located[0], 24) if located is not None else None
        refined = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=3,
            roi=roi,
        )
        # 3 px 接不上红蓝时，掩膜会只剩一条。同一块区域再用定位核重描，足迹才盖住顶面。
        if located is not None and refined is not None:
            if int(np.count_nonzero(refined[0])) * 2 < int(np.count_nonzero(located[0])):
                refined = _mask_from_plane(
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
        if refined is not None:
            return refined
        return located
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
    return points, front_flag, int(good.sum()), mask, pixels


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


def _ground_under_cube(depth: np.ndarray, mask: np.ndarray, plane: Plane, intr) -> bool:
    # 整幅里内点够多，只说明有一块大平面，木板和地砖都会过。这里只看方块轮廓外面几厘米。
    # 这一圈的高度中位数离开 0，或散布超过 1.5 cm，平面就不是方块脚下的那一层。
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return False
    z_mask = depth[ys, xs]
    z_mask = z_mask[np.isfinite(z_mask) & (z_mask > 0.2)]
    if len(z_mask) < 10:
        return False
    radius = int(np.clip(round(_GROUND_RING_M * float(intr.fx) / float(np.median(z_mask))), 6, 48))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.subtract(cv2.dilate(mask, kernel), mask)
    ry, rx = np.nonzero(ring)
    if len(rx) < 30:
        return False
    step = max(len(rx) // 800, 1)
    rx = rx[::step]
    ry = ry[::step]
    rz = depth[ry, rx]
    good = np.isfinite(rz) & (rz > 0.2)
    if int(good.sum()) < 25:
        return False
    points = deproject_pixels(np.column_stack([rx[good], ry[good]]), rz[good], intr)
    dist = plane.signed_distance(points)
    band = np.abs(dist) < 0.05
    if int(band.sum()) < 25 or int(band.sum()) < 0.45 * len(dist):
        return False
    sample = dist[band]
    if abs(float(np.median(sample))) > _GROUND_BIAS_M:
        return False
    return float(np.percentile(np.abs(sample), 75)) <= _GROUND_SPREAD_M


def _intersect_lifted_plane(pixels: np.ndarray, plane: Plane, intr, lift: float) -> np.ndarray:
    # 像素只定射线。交点落在与地面平行、抬高 lift 的平面上，不乘测到的深度。
    dirs = deproject_pixels(pixels, np.ones(len(pixels)), intr)
    denom = dirs @ np.asarray(plane.normal, dtype=float)
    level = float(plane.offset) + float(lift)
    t = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    t[ok] = level / denom[ok]
    t[~(t > 0.0)] = np.nan
    return t[:, None] * dirs


def _ray_top_points(
    pixels: np.ndarray,
    measured: np.ndarray,
    front: np.ndarray,
    depth_shape: tuple[int, int],
    plane: Plane,
    intr,
    cube_size: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    # 顶面外圈两三像素是棱，深度常混进正面。先剥掉，点不够再退回没剥的那一版。
    # 测到的深度只用来丢掉侧面：侧面的射线要走得更远才碰到抬高的平面。
    if len(pixels) < 3:
        return None
    sel = np.zeros(depth_shape, dtype=np.uint8)
    sel[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(sel, kernel)
    keep = eroded[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] > 0
    if int(keep.sum()) < 20:
        keep = np.ones(len(pixels), dtype=bool)
    hits = _intersect_lifted_plane(pixels[keep], plane, intr, cube_size)
    z_hit = hits[:, 2]
    agree = np.isfinite(z_hit) & (np.abs(z_hit - measured[keep]) < _RAY_DEPTH_AGREE_M)
    if int(agree.sum()) < 3:
        return None
    median_dz = float(np.median(np.abs(z_hit[agree] - measured[keep][agree])))
    return hits[agree], front[keep][agree], median_dz


def _rectangle_on_plane(
    points: np.ndarray,
    plane: Plane,
    cube_size: float,
    min_footprint_fraction: float,
    size_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
    # 投影后做最小外接矩形。长边、短边或面积不合格就不用这套点。
    if len(points) < 3:
        return None
    coords, _projected, rect = _footprint(points, plane)
    (center_u, center_v), (width, height_rect), angle_deg = rect
    long_side = max(width, height_rect)
    short_side = min(width, height_rect)
    fraction = float(width * height_rect) / (cube_size * cube_size)
    if fraction < min_footprint_fraction:
        return None
    z_ref = float(np.median(points[:, 2]))
    if not np.isfinite(z_ref) or z_ref <= 0.0:
        z_ref = 1.0
    side_limit = min(float(size_tolerance), max(0.12 * cube_size, 0.02 * z_ref))
    if abs(long_side - cube_size) > side_limit or short_side < cube_size * 0.45:
        return None
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    bottom = origin + center_u * axis_x + center_v * axis_y
    box = cv2.boxPoints(rect)
    corners = origin + box[:, 0:1] * axis_x + box[:, 1:2] * axis_y
    angle = np.deg2rad(angle_deg)
    edge_2d = np.array([np.cos(angle), np.sin(angle)])
    return bottom, corners, coords, edge_2d, fraction, float(long_side), float(short_side)


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
    points, front_flag, count, mask, pixels = packed
    if count < min_pixels:
        return fail("inliers")
    height = plane.signed_distance(points)
    # 掩膜已经去掉了方块占地以外的地面。这里再丢掉钻到地下和高过方块的点，落地边保留。
    on_cube = (height > -0.02) & (height < cube_size + 0.08)
    if int(on_cube.sum()) < min_pixels:
        return fail("inliers")
    points = points[on_cube]
    front_flag = front_flag[on_cube]
    pixels = pixels[on_cube]
    # 落地边和凸包里贴近地面的点不参与 90 分位。峰值只在离地超过半个边长的点上取。
    height = plane.signed_distance(points)
    body = height > max(GROUND_CLEARANCE_M, 0.5 * cube_size)
    if int(body.sum()) < 3:
        return fail("height")
    peak = float(np.percentile(height[body], 90))
    if abs(peak - cube_size) > size_tolerance + 0.05 and peak < cube_size * 0.5:
        return fail("height")
    # 顶面往下 1 cm。正面上沿留在这层外面，投到地面时不会把框拉出绿线。
    top = body & (height >= peak - TOP_LAYER_BELOW_PEAK_M)
    if int(top.sum()) < 3:
        return fail("height")
    # 2. 两套足迹。旧的把测到的顶面点压到地面。新的用射线和抬高一个边长的平面求交。
    # 中心差过 2 cm 时不取平均：方块脚下贴着平面就用新的，贴不上就用旧的。
    # 中心还接近时留旧的，避免几毫米的平面误差被射线放大。
    points_top = points[top]
    front_top = front_flag[top]
    pixels_top = pixels[top]
    keep = _drop_near_camera(points_top, plane, 0.05)
    old = _rectangle_on_plane(
        points_top[keep],
        plane,
        cube_size,
        min_footprint_fraction,
        size_tolerance,
    )
    ray = _ray_top_points(
        pixels_top,
        points_top[:, 2],
        front_top,
        depth.shape[:2],
        plane,
        intr,
        cube_size,
    )
    new = None if ray is None else _rectangle_on_plane(
        ray[0],
        plane,
        cube_size,
        min_footprint_fraction,
        size_tolerance,
    )
    ray_front = None if ray is None or new is None else ray[1]
    # 测到的深度和射线交点差不多时，位置差来自棱上的点，不是顶面深度偏了。这种差不改用射线。
    ground_ok = _ground_under_cube(depth, mask, plane, intr)
    # 中心差过 2 cm 才算两套足迹真正不一致。这时不取中点：脚下贴着平面、并且深度确实偏了，才信射线。
    # 中心还接近，或深度并没有偏，留深度投影。射线会把平面上几毫米的高度误差放大到足迹上。
    center_gap = 0.0 if old is None or new is None else float(np.linalg.norm(old[0] - new[0]))
    center_split = old is not None and new is not None and center_gap > _SCHEME_SPLIT_M
    depth_biased = ray is not None and ray[2] > _RAY_BIAS_TRUST_M
    trust_ray = new is not None and ground_ok and depth_biased and (old is None or center_split)
    if trust_ray:
        bottom, corners, coords, edge_2d, fraction, long_side, short_side = new
        front_flag = ray_front
        source = "ray"
    elif old is not None:
        bottom, corners, coords, edge_2d, fraction, long_side, short_side = old
        front_flag = front_top[keep]
        source = "depth"
    else:
        return fail("footprint_long")
    # 3. 底面中心给位置。能分出正面颜色时，用颜色重心相对足迹中心的方向当 yaw；
    #    否则用矩形棱的方向，并折进 90° 周期。
    axis_x, axis_y = plane.basis()
    front_ratio = float(front_flag.mean()) if len(front_flag) else 0.0
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
        source=source,
        long_side=float(long_side),
        short_side=float(short_side),
        peak=float(peak),
    )


def detect_cube_candidates(
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
) -> list[CubePose]:
    # 最多 3 块各自做足迹。绿线可以在这几块之间跳，选择留到多帧比较。
    # 一块都没通过时只记第一个拒绝原因，不把 height 和后面的 mask 叠在同一帧上。
    prepared_list = prepare_candidate_masks(
        color_bgr, depth, colors, zmin, zmax, plane, intr, cube_size
    )
    if not prepared_list:
        if rejects is not None:
            rejects["mask"] = int(rejects.get("mask", 0)) + 1
        return []
    poses: list[CubePose] = []
    first_reason: str | None = None
    for prepared in prepared_list[:3]:
        local: dict[str, int] = {}
        pose = detect_visible_cube(
            color_bgr,
            depth,
            intr,
            plane,
            colors,
            cube_size=cube_size,
            min_pixels=min_pixels,
            min_footprint_fraction=min_footprint_fraction,
            size_tolerance=size_tolerance,
            rotation=rotation,
            translation=translation,
            zmin=zmin,
            zmax=zmax,
            rejects=local,
            prepared=prepared,
        )
        if pose is None:
            if first_reason is None and local:
                first_reason = next(iter(local))
            continue
        pose.mask = prepared[0]
        poses.append(pose)
    if not poses and rejects is not None:
        reason = first_reason or "mask"
        rejects[reason] = int(rejects.get(reason, 0)) + 1
    return poses
