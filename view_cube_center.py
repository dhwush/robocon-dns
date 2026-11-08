"""实时查看方块识别和体心。

体心是地面足迹中心沿地面法向抬起半个边长，与正式输出的几何中心同一点。
橙色线是拟合地面上最大的一块，内点放宽到 18 mm。画面写出地面上的前方距离、左右距离和地面 yaw。
另一行是当前相机右前点坐标系里的体心。轴跟着相机：向右 X、向前 Y、向上 Z。按 q 或 Esc 退出。

    python view_cube_center.py
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np

from camera.source import RealSenseSource, color_lock_from_config, filter_depth
from cube.detect import (
    detect_visible_cube,
    measure_scene_hsv,
    prepare_mask,
    render_color_standard,
    show_color_standard,
)
from field.extrinsic import camera_corner_extrinsic
from geom.transform import camera_to_world, deproject_pixels, fold_yaw_quarter, project_points
from ground.plane import fit_ground, plane_holds, sample_points
from main import live_intrinsics, load_config


def _on_ground(direction: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    # 去掉沿法向的分量。相机正对地面时投影接近 0，这一帧就没有前后方向。
    flat = np.asarray(direction, dtype=float) - float(np.dot(direction, normal)) * normal
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        return None
    return flat / norm


def _ground_readout(bottom: np.ndarray, plane, edge: np.ndarray, cube_size: float):
    # 前方、右方都在地面上，并且互相垂直。yaw 从右方量起，正方形折进 ±45°。
    normal = plane.normal
    forward = _on_ground(np.array([0.0, 0.0, 1.0]), normal)
    if forward is None:
        return None
    right = np.cross(forward, normal)
    right = right / np.linalg.norm(right)
    horizontal = np.asarray(bottom, dtype=float) - plane.origin()
    ahead = float(horizontal @ forward)
    side = float(horizontal @ right)
    above_top = float(plane.signed_distance(np.zeros(3))[0]) - cube_size
    flat_edge = _on_ground(edge, normal)
    if flat_edge is None:
        yaw = 0.0
    else:
        yaw = fold_yaw_quarter(float(np.arctan2(flat_edge @ forward, flat_edge @ right)))
    return ahead, side, above_top, yaw


def _pixel(point: np.ndarray, intr) -> tuple[int, int] | None:
    # 相机后方的点没有像素。体心和足迹角都先过这一步再画。
    if float(point[2]) <= 1e-4:
        return None
    uv = project_points(np.asarray(point, dtype=float), intr)[0]
    if not np.isfinite(uv).all():
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def _cross(image: np.ndarray, pixel: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    u, v = pixel
    cv2.drawMarker(image, (u, v), color, cv2.MARKER_CROSS, radius, 2, cv2.LINE_AA)
    cv2.circle(image, (u, v), 3, color, -1, cv2.LINE_AA)


def _label(image: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


_CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
)
_cjk_face = None
_cjk_size = 0


class _FTGeneric(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("finalizer", ctypes.c_void_p)]


class _FTVector(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _FTMetrics(ctypes.Structure):
    _fields_ = [(name, ctypes.c_long) for name in (
        "width", "height", "horiBearingX", "horiBearingY", "horiAdvance",
        "vertBearingX", "vertBearingY", "vertAdvance",
    )]


class _FTBitmap(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_uint),
        ("width", ctypes.c_uint),
        ("pitch", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("num_grays", ctypes.c_ushort),
        ("pixel_mode", ctypes.c_ubyte),
        ("palette_mode", ctypes.c_ubyte),
        ("palette", ctypes.c_void_p),
    ]


class _FTGlyph(ctypes.Structure):
    # 只保留到位图顶部。后面的轮廓字段这一帧用不到。
    _fields_ = [
        ("library", ctypes.c_void_p),
        ("face", ctypes.c_void_p),
        ("next", ctypes.c_void_p),
        ("glyph_index", ctypes.c_uint),
        ("generic", _FTGeneric),
        ("metrics", _FTMetrics),
        ("linearHoriAdvance", ctypes.c_long),
        ("linearVertAdvance", ctypes.c_long),
        ("advance", _FTVector),
        ("format", ctypes.c_int),
        ("bitmap", _FTBitmap),
        ("bitmap_left", ctypes.c_int),
        ("bitmap_top", ctypes.c_int),
    ]


class _FTBBox(ctypes.Structure):
    _fields_ = [("xMin", ctypes.c_long), ("yMin", ctypes.c_long), ("xMax", ctypes.c_long), ("yMax", ctypes.c_long)]


class _FTFace(ctypes.Structure):
    _fields_ = [
        ("num_faces", ctypes.c_long),
        ("face_index", ctypes.c_long),
        ("face_flags", ctypes.c_long),
        ("style_flags", ctypes.c_long),
        ("num_glyphs", ctypes.c_long),
        ("family_name", ctypes.c_char_p),
        ("style_name", ctypes.c_char_p),
        ("num_fixed_sizes", ctypes.c_int),
        ("available_sizes", ctypes.c_void_p),
        ("num_charmaps", ctypes.c_int),
        ("charmaps", ctypes.c_void_p),
        ("generic", _FTGeneric),
        ("bbox", _FTBBox),
        ("units_per_EM", ctypes.c_ushort),
        ("ascender", ctypes.c_short),
        ("descender", ctypes.c_short),
        ("height", ctypes.c_short),
        ("max_advance_width", ctypes.c_short),
        ("max_advance_height", ctypes.c_short),
        ("underline_position", ctypes.c_short),
        ("underline_thickness", ctypes.c_short),
        ("glyph", ctypes.POINTER(_FTGlyph)),
    ]


def _freetype():
    library = ctypes.CDLL("libfreetype.so.6")
    library.FT_Init_FreeType.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.FT_New_Face.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)]
    library.FT_Set_Pixel_Sizes.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    library.FT_Load_Char.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int]
    library.FT_Done_Face.argtypes = [ctypes.c_void_p]
    return library


def _open_cjk(size: int):
    # Hershey 字体没有汉字。用系统里的思源黑体，经 FreeType 画到画面上。
    global _cjk_face, _cjk_size
    if _cjk_face is not None and _cjk_size == size:
        return _cjk_face
    path = next((item for item in _CJK_FONTS if Path(item).exists()), None)
    if path is None:
        return None
    try:
        library = _freetype()
        handle = ctypes.c_void_p()
        if library.FT_Init_FreeType(ctypes.byref(handle)) != 0:
            return None
        chosen = None
        for index in range(8):
            face_ptr = ctypes.c_void_p()
            if library.FT_New_Face(handle, path.encode(), index, ctypes.byref(face_ptr)) != 0:
                break
            face = ctypes.cast(face_ptr, ctypes.POINTER(_FTFace))
            family = face.contents.family_name or b""
            if b"CJK SC" in family or (chosen is None and b"CJK" in family):
                if chosen is not None:
                    library.FT_Done_Face(chosen[1])
                chosen = (library, face_ptr, face)
                if b"CJK SC" in family:
                    break
            else:
                library.FT_Done_Face(face_ptr)
        if chosen is None:
            return None
        if chosen[0].FT_Set_Pixel_Sizes(chosen[1], 0, int(size)) != 0:
            return None
        _cjk_face = chosen
        _cjk_size = size
        return chosen
    except OSError:
        return None


def _glyph(face_ptr, library, code: int):
    if library.FT_Load_Char(face_ptr, code, 4) != 0:
        return None
    face = ctypes.cast(face_ptr, ctypes.POINTER(_FTFace))
    slot = face.contents.glyph.contents
    bitmap = slot.bitmap
    if bitmap.buffer is None or bitmap.width == 0 or bitmap.rows == 0 or bitmap.pixel_mode != 2:
        advance = max(int(slot.advance.x >> 6), 1)
        return None, advance
    pitch = int(bitmap.pitch)
    rows = int(bitmap.rows)
    width = int(bitmap.width)
    raw = ctypes.string_at(bitmap.buffer, rows * abs(pitch))
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(rows, abs(pitch))[:, :width].copy()
    if pitch < 0:
        pixels = np.flipud(pixels)
    return pixels, int(slot.bitmap_left), int(slot.bitmap_top), max(int(slot.advance.x >> 6), width)


def paint_notice(image: np.ndarray, text: str) -> None:
    """在画面底部写中文提示。没有字体时保持原图。"""
    if not text:
        return
    opened = _open_cjk(32)
    if opened is None:
        return
    library, face_ptr, _face = opened
    pen = 0
    placed = []
    for char in text:
        glyph = _glyph(face_ptr, library, ord(char))
        if glyph is None:
            pen += 32
            continue
        if glyph[0] is None:
            pen += int(glyph[1])
            continue
        pixels, left, top, advance = glyph
        placed.append((pen + left, top, pixels))
        pen += advance
    if not placed:
        return
    tops = [top for _x, top, pixels in placed]
    bottoms = [top - pixels.shape[0] for _x, top, pixels in placed]
    ascent = max(tops)
    descent = min(bottoms)
    height = ascent - descent
    origin_x = 16
    origin_y = max(8, image.shape[0] - height - 16)

    def blit(dx: int, dy: int, color: tuple[int, int, int]) -> None:
        for x, top, pixels in placed:
            y0 = origin_y + ascent - top + dy
            x0 = origin_x + x + dx
            y1 = min(image.shape[0], y0 + pixels.shape[0])
            x1 = min(image.shape[1], x0 + pixels.shape[1])
            ys = max(0, y0)
            xs = max(0, x0)
            if y1 <= ys or x1 <= xs:
                continue
            patch = pixels[ys - y0 : y1 - y0, xs - x0 : x1 - x0]
            alpha = patch.astype(np.float32) / 255.0
            view = image[ys:y1, xs:x1]
            for channel, value in enumerate(color):
                view[:, :, channel] = (alpha * value + (1.0 - alpha) * view[:, :, channel]).astype(np.uint8)

    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        blit(dx, dy, (0, 0, 0))
    blit(0, 0, (255, 255, 255))


def _draw_ground(image: np.ndarray, depth: np.ndarray, intr, plane, thresh: float) -> None:
    # 拟合仍用 8 mm。橙线放宽到至少 18 mm，只画最大的一块，刨花板上的碎斑不画。
    # 方块高出地面，不进这条线。
    stride = 8
    draw_thresh = max(float(thresh), 0.018)
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us]
    good = np.isfinite(z)
    if int(good.sum()) < 30:
        return
    points = deproject_pixels(
        np.column_stack([us[good].ravel(), vs[good].ravel()]),
        z[good].ravel(),
        intr,
    )
    on = np.abs(plane.signed_distance(points)) < draw_thresh
    if int(on.sum()) < 30:
        return
    mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    mask[vs[good][on], us[good][on]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stride * 2 + 1, stride * 2 + 1))
    mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest], -1, (0, 140, 255), 2, cv2.LINE_AA)


def _body_center(bottom: np.ndarray, plane, height: float) -> np.ndarray:
    # 足迹中心在地面上。沿法向抬半高就是体心，不再用顶面去减一次半高。
    return np.asarray(bottom, dtype=float) + plane.normal * (0.5 * height)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看方块识别和体心")
    parser.add_argument("--light", default=None, help="亮度档：bright、daylight、indoor、dim、dark")
    args = parser.parse_args()
    cfg = load_config("config/default.yaml")
    color_lock = color_lock_from_config(cfg, args.light)
    serial, intr = live_intrinsics()
    source = RealSenseSource(
        intr.width,
        intr.height,
        fps=int(cfg.get("fps", 30)),
        serial=serial,
        color_lock=color_lock,
    )
    height = float(cfg["cube_size"])
    last = time.monotonic()
    fps = 0.0
    held_plane = None
    stride = int(cfg.get("ground_stride", 8))
    thresh = float(cfg["ground_ransac_thresh"])
    max_angle = float(cfg["ground_max_angle_deg"])
    iterations = int(cfg.get("ground_ransac_iterations", 50))
    try:
        while True:
            frame = source.read()
            depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
            gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 0.0, 1.0])
            cloud = sample_points(depth, intr, stride=stride, gravity=gravity)
            if held_plane is not None and plane_holds(cloud, held_plane, gravity, thresh, max_angle):
                plane = held_plane
            else:
                plane = fit_ground(
                    cloud,
                    gravity,
                    thresh=thresh,
                    max_angle_deg=max_angle,
                    iterations=iterations,
                )
            held_plane = plane
            vis = frame.color.copy()
            prepared = None
            if plane is not None:
                _draw_ground(vis, depth, intr, plane, float(cfg["ground_ransac_thresh"]))
                prepared = prepare_mask(
                    frame.color,
                    depth,
                    cfg["colors"],
                    float(cfg["depth_min"]),
                    float(cfg["depth_max"]),
                    plane=plane,
                    intr=intr,
                    cube_size=height,
                )
            mask_view = np.zeros(frame.color.shape[:2], dtype=np.uint8)
            if prepared is not None:
                mask_view = prepared[0]
                contours, _ = cv2.findContours(mask_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            lock = color_lock
            lines = [f"{fps:.0f} fps  {lock['name']}  exp {lock['exposure']:.0f}  gain {lock['gain']:.0f}"]
            if plane is None:
                lines.append("no ground")
            elif prepared is None:
                lines.append("no cube (mask)")
            else:
                rejects: dict[str, int] = {}
                pose = detect_visible_cube(
                    frame.color,
                    depth,
                    intr,
                    plane,
                    cfg["colors"],
                    cube_size=height,
                    min_pixels=int(cfg["min_face_pixels"]),
                    min_footprint_fraction=float(cfg["min_footprint_fraction"]),
                    size_tolerance=float(cfg["size_tolerance"]),
                    zmin=float(cfg["depth_min"]),
                    zmax=float(cfg["depth_max"]),
                    rejects=rejects,
                    prepared=prepared,
                )
                if pose is None or pose.corners is None:
                    reason = next(iter(rejects), "miss")
                    lines.append(f"no cube ({reason})")
                else:
                    bottom = pose.bottom_center
                    body = _body_center(bottom, plane, height)
                    pixels = []
                    for corner in pose.corners:
                        pix = _pixel(corner, intr)
                        if pix is not None:
                            pixels.append(pix)
                    if len(pixels) >= 2:
                        cv2.polylines(vis, [np.array(pixels, dtype=np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
                    bottom_px = _pixel(bottom, intr)
                    body_px = _pixel(body, intr)
                    if bottom_px is not None:
                        cv2.circle(vis, bottom_px, 5, (255, 255, 0), 1, cv2.LINE_AA)
                    if body_px is not None:
                        _cross(vis, body_px, (0, 255, 255), 18)
                    readout = _ground_readout(bottom, plane, pose.edge_direction, height)
                    if readout is None:
                        lines.append("camera looks straight down")
                    else:
                        ahead, side, above_top, _ground_yaw = readout
                        lines.append(f"ahead={ahead:+.3f} m   right={side:+.3f} m")
                        lines.append(f"above top={above_top:+.3f} m")
                    mount_r, mount_t = camera_corner_extrinsic()
                    world = camera_to_world(body, mount_r, mount_t)
                    edge_w = mount_r.T @ np.asarray(pose.edge_direction, dtype=float)
                    yaw = fold_yaw_quarter(float(np.arctan2(edge_w[1], edge_w[0])))
                    lines.append(f"yaw={np.degrees(yaw):+.1f} deg")
                    lines.append(
                        f"world x={float(world[0]):+.3f} m   y={float(world[1]):+.3f} m   z={float(world[2]):+.3f} m"
                    )
            _label(vis, lines)
            chart = render_color_standard(
                cfg["colors"],
                lock["name"],
                cfg.get("light_levels"),
                measure_scene_hsv(frame.color, cfg["colors"]),
            )
            show_color_standard(chart, pump=False)
            # 先缩小再并排，避免每帧拼一张 2560 宽的图再缩回去。
            half = (vis.shape[1] // 2, vis.shape[0] // 2)
            preview = np.hstack(
                [
                    cv2.resize(vis, half, interpolation=cv2.INTER_AREA),
                    cv2.cvtColor(cv2.resize(mask_view, half, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR),
                ]
            )
            cv2.imshow("cube center", preview)
            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-3)
            last = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
