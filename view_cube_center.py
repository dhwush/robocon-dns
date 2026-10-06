"""实时查看方块识别和体心。

体心是地面足迹中心沿地面法向抬起半个边长，与正式输出的几何中心同一点。
这一路不读、不画棱方向和 yaw。按 q 或 Esc 退出。

    python view_cube_center.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from camera.source import RealSenseSource, filter_depth
from cube.detect import detect_visible_cube, prepare_mask
from geom.transform import load_intrinsics, project_points
from ground.plane import fit_ground, sample_points
from main import load_config


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
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def _body_center(bottom: np.ndarray, plane, height: float) -> np.ndarray:
    # 足迹中心在地面上。沿法向抬半高就是体心，不再用顶面去减一次半高。
    return np.asarray(bottom, dtype=float) + plane.normal * (0.5 * height)


def main() -> None:
    cfg = load_config("config/default.yaml")
    intr = load_intrinsics("config/intrinsics.yaml")
    source = RealSenseSource(intr.width, intr.height, fps=int(cfg.get("fps", 30)))
    height = float(cfg["cube_size"])
    last = time.monotonic()
    try:
        while True:
            frame = source.read()
            depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
            gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 0.0, 1.0])
            plane = fit_ground(
                sample_points(depth, intr, stride=4),
                gravity,
                thresh=float(cfg["ground_ransac_thresh"]),
                max_angle_deg=float(cfg["ground_max_angle_deg"]),
                iterations=int(cfg.get("ground_ransac_iterations", 80)),
            )
            vis = frame.color.copy()
            prepared = None
            if plane is not None:
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
            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-3)
            last = now
            lines = [f"{fps:.0f} fps"]
            if plane is None:
                lines.append("no ground")
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
                    lines.append(f"body  x={body[0]:+.3f}  y={body[1]:+.3f}  z={body[2]:+.3f} m")
                    lines.append(f"floor x={bottom[0]:+.3f}  y={bottom[1]:+.3f}  z={bottom[2]:+.3f} m")
                    lines.append(f"inliers={pose.inliers}  footprint={pose.footprint_fraction:.2f}")
            _label(vis, lines)
            mask_bgr = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
            preview = np.hstack([vis, mask_bgr])
            scale = 1280 / preview.shape[1]
            if scale < 1.0:
                preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("cube center", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
