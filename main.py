"""入口：由 D435i 的一帧估计地面方块的世界系 x、y 和 yaw。

作业基础要求只要求相机在世界原点，这里用单位旋转和零平移，世界系就是相机系。
文档 2.3 那组固定旋转只在相机水平向前、光轴对齐场地 X 时成立；安装方向不同，不用那组矩阵。
提高要求只要求相机挪走后仍给出世界系位姿。外参来自场地边标定，不用 AprilTag。

方块位置和 yaw 都来自地面足迹，不拟合上表面。上表面 RMS、与地面的 12° 夹角、
6 mm 正方形残差没有对应平面，不参与丢帧。足迹的长边、短边和面积仍然丢帧。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from camera.source import Frame, RealSenseSource, filter_depth
from cube.detect import calibrate_hsv, detect_visible_cube, prepare_mask
from field.extrinsic import (
    calibrate_from_image,
    identity_extrinsic,
    load_extrinsic,
    save_extrinsic,
)
from fuse.median import average_scores, fuse_poses, score_trial
from geom.transform import camera_to_world, load_intrinsics
from ground.plane import fit_ground, sample_points


# --- 单帧估计 -----------------------------------------------------------------
# 深度清洗 → 地面 → 可见面位姿 →（可选）看不清面时的补救 → 变到世界系。
# enable_face_fallback 默认关闭，基本项目不调用补救。

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    hsv_path = Path(cfg.get("hsv_path", "config/hsv.yaml"))
    if hsv_path.exists():
        with open(hsv_path, "r", encoding="utf-8") as handle:
            saved = yaml.safe_load(handle) or {}
        if saved.get("colors"):
            cfg["colors"] = saved["colors"]
    return cfg


def _bump(rejects: dict | None, reason: str) -> None:
    if rejects is not None:
        rejects[reason] = int(rejects.get(reason, 0)) + 1


def _ground_up_ok(plane, rotation: np.ndarray) -> bool:
    # 单位外参时世界轴跟着相机走，不拿地面法向去对文档 2.3 的 +Z_W。
    # 相机已经挪走、外参来自场地边时，世界 Z 向上，法向点积应不小于 0.95。
    if np.allclose(rotation, np.eye(3), atol=1e-5):
        return True
    up = rotation.T @ plane.normal
    return float(up[2]) >= 0.95


def pose_to_world(pose, rotation: np.ndarray, translation: np.ndarray, plane, height: float) -> dict:
    # 足迹中心在地面上，x、y 就是几何中心的水平位置。沿地面法向抬 h/2 得到几何中心。
    # 没有上表面，不能按顶面公式再减一次半高。
    center = pose.bottom_center + plane.normal * (0.5 * height)
    world = camera_to_world(center, rotation, translation)
    return {
        "x": float(world[0]),
        "y": float(world[1]),
        "z": float(world[2]),
        "yaw": float(pose.yaw),
        "inliers": int(pose.inliers),
        "quarter": not pose.front_color,
        "corners_cam": pose.corners,
        "plane_origin": plane.origin(),
    }


def estimate_frame(
    frame: Frame,
    intr,
    cfg: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
    rejects: dict | None = None,
):
    # 深度先清洗。没有 IMU 时按俯视自检的约定，重力沿相机 +Z。有加速度计就用实测重力，不假设光轴朝向。
    depth = filter_depth(
        frame.depth,
        zmin=float(cfg["depth_min"]),
        zmax=float(cfg["depth_max"]),
        flying_pixel_m=float(cfg["flying_pixel_m"]),
    )
    gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 0.0, 1.0])
    cloud = sample_points(depth, intr, stride=4)
    plane = fit_ground(
        cloud,
        gravity,
        thresh=float(cfg["ground_ransac_thresh"]),
        max_angle_deg=float(cfg["ground_max_angle_deg"]),
        iterations=int(cfg.get("ground_ransac_iterations", 80)),
    )
    # 地面没拟合上就整帧作废，避免在错误平面上给出 yaw。
    if plane is None:
        _bump(rejects, "ground")
        return None
    if not _ground_up_ok(plane, rotation):
        _bump(rejects, "world_up")
        return None
    pose = detect_visible_cube(
        frame.color,
        depth,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        min_pixels=int(cfg["min_face_pixels"]),
        min_footprint_fraction=float(cfg["min_footprint_fraction"]),
        size_tolerance=float(cfg["size_tolerance"]),
        rotation=rotation,
        translation=translation,
        zmin=float(cfg["depth_min"]),
        zmax=float(cfg["depth_max"]),
        rejects=rejects,
    )
    # 补救默认关闭。打开后也只在可见面检测拒绝时才跑，不改已经成功的结果。
    if pose is None and cfg.get("enable_face_fallback"):
        from cube.face_fallback import fit_when_face_missing

        pose = fit_when_face_missing(
            frame.color,
            depth,
            intr,
            plane,
            cfg["colors"],
            cube_size=float(cfg["cube_size"]),
            rotation=rotation,
            translation=translation,
        )
    if pose is None:
        return None
    world = pose_to_world(pose, rotation, translation, plane, float(cfg["cube_size"]))
    # 平面上一点用于判断相机或地面是否跳了 30 mm。不写进对外的 JSON。
    world["plane_origin"] = plane.origin()
    return world


# --- 外参来源 -----------------------------------------------------------------
# basic 固定为单位阵，表示相机就在世界原点。field 读取场地边标定的缓存，不用 AprilTag。

def resolve_extrinsic(cfg: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "basic":
        return identity_extrinsic()
    path = Path(cfg["extrinsic_path"])
    if not path.exists():
        raise SystemExit(f"missing extrinsic cache {path}; run with --calibrate")
    return load_extrinsic(path)


# --- 合成自检 -----------------------------------------------------------------
# 不接相机。先测相机在原点的三个方块；平均分达到 4 再测三个相机点位。
# 每个点位先对场地边求外参，再识别三个方块。

def self_check(cfg: dict) -> int:
    from geom.transform import compose_R
    from sim.scene import intrinsics_for, render_cube, render_field

    intr = intrinsics_for(640, 360)
    gravity = np.array([0.0, 0.0, 1.0])
    basic_trials = []
    targets = [
        (np.array([0.70, 0.35, 2.0]), 0.35),
        (np.array([-0.45, 0.50, 2.0]), -0.4),
        (np.array([0.30, -0.55, 2.0]), 0.2),
    ]
    rotation, translation = identity_extrinsic()
    # 基础：三个俯视方块，外参为单位阵。同一张图重复估计再取中值，模拟多帧。
    print("basic (camera frame is the world frame)")
    for center, yaw in targets:
        color, depth = render_cube(intr, center, yaw, noise_m=0.004, seed=1, top_bgr=(255, 80, 40))
        samples = []
        for _ in range(int(cfg.get("min_samples", 8))):
            sample = estimate_frame(Frame(color, depth, gravity), intr, cfg, rotation, translation)
            if sample is not None:
                samples.append(sample)
        fused = fuse_poses(samples)
        if fused is None:
            print(f"  rejected center={center[:2]} yaw={yaw:.3f}")
            basic_trials.append(
                {"position_score": 0, "yaw_score": 0, "position_error": 1.0, "yaw_error": 1.0}
            )
            continue
        trial = score_trial(
            np.array([fused.x, fused.y]),
            fused.yaw,
            center[:2],
            yaw,
        )
        basic_trials.append(trial)
        print(
            f"  est=({fused.x:.3f}, {fused.y:.3f}, yaw={fused.yaw:.3f}) "
            f"gt=({center[0]:.3f}, {center[1]:.3f}, yaw={yaw:.3f}) "
            f"abs=({trial['abs_x']:.3f}, {trial['abs_y']:.3f}, {trial['abs_yaw']:.3f}) "
            f"pos={trial['position_score']} yaw={trial['yaw_score']}"
        )
        if trial.get("sensitive_percent"):
            print("  真值落在 0.15 m 内，百分比对小坐标过敏")
    summary = average_scores(basic_trials)
    print(
        f"basic average position={summary['position']:.2f} yaw={summary['yaw']:.2f} "
        f"passed={summary['basic_passed']}"
    )
    # 作业规定基础不到 4 分就不能做提高要求。
    if not summary["basic_passed"]:
        print("basic requirement is below 4; advanced stations are not scored")
        return 1

    # 提高：三个相机位姿。每个点位先看场地边求 R、t，再识别三个方块。
    print("advanced (field edges define the world frame)")
    stations = [
        compose_R(0.0, -0.45, 0.08),
        compose_R(0.25, -0.4, -0.05),
        compose_R(-0.2, -0.5, 0.1),
    ]
    translations = [
        np.array([0.4, 0.3, 1.6]),
        np.array([-0.2, 0.5, 1.8]),
        np.array([0.6, -0.2, 1.5]),
    ]
    cubes = [
        (np.array([0.8, 0.4, 0.0]), 0.3),
        (np.array([1.3, 0.7, 0.0]), -0.5),
        (np.array([0.5, 0.9, 0.0]), 0.15),
    ]
    advanced = []
    for rotation, translation in zip(stations, translations):
        # 世界重力是 -Z。变到这个相机位姿后，地面法向才能和重力对齐。
        gravity_cam = rotation @ np.array([0.0, 0.0, -1.0])
        field_color, field_depth = render_field(intr, rotation, translation, cfg["field_length"], cfg["field_width"])
        cloud = sample_points(filter_depth(field_depth, 0.3, 3.0, 0.2), intr, stride=2)
        from ground.plane import fit_ground as fit

        plane = fit(cloud, gravity_cam)
        if plane is None:
            print("  station rejected: no ground")
            continue
        pose = calibrate_from_image(field_color, intr, plane, cfg["field_length"], cfg["field_width"])
        if pose is None:
            print("  station rejected: field edges not found")
            continue
        est_r, est_t = pose
        for center, yaw in cubes:
            color, depth = render_cube_in_world(intr, rotation, translation, center, yaw)
            sample = estimate_frame(Frame(color, depth, gravity_cam), intr, cfg, est_r, est_t)
            if sample is None:
                advanced.append({"position_score": 0, "yaw_score": 0, "position_error": 1.0, "yaw_error": 1.0})
                print(f"  cube rejected gt={center[:2]}")
                continue
            trial = score_trial(np.array([sample["x"], sample["y"]]), sample["yaw"], center[:2], yaw)
            advanced.append(trial)
            print(
                f"  est=({sample['x']:.3f}, {sample['y']:.3f}, yaw={sample['yaw']:.3f}) "
                f"gt=({center[0]:.3f}, {center[1]:.3f}, yaw={yaw:.3f}) "
                f"abs=({trial['abs_x']:.3f}, {trial['abs_y']:.3f}, {trial['abs_yaw']:.3f}) "
                f"pos={trial['position_score']} yaw={trial['yaw_score']}"
            )
    if advanced:
        adv = average_scores(advanced)
        print(f"advanced average position={adv['position']:.2f} yaw={adv['yaw']:.2f}")
    return 0 if summary["basic_passed"] else 1


def render_cube_in_world(intr, rotation, translation, center, yaw):
    """按给定外参把世界系方块画进相机画面，供提高要求的自检使用。"""
    from sim.scene import _face, _paint

    # 先铺一块世界系地面，再把方块六个面变到相机系后光栅化。相机后方的点丢掉。
    image = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    xs = np.linspace(center[0] - 1.2, center[0] + 1.2, 70)
    ys = np.linspace(center[1] - 1.2, center[1] + 1.2, 70)
    gx, gy = np.meshgrid(xs, ys)
    world = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    from geom.transform import world_to_camera

    ground = world_to_camera(world, rotation, translation)
    keep = ground[:, 2] > 0.2
    _paint(ground[keep], np.full((int(keep.sum()), 3), 40, dtype=np.uint8), intr, depth, image)
    size = 0.6
    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = rotation @ np.array([c, s, 0.0]) * size
    axis_y = rotation @ np.array([-s, c, 0.0]) * size
    axis_z = rotation @ np.array([0.0, 0.0, size])
    bottom = world_to_camera(center, rotation, translation)
    top = bottom + axis_z
    blue = (255, 80, 40)
    for pts, cols in (
        _face(top, axis_x, axis_y, blue),
        _face(bottom + axis_x * 0.5, axis_y, axis_z, blue),
        _face(bottom - axis_x * 0.5, axis_y, axis_z, blue),
        _face(bottom + axis_y * 0.5, axis_x, axis_z, blue),
        _face(bottom - axis_y * 0.5, axis_x, axis_z, blue),
    ):
        facing = pts[:, 2] > 0.2
        _paint(pts[facing], cols[facing], intr, depth, image)
    return image, depth


def _yaw_std_deg(samples: list[dict]) -> float:
    yaw = np.array([s["yaw"] for s in samples], dtype=float)
    center = yaw[0]
    delta = np.arctan2(np.sin(yaw - center), np.cos(yaw - center))
    return float(np.degrees(np.std(delta)))


def _record(fused, samples: list[dict]) -> dict:
    # yaw 用弧度。质量里只留内点数和 yaw 散布。
    # 上表面 RMS、12° 法向、6 mm 正方形残差没有平面可算，不放进 quality，也不用它们丢帧。
    return {
        "frame": "world",
        "x": fused.x,
        "y": fused.y,
        "z": fused.z,
        "yaw": fused.yaw,
        "stamp_ms": int(time.time() * 1000),
        "quality": {
            "inliers": int(np.median([s.get("inliers", 0) for s in samples])),
            "yaw_std_deg": _yaw_std_deg(samples),
        },
    }


def _show_debug(color: np.ndarray, depth: np.ndarray, cfg: dict, record: dict, intr, corners) -> None:
    import cv2

    from geom.transform import project_points

    prepared = prepare_mask(color, depth, cfg["colors"], float(cfg["depth_min"]), float(cfg["depth_max"]))
    vis = color.copy()
    if prepared is not None:
        contours, _ = cv2.findContours(prepared[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    # 绿色轮廓是颜色掩膜，青色是地面足迹。没有上表面内点，也不画标签。
    if corners is not None:
        pixels = np.rint(project_points(corners, intr)).astype(np.int32)
        cv2.polylines(vis, [pixels], True, (255, 255, 0), 2, cv2.LINE_AA)
    text = f"x={record['x']:.3f} y={record['y']:.3f} yaw={record['yaw']:.3f}"
    cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("cube", vis)
    cv2.waitKey(1)


# --- 实机循环 -----------------------------------------------------------------
# --calibrate 保存场地边外参。--calibrate-color 用 30 帧收紧 HSV。
# 合格样本满 8 帧才输出 JSON。地面或外参平移跳过 30 mm 时清空缓冲。
# 失败后 0.5 s 内重复上一结果，再失败则输出空对象。丢帧原因累加后写到标准错误。

def run_live(cfg: dict, intr, mode: str, calibrate: bool, calibrate_color: bool, debug: bool) -> None:
    source = RealSenseSource(intr.width, intr.height, fps=int(cfg.get("fps", 30)))
    rotation, translation = identity_extrinsic() if mode == "basic" else (None, None)
    samples: list[dict] = []
    color_frames: list[tuple[np.ndarray, np.ndarray]] = []
    last_record: dict | None = None
    fail_since: float | None = None
    last_print = 0.0
    last_plane = None
    last_translation = None
    rejects: dict[str, int] = {}
    logged_rejects = {}
    hold_s = float(cfg.get("hold_s", 0.5))
    jump_m = float(cfg.get("jump_m", 0.03))
    min_samples = int(cfg.get("min_samples", 8))
    window = int(cfg.get("frames_to_fuse", 15))
    output_period = 1.0 / float(cfg.get("output_hz", 10))
    try:
        while True:
            frame = source.read()
            depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
            if calibrate_color:
                color_frames.append((frame.color.copy(), depth))
                if len(color_frames) < int(cfg.get("hsv_calib_frames", 30)):
                    continue
                colors = calibrate_hsv(color_frames, cfg["colors"], cfg["depth_min"], cfg["depth_max"])
                path = Path(cfg.get("hsv_path", "config/hsv.yaml"))
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump({"colors": colors}, handle, allow_unicode=True)
                print(f"saved hsv to {path}")
                return
            if calibrate:
                # 没有加速度计时才用这个方向。图像向下当重力，只为场地边标定能拟合地面，不是 2.3 的旋转矩阵。
                gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 1.0, 0.0])
                plane = fit_ground(
                    sample_points(depth, intr),
                    gravity,
                    thresh=float(cfg["ground_ransac_thresh"]),
                    max_angle_deg=float(cfg["ground_max_angle_deg"]),
                    iterations=int(cfg.get("ground_ransac_iterations", 80)),
                )
                if plane is None:
                    print("ground not found")
                    continue
                pose = calibrate_from_image(frame.color, intr, plane, cfg["field_length"], cfg["field_width"])
                if pose is None:
                    print("field edges not found")
                    continue
                save_extrinsic(cfg["extrinsic_path"], pose[0], pose[1])
                print(f"saved extrinsic to {cfg['extrinsic_path']}")
                return
            if rotation is None:
                rotation, translation = resolve_extrinsic(cfg, mode)
            sample = estimate_frame(frame, intr, cfg, rotation, translation, rejects)
            now = time.monotonic()
            corners = None if sample is None else sample.pop("corners_cam", None)
            origin = None if sample is None else np.asarray(sample.pop("plane_origin"), dtype=float)
            moved = False
            if origin is not None and last_plane is not None and np.linalg.norm(origin - last_plane) > jump_m:
                moved = True
            if translation is not None and last_translation is not None:
                if np.linalg.norm(np.asarray(translation) - last_translation) > jump_m:
                    moved = True
            if moved:
                samples.clear()
                last_record = None
            if origin is not None:
                last_plane = origin
            if translation is not None:
                last_translation = np.asarray(translation, dtype=float).copy()
            if sample is None:
                if fail_since is None:
                    fail_since = now
                if last_record is not None and now - fail_since <= hold_s:
                    payload = last_record
                else:
                    payload = {}
            else:
                fail_since = None
                samples.append(sample)
                samples[:] = samples[-window:]
                payload = None
                if len(samples) >= min_samples:
                    fused = fuse_poses(samples)
                    if fused is not None:
                        last_record = _record(fused, samples)
                        payload = last_record
            if payload is not None and now - last_print >= output_period:
                print(json.dumps(payload))
                last_print = now
                if debug and sample is not None and payload:
                    _show_debug(frame.color, depth, cfg, payload, intr, corners)
            if rejects != logged_rejects:
                print(json.dumps({"reject_counts": rejects}), file=sys.stderr)
                logged_rejects = dict(rejects)
    finally:
        source.close()


def main() -> None:
    # --self-check 不打开相机。其余路径读取配置内参后进入实机循环。
    parser = argparse.ArgumentParser(description="Cube world pose from a D435i")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--intrinsics", default="config/intrinsics.yaml")
    parser.add_argument("--mode", choices=("basic", "field"), default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-color", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "basic")
    if args.self_check:
        raise SystemExit(self_check(cfg))
    intr = load_intrinsics(args.intrinsics)
    run_live(cfg, intr, mode, args.calibrate, args.calibrate_color, args.debug)


if __name__ == "__main__":
    main()
