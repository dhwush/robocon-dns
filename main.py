"""Estimate one ground cube's world x, y, and yaw from a D435i."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from camera.source import Frame, RealSenseSource, filter_depth
from cube.detect import detect_visible_cube
from field.extrinsic import (
    calibrate_from_image,
    identity_extrinsic,
    load_extrinsic,
    save_extrinsic,
)
from fuse.median import average_scores, fuse_poses, score_trial
from geom.transform import camera_to_world, load_intrinsics
from ground.plane import fit_ground, sample_points


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pose_to_world(bottom_center: np.ndarray, yaw: float, rotation: np.ndarray, translation: np.ndarray) -> dict:
    world = camera_to_world(bottom_center, rotation, translation)
    return {"x": float(world[0]), "y": float(world[1]), "z": float(world[2]), "yaw": float(yaw)}


def estimate_frame(frame: Frame, intr, cfg: dict, rotation: np.ndarray, translation: np.ndarray):
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
    )
    if plane is None:
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
    )
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
    return pose_to_world(pose.bottom_center, pose.yaw, rotation, translation)


def resolve_extrinsic(cfg: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "basic":
        return identity_extrinsic()
    path = Path(cfg["extrinsic_path"])
    if not path.exists():
        raise SystemExit(f"missing extrinsic cache {path}; run with --calibrate")
    return load_extrinsic(path)


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
    print("basic (camera frame is the world frame)")
    for center, yaw in targets:
        color, depth = render_cube(intr, center, yaw, noise_m=0.004, seed=1, top_bgr=(255, 80, 40))
        samples = []
        for _ in range(5):
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
            f"pos={trial['position_score']} yaw={trial['yaw_score']}"
        )
    summary = average_scores(basic_trials)
    print(
        f"basic average position={summary['position']:.2f} yaw={summary['yaw']:.2f} "
        f"passed={summary['basic_passed']}"
    )
    if not summary["basic_passed"]:
        print("basic requirement is below 4; advanced stations are not scored")
        return 1

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
                f"pos={trial['position_score']} yaw={trial['yaw_score']}"
            )
    if advanced:
        adv = average_scores(advanced)
        print(f"advanced average position={adv['position']:.2f} yaw={adv['yaw']:.2f}")
    return 0 if summary["basic_passed"] else 1


def render_cube_in_world(intr, rotation, translation, center, yaw):
    """Render a world-frame cube by expressing the camera as identity in a shifted scene.

    The detector's basic images are top-down. For a moved camera, sample the cube
    in the camera frame and reuse the top-down painter via an explicit raster.
    """
    from sim.scene import _face, _paint

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


def run_live(cfg: dict, intr, mode: str, calibrate: bool) -> None:
    source = RealSenseSource(intr.width, intr.height)
    rotation, translation = identity_extrinsic() if mode == "basic" else (None, None)
    samples: list[dict] = []
    try:
        while True:
            frame = source.read()
            if calibrate:
                depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
                gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 1.0, 0.0])
                plane = fit_ground(sample_points(depth, intr), gravity)
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
            sample = estimate_frame(frame, intr, cfg, rotation, translation)
            if sample is None:
                print("frame rejected")
                continue
            samples.append(sample)
            if len(samples) >= int(cfg["frames_to_fuse"]):
                fused = fuse_poses(samples[-int(cfg["frames_to_fuse"]) :])
                if fused is not None:
                    print(f"{fused.x:.4f} {fused.y:.4f} {np.degrees(fused.yaw):.2f}")
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cube world pose from a D435i")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--intrinsics", default="config/intrinsics.yaml")
    parser.add_argument("--mode", choices=("basic", "field"), default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "basic")
    if args.self_check:
        raise SystemExit(self_check(cfg))
    intr = load_intrinsics(args.intrinsics)
    run_live(cfg, intr, mode, args.calibrate)


if __name__ == "__main__":
    main()
