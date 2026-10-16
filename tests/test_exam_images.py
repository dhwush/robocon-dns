"""生成考试用 PNG，跑位姿流程，并写出叠加了估计结果的图。"""

from __future__ import annotations

import json

import cv2
import numpy as np

from camera.source import Frame, filter_depth
from field.extrinsic import calibrate_from_image, level_extrinsic
from fuse.median import average_scores, score_trial
from geom.transform import fold_yaw_quarter, load_intrinsics_for_serial, project_points, world_to_camera
from ground.plane import fit_ground, sample_points
from main import estimate_frame, load_config
from sim.exam_scene import (
    BLUE_BGR,
    EXAM_SERIAL,
    GREEN_BGR,
    RED_BGR,
    SCENE_DIR,
    VIS_DIR,
    WHITE_BGR,
    look_at,
    load_rgbd,
    render_field,
    render_nadir,
    save_rgbd,
)

FIELD_LENGTH = 2.0
FIELD_WIDTH = 1.2
CUBE = 0.2


# --- 场景描述 -----------------------------------------------------------------
# 基础要求：俯视、三个方块。提高要求：三个相机点位，每点三个方块。

def _cube(x, y, yaw, z=0.0):
    # 上半纯红、下半纯蓝，没有正面标记，yaw 按 90° 对称评分。
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "size": CUBE,
        "height": CUBE,
        "top": RED_BGR,
        "lower": BLUE_BGR,
        "front": None,
        "marked": False,
    }


def _distractor(x, y, yaw, color, size, height, z=0.0):
    # 干扰物不是 0.2 m 方块。颜色、边长或高度对不上时，检测应把它排除。
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "size": size,
        "height": height,
        "color": color,
        "top": color,
    }


def basic_cases() -> list[dict]:
    # 基础要求：俯视三个方块，远近两档高度，旁边放尺寸不对的干扰物。
    far_z = 1.65
    near_z = 1.45
    far_distractors = [
        _distractor(-0.72, -0.28, 0.4, GREEN_BGR, 0.26, 0.22, far_z),
        _distractor(0.68, -0.22, -0.2, WHITE_BGR, 0.32, 0.04, far_z),
    ]
    near_distractors = [
        _distractor(-0.48, -0.22, 0.35, GREEN_BGR, 0.24, 0.20, near_z),
        _distractor(-0.10, 0.42, -0.15, WHITE_BGR, 0.28, 0.04, near_z),
    ]
    specs = [
        (_cube(0.42, 0.12, 0.22, z=far_z), far_distractors, far_z),
        (_cube(-0.36, 0.16, -0.30, z=far_z), far_distractors, far_z),
        (_cube(0.55, 0.05, -2.84, z=near_z), near_distractors, near_z),
    ]
    return [
        {
            "name": f"basic_{index}",
            "cube": cube,
            "distractors": distractors,
            "ground_z": ground_z,
            "seed": 10 + index,
        }
        for index, (cube, distractors, ground_z) in enumerate(specs)
    ]


def advanced_stations() -> list[dict]:
    # 提高要求：三个相机点位看同一组方块。外参由场地边决定，不把相机放在原点。
    cubes = [
        _cube(0.72, 0.50, 0.22),
        _cube(0.78, 0.86, -0.18),
        _cube(1.20, 0.52, -2.45),
    ]
    distractors = [
        _distractor(1.62, 0.96, 0.5, GREEN_BGR, 0.22, 0.24),
        _distractor(1.68, 0.32, -0.25, WHITE_BGR, 0.28, 0.05),
    ]
    cameras = [
        (np.array([-0.55, -0.45, 1.35]), np.array([1.05, 0.60, 0.0])),
        (np.array([-0.20, -0.90, 1.55]), np.array([1.10, 0.55, 0.0])),
        (np.array([0.20, -0.75, 1.20]), np.array([1.15, 0.65, 0.0])),
    ]
    stations = []
    for index, (camera, target) in enumerate(cameras):
        rotation, translation = look_at(camera, target)
        stations.append(
            {
                "name": f"station_{index}",
                "camera": camera,
                "rotation": rotation,
                "translation": translation,
                "cubes": cubes,
                "distractors": distractors,
                "seed": 30 + index,
            }
        )
    return stations


def write_scenes(intr) -> None:
    # 写出彩色、深度和真值。提高要求的标定帧不放方块，避免边被方块挡住。
    for case in basic_cases():
        color, depth = render_nadir(
            intr,
            [case["cube"]],
            case["distractors"],
            ground_z=case["ground_z"],
            seed=case["seed"],
        )
        save_rgbd(SCENE_DIR, case["name"], color, depth)
        # 图仍按俯视光学系画。真值改到水平世界系：向前是光学 Y 的反向，yaw 跟着反号。
        flat = dict(case["cube"])
        flat["y"] = -float(flat["y"])
        flat["yaw"] = -float(flat["yaw"])
        _write_gt(case["name"], flat)
    for station in advanced_stations():
        color, depth = render_field(
            intr,
            station["rotation"],
            station["translation"],
            [],
            station["distractors"],
            FIELD_LENGTH,
            FIELD_WIDTH,
            seed=station["seed"],
        )
        save_rgbd(SCENE_DIR, f"{station['name']}_calib", color, depth)
        for index, cube in enumerate(station["cubes"]):
            color, depth = render_field(
                intr,
                station["rotation"],
                station["translation"],
                [cube],
                station["distractors"],
                FIELD_LENGTH,
                FIELD_WIDTH,
                seed=station["seed"] + 10 + index,
            )
            name = f"{station['name']}_cube_{index}"
            save_rgbd(SCENE_DIR, name, color, depth)
            _write_gt(name, cube)


def _write_gt(name: str, cube: dict) -> None:
    # 真值单独存 JSON，评分时不从渲染参数里反推，避免和图像生成耦在一起。
    payload = {
        "x": cube["x"],
        "y": cube["y"],
        "z": cube["z"],
        "yaw": cube["yaw"],
        "marked": cube["marked"],
        "size": cube["size"],
    }
    with open(SCENE_DIR / f"{name}_gt.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_gt(name: str) -> dict:
    # 读回这一帧的世界系真值，供相对误差分档。
    with open(SCENE_DIR / f"{name}_gt.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _frame(name: str, gravity: np.ndarray, cfg: dict | None = None) -> Frame:
    # 把存盘的彩色和深度装成与实机相同的 Frame，重力由机位决定。
    # 估计前先洗深度，和实机循环把清洗留在循环里的做法一致。
    color, depth = load_rgbd(SCENE_DIR, name)
    if cfg is not None:
        depth = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    return Frame(color, depth, gravity)


# --- 打分与可视化 -------------------------------------------------------------
# 按作业分档比较 x、y 和 yaw。方框画在地面高度上，因为输出的 z 是几何中心，不是足迹所在的地面。

def _score_pose(est: dict | None, gt: dict) -> dict:
    if est is None:
        return {"position_error": 1.0, "yaw_error": 1.0, "position_score": 0, "yaw_score": 0, "est": None}
    if gt["marked"]:
        est_yaw, gt_yaw = est["yaw"], gt["yaw"]
        quarter = False
    else:
        est_yaw, gt_yaw = fold_yaw_quarter(est["yaw"]), fold_yaw_quarter(gt["yaw"])
        quarter = True
    trial = score_trial(
        np.array([est["x"], est["y"]]),
        est_yaw,
        np.array([gt["x"], gt["y"]]),
        gt_yaw,
        quarter=quarter,
    )
    trial["est"] = {"x": est["x"], "y": est["y"], "z": est["z"], "yaw": est["yaw"]}
    return trial


def _square(center: np.ndarray, yaw: float, size: float) -> np.ndarray:
    # 地面高度上的正方形四角。叠加图画在足迹所在平面，不用几何中心的 z。
    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = np.array([c, s, 0.0]) * size * 0.5
    axis_y = np.array([-s, c, 0.0]) * size * 0.5
    return np.vstack(
        [
            center - axis_x - axis_y,
            center + axis_x - axis_y,
            center + axis_x + axis_y,
            center - axis_x + axis_y,
        ]
    )


def _draw_square(image, intr, points_cam, bgr) -> None:
    # 相机后方的角不画，避免投影出现在画面对面。
    if np.any(points_cam[:, 2] <= 0.05):
        return
    pixels = np.rint(project_points(points_cam, intr)).astype(np.int32)
    cv2.polylines(image, [pixels], True, bgr, 2, cv2.LINE_AA)


def _overlay(image, lines: list[str]) -> np.ndarray:
    # 在图上写估计值和真值。先描黑边再写白字，深色和浅色背景都能看清。
    canvas = image.copy()
    for index, line in enumerate(lines):
        origin = (16, 32 + 28 * index)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _save_vis(name: str, image: np.ndarray) -> None:
    # 叠加结果单独放到 vis 目录，不覆盖用来检测的原始场景图。
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(VIS_DIR / f"{name}.png"), image)


def _basic_overlay(intr, name: str, gt: dict, trial: dict) -> None:
    # 俯视时相机系就是世界系。青色是真值足迹，绿色是估计足迹。
    color, _depth = load_rgbd(SCENE_DIR, name)
    gt_square = _square(np.array([gt["x"], gt["y"], gt["z"]]), gt["yaw"], gt["size"])
    _draw_square(color, intr, gt_square, (0, 255, 255))
    est = trial["est"]
    lines = [f"gt  x={gt['x']:.3f} y={gt['y']:.3f} yaw={np.degrees(gt['yaw']):.1f} deg"]
    if est is not None:
        est_square = _square(np.array([est["x"], est["y"], gt["z"]]), est["yaw"], gt["size"])
        _draw_square(color, intr, est_square, (0, 255, 0))
        lines.insert(0, f"est x={est['x']:.3f} y={est['y']:.3f} yaw={np.degrees(est['yaw']):.1f} deg")
        lines.append(f"score pos={trial['position_score']} yaw={trial['yaw_score']}")
    else:
        lines.insert(0, "rejected")
    _save_vis(name, _overlay(color, lines))


def _world_overlay(intr, rotation, translation, name: str, gt: dict | None, trial: dict | None) -> None:
    # 斜视时机位不在原点，方框要先变到相机系再画。
    color, _depth = load_rgbd(SCENE_DIR, name)
    if gt is not None:
        gt_square = _square(np.array([gt["x"], gt["y"], gt["z"]]), gt["yaw"], gt["size"])
        _draw_square(color, intr, world_to_camera(gt_square, rotation, translation), (0, 255, 255))
    est = None if trial is None else trial.get("est")
    if est is not None and gt is not None:
        est_square = _square(np.array([est["x"], est["y"], gt["z"]]), est["yaw"], gt["size"])
        _draw_square(color, intr, world_to_camera(est_square, rotation, translation), (0, 255, 0))
    if trial is None:
        lines = ["calibration frame"]
    elif est is None or gt is None:
        lines = ["rejected"]
    else:
        lines = [
            f"est x={est['x']:.3f} y={est['y']:.3f} yaw={np.degrees(est['yaw']):.1f} deg",
            f"gt  x={gt['x']:.3f} y={gt['y']:.3f} yaw={np.degrees(gt['yaw']):.1f} deg",
            f"score pos={trial['position_score']} yaw={trial['yaw_score']}",
        ]
    _save_vis(name, _overlay(color, lines))


def _fit_plane(frame: Frame, intr, cfg):
    # 标定帧先拟合地面，场地边的端点才能交到这个平面上。
    depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 0.0, 1.0])
    return fit_ground(
        sample_points(depth, intr, stride=int(cfg.get("ground_stride", 8))),
        gravity,
        thresh=float(cfg["ground_ransac_thresh"]),
        max_angle_deg=float(cfg["ground_max_angle_deg"]),
        iterations=int(cfg.get("ground_ransac_iterations", 50)),
    )


def test_exam_images_match_assignment():
    # 基础三项平均要到 4 分。提高要求每个点位先求外参，再给三个方块打分。
    intr = load_intrinsics_for_serial(EXAM_SERIAL)
    cfg = load_config("config/default.yaml")
    write_scenes(intr)

    basic_trials = []
    gravity = np.array([0.0, 0.0, 1.0])
    level_r, level_t = level_extrinsic(gravity)
    for case in basic_cases():
        gt = _read_gt(case["name"])
        est = estimate_frame(_frame(case["name"], gravity, cfg), intr, cfg, level_r, level_t)
        trial = _score_pose(est, gt)
        basic_trials.append(trial)
        _basic_overlay(intr, case["name"], gt, trial)
    basic_summary = average_scores(basic_trials)
    assert basic_summary["basic_passed"], basic_summary

    advanced_trials = []
    station_reports = []
    for station in advanced_stations():
        gravity_cam = station["rotation"] @ np.array([0.0, 0.0, -1.0])
        calib_name = f"{station['name']}_calib"
        plane = _fit_plane(_frame(calib_name, gravity_cam), intr, cfg)
        assert plane is not None, station["name"]
        pose = calibrate_from_image(
            load_rgbd(SCENE_DIR, calib_name)[0],
            intr,
            plane,
            FIELD_LENGTH,
            FIELD_WIDTH,
        )
        assert pose is not None, station["name"]
        rotation, translation = pose
        found = -rotation.T @ translation
        assert np.linalg.norm(found - station["camera"]) < 0.05, (station["name"], found, station["camera"])
        _world_overlay(intr, rotation, translation, calib_name, None, None)
        station_trials = []
        for index in range(3):
            name = f"{station['name']}_cube_{index}"
            gt = _read_gt(name)
            est = estimate_frame(_frame(name, gravity_cam, cfg), intr, cfg, rotation, translation)
            trial = _score_pose(est, gt)
            station_trials.append(trial)
            advanced_trials.append(trial)
            _world_overlay(intr, rotation, translation, name, gt, trial)
        station_reports.append({"name": station["name"], "summary": average_scores(station_trials)})
    advanced_summary = average_scores(advanced_trials)
    report = {"basic": basic_summary, "advanced": advanced_summary, "stations": station_reports}
    with open(SCENE_DIR.parent / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    assert advanced_summary["position"] >= 4.0, report
    assert advanced_summary["yaw"] >= 4.0, report
