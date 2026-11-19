# 绿线在 2 到 3 个物体之间跳时，留下稳定的那一块方块。

import numpy as np

from cube.detect import detect_cube_candidates
from cube.rival import MISS_FRAMES, RivalPicker


def _sample(x, y, short=0.19, long=0.20, peak=0.20, fraction=0.8):
    return {
        "x": x,
        "y": y,
        "z": 0.1,
        "yaw": 0.2,
        "quarter": True,
        "short_side": short,
        "long_side": long,
        "peak": peak,
        "footprint_fraction": fraction,
    }


def test_two_objects_keep_the_stable_cube():
    """先锁定方块。之后方块和杂物同时出现时，仍留下方块，不取两者的中点。"""
    picker = RivalPicker()
    cube = _sample(0.20, 0.30)
    clutter = _sample(0.70, 0.30, short=0.06, long=0.22, peak=0.04, fraction=0.2)
    assert abs(picker.choose([cube])["x"] - 0.20) < 1e-6
    chosen = picker.choose([clutter, cube])
    assert chosen is not None
    assert abs(chosen["x"] - 0.20) < 0.05
    assert abs(chosen["y"] - 0.30) < 0.05


def test_three_objects_jump_stays_on_the_cube():
    """三块同时入画、绿线可以跳到另外两块上时，输出仍是已经稳定的方块。"""
    picker = RivalPicker()
    cube = _sample(0.20, 0.30)
    left = _sample(-0.25, 0.55, short=0.05, long=0.25, peak=0.06, fraction=0.25)
    right = _sample(0.80, 0.10, short=0.07, long=0.18, peak=0.03, fraction=0.3)
    picker.choose([cube])
    chosen = picker.choose([right, left, cube])
    assert chosen is not None
    assert abs(chosen["x"] - 0.20) < 0.05
    assert abs(chosen["y"] - 0.30) < 0.05


def test_alternating_green_box_is_not_fused_until_the_cube_is_gone():
    """方块和杂物隔帧出现时，杂物那一帧不输出。连续缺席达到门槛后才换物体。"""
    picker = RivalPicker()
    cube = _sample(0.20, 0.30)
    clutter = _sample(0.70, 0.40)
    picker.choose([cube])
    assert picker.choose([clutter]) is None
    assert picker.choose([cube])["x"] == 0.20
    for _ in range(MISS_FRAMES - 1):
        assert picker.choose([clutter]) is None
    switched = picker.choose([clutter])
    assert switched is not None
    assert abs(switched["x"] - 0.70) < 1e-6


def test_three_separate_cubes_each_keep_a_mask():
    """三块都在画面里、彼此隔开时，各自留下一条绿线和一个足迹，不并成一块。"""
    from camera.source import filter_depth
    from ground.plane import fit_ground, sample_points
    from main import load_config
    from sim.scene import intrinsics_for, render_cube

    cfg = load_config("config/default.yaml")
    intr = intrinsics_for(320, 240)
    size = float(cfg["cube_size"])
    centers = [np.array([-0.24, -0.08, 0.7]), np.array([0.24, -0.08, 0.7]), np.array([0.0, 0.16, 0.7])]
    color = None
    depth = None
    for index, center in enumerate(centers):
        top = (40, 40, 220) if index != 1 else (220, 40, 40)
        part_color, part_depth = render_cube(
            intr, center, 0.15 * index, size=size, ground_z=0.7, noise_m=0.0, seed=index, top_bgr=top
        )
        if color is None:
            color, depth = part_color, part_depth
            continue
        use = np.isfinite(part_depth) & ((~np.isfinite(depth)) | (part_depth < depth))
        color = color.copy()
        depth = depth.copy()
        color[use] = part_color[use]
        depth[use] = part_depth[use]
    depth_f = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    plane = fit_ground(sample_points(depth_f, intr, stride=3), np.array([0.0, 0.0, 1.0]))
    assert plane is not None
    poses = detect_cube_candidates(
        color,
        depth_f,
        intr,
        plane,
        cfg["colors"],
        cube_size=size,
        min_pixels=200,
        min_footprint_fraction=0.35,
        size_tolerance=float(cfg["size_tolerance"]),
    )
    assert len(poses) == 3
    found = [pose.bottom_center[:2] for pose in poses]
    for center in centers:
        assert any(np.linalg.norm(center[:2] - xy) < 0.04 for xy in found)


def test_without_history_a_thin_strip_loses_to_the_square():
    """还没有稳定中心时，短边太窄的一块让给足迹更像方块的那一块。"""
    picker = RivalPicker()
    strip = _sample(0.15, 0.15, short=0.04, long=0.20, peak=0.20, fraction=0.4)
    square = _sample(0.55, 0.40, short=0.19, long=0.20, peak=0.20, fraction=0.9)
    chosen = picker.choose([strip, square])
    assert chosen is not None
    assert abs(chosen["x"] - 0.55) < 1e-6
