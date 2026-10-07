# 流水线：飞点、地面重力约束、可见面位姿、场地角外参、多帧评分、看不清面时的补救。

import numpy as np

from camera.source import color_lock_from_config, filter_depth, remove_flying_pixels
from cube.detect import detect_visible_cube, prepare_mask
from cube.face_fallback import fit_when_face_missing
from field.extrinsic import Segment, estimate_from_lines, save_extrinsic, load_extrinsic
from fuse.median import angle_delta, average_scores, band_score, fuse_poses, score_trial
from geom.transform import compose_R, fold_yaw_quarter, world_to_camera
from ground.plane import Plane, fit_ground
from main import estimate_frame, load_config
from camera.source import Frame
from sim.scene import intrinsics_for, render_cube


def test_light_levels_change_exposure_gain_and_brightness():
    """亮度档改曝光、增益和亮度偏移。默认白天档保持当前锁定值，没有的档位不静默退回。"""
    cfg = load_config("config/default.yaml")
    daylight = color_lock_from_config(cfg)
    assert daylight["name"] == "daylight"
    assert daylight["exposure"] == 50
    assert daylight["gain"] == 16
    assert daylight["brightness"] == 0
    dark = color_lock_from_config(cfg, "dark")
    assert dark["exposure"] > daylight["exposure"]
    assert dark["gain"] > daylight["gain"]
    assert dark["brightness"] > daylight["brightness"]
    bright = color_lock_from_config(cfg, "bright")
    assert bright["exposure"] < daylight["exposure"]
    assert bright["brightness"] < daylight["brightness"]
    try:
        color_lock_from_config(cfg, "noon")
    except SystemExit:
        return
    raise AssertionError("unknown light level should be rejected")


def test_flying_pixels_removed():
    """一块平坦深度里只有中心像素跳了 1 m。飞点滤波应删掉它，邻域里的平坦像素保留。"""
    depth = np.ones((9, 9))
    depth[4, 4] = 2.0
    cleaned = remove_flying_pixels(depth, 0.2)
    assert not np.isfinite(cleaned[4, 4])
    assert np.isfinite(cleaned[1, 1])


def test_ground_rejects_tilted_normal():
    """重力沿 +Z 时，z = 1.5 的水平点云应拟合成功，点到平面的距离小于 1 cm。
    把高度改成随 x 倾斜约 22° 后，法向和重力夹角超过 8°，拟合必须拒绝。"""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, size=(400, 2))
    pts = np.column_stack([xy, np.full(len(xy), 1.5)])
    pts += rng.normal(0, 0.002, pts.shape)
    gravity = np.array([0.0, 0.0, 1.0])
    plane = fit_ground(pts, gravity, max_angle_deg=8.0)
    assert plane is not None
    assert abs(plane.signed_distance(np.array([0.0, 0.0, 1.5]))) < 0.01
    tilted = pts.copy()
    tilted[:, 2] = 1.5 + 0.4 * tilted[:, 0]
    assert fit_ground(tilted, gravity, max_angle_deg=8.0) is None


def test_visible_cube_pose_within_tolerance():
    """俯视 0.7 m 渲染一个带 4 mm 深度噪声的 0.2 m 方块。可见面检测的底面中心误差要小于 3 cm，
    yaw 按 90° 对称折返后与真值相差小于 3°。"""
    intr = intrinsics_for(320, 240)
    center = np.array([0.08, 0.06, 0.7])
    yaw = 0.35
    cfg = load_config("config/default.yaml")
    color, depth = render_cube(
        intr,
        center,
        yaw,
        size=float(cfg["cube_size"]),
        ground_z=float(center[2]),
        noise_m=0.004,
        seed=2,
        top_bgr=(255, 80, 40),
    )
    depth_f = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    from ground.plane import sample_points

    plane = fit_ground(sample_points(depth_f, intr, stride=3), np.array([0.0, 0.0, 1.0]))
    assert plane is not None
    pose = detect_visible_cube(
        color,
        depth_f,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        min_pixels=200,
        min_footprint_fraction=0.35,
        size_tolerance=float(cfg["size_tolerance"]),
    )
    assert pose is not None
    assert np.linalg.norm(pose.bottom_center[:2] - center[:2]) < 0.03
    assert abs(fold_yaw_quarter(pose.yaw - yaw)) < np.deg2rad(3.0)


def test_dim_cube_on_red_and_blue_ground():
    """地面半边暗红、半边暗蓝，方块本身也偏暗。底面中心仍要落在方块上，不能铺到地面。"""
    intr = intrinsics_for(320, 240)
    center = np.array([0.08, 0.06, 0.7])
    yaw = 0.35
    cfg = load_config("config/default.yaml")
    color, depth = render_cube(
        intr,
        center,
        yaw,
        size=float(cfg["cube_size"]),
        ground_z=float(center[2]),
        noise_m=0.004,
        seed=2,
        top_bgr=(255, 80, 40),
    )
    cube = np.any(color != 40, axis=2)
    ground = ~cube
    color[ground] = (90, 40, 30)
    right = ground.copy()
    right[:, : intr.width // 2] = False
    color[right] = (30, 25, 55)
    color[cube] = (30, 25, 55)
    depth_f = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    from ground.plane import sample_points

    plane = fit_ground(sample_points(depth_f, intr, stride=3), np.array([0.0, 0.0, 1.0]))
    assert plane is not None
    pose = detect_visible_cube(
        color,
        depth_f,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        min_pixels=200,
        min_footprint_fraction=0.35,
        size_tolerance=float(cfg["size_tolerance"]),
    )
    assert pose is not None
    assert np.linalg.norm(pose.bottom_center[:2] - center[:2]) < 0.04


def test_either_color_on_top_keeps_outer_contour():
    """红面在上或蓝面在上都要检出。顶面内部的深度跳变留在掩膜里，贴地的底边也还在。"""
    from geom.transform import deproject_pixels, project_points

    intr = intrinsics_for(320, 240)
    center = np.array([0.08, 0.06, 0.7])
    cfg = load_config("config/default.yaml")
    red = (40, 40, 220)
    blue = (220, 40, 40)
    for top_bgr, side_bgr in ((red, blue), (blue, red)):
        color, depth = render_cube(
            intr,
            center,
            0.35,
            size=float(cfg["cube_size"]),
            ground_z=float(center[2]),
            noise_m=0.0,
            top_bgr=top_bgr,
        )
        depth_f = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
        from ground.plane import sample_points

        plane = fit_ground(sample_points(depth_f, intr, stride=3), np.array([0.0, 0.0, 1.0]))
        assert plane is not None
        ys, xs = np.nonzero(np.any(color != 40, axis=2))
        z = depth_f[ys, xs]
        good = np.isfinite(z)
        points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
        height = plane.signed_distance(points)
        lower = np.zeros(len(xs), dtype=bool)
        lower[np.flatnonzero(good)] = height < 0.10
        color[ys[lower], xs[lower]] = side_bgr
        seam = np.zeros(len(xs), dtype=bool)
        seam[np.flatnonzero(good)] = np.abs(height - 0.10) < 0.012
        color[ys[seam], xs[seam]] = 40
        top = project_points(np.array([[center[0], center[1], center[2] - 0.2]]), intr)[0]
        u, v = int(round(float(top[0]))), int(round(float(top[1])))
        depth_f[v, u] += 0.05
        prepared = prepare_mask(
            color,
            depth_f,
            cfg["colors"],
            float(cfg["depth_min"]),
            float(cfg["depth_max"]),
            plane=plane,
            intr=intr,
            cube_size=float(cfg["cube_size"]),
        )
        assert prepared is not None
        assert prepared[0][v, u] > 0
        pose = detect_visible_cube(
            color,
            depth_f,
            intr,
            plane,
            cfg["colors"],
            cube_size=float(cfg["cube_size"]),
            min_pixels=200,
            min_footprint_fraction=0.35,
            size_tolerance=float(cfg["size_tolerance"]),
            prepared=prepared,
        )
        assert pose is not None
        assert np.linalg.norm(pose.bottom_center[:2] - center[:2]) < 0.04
        on = prepared[0][ys, xs] > 0
        bottom = np.zeros(len(xs), dtype=bool)
        bottom[np.flatnonzero(good)] = (height > 0.0) & (height < 0.03)
        assert int(on[bottom].sum()) > 20


def test_mask_stays_on_cube_when_background_is_raised():
    """旁边一块没有红蓝的凸起，和远处稀疏红点，都不能进绿线。足迹仍落在方块上。"""
    from geom.transform import project_points

    intr = intrinsics_for(320, 240)
    center = np.array([0.08, 0.06, 0.7])
    cfg = load_config("config/default.yaml")
    color, depth = render_cube(
        intr,
        center,
        0.2,
        size=float(cfg["cube_size"]),
        ground_z=float(center[2]),
        noise_m=0.0,
        top_bgr=(40, 40, 220),
        front_bgr=(220, 40, 40),
    )
    board = project_points(np.array([[0.55, 0.06, 0.62]]), intr)[0]
    bu, bv = int(round(float(board[0]))), int(round(float(board[1])))
    color[bv - 12 : bv + 12, bu - 16 : bu + 16] = (140, 140, 140)
    depth[bv - 12 : bv + 12, bu - 16 : bu + 16] = 0.55
    specks = project_points(
        np.array([[-0.45 + 0.04 * i, -0.25, 0.55] for i in range(6)]),
        intr,
    )
    for u, v in np.rint(specks).astype(int):
        color[v - 1 : v + 2, u - 1 : u + 2] = (40, 40, 220)
        depth[v - 1 : v + 2, u - 1 : u + 2] = 0.55
    depth_f = filter_depth(depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
    from ground.plane import sample_points

    plane = fit_ground(sample_points(depth_f, intr, stride=3), np.array([0.0, 0.0, 1.0]))
    assert plane is not None
    prepared = prepare_mask(
        color,
        depth_f,
        cfg["colors"],
        float(cfg["depth_min"]),
        float(cfg["depth_max"]),
        plane=plane,
        intr=intr,
        cube_size=float(cfg["cube_size"]),
    )
    assert prepared is not None
    assert int(prepared[0][bv - 8 : bv + 8, bu - 8 : bu + 8].sum()) == 0
    pose = detect_visible_cube(
        color,
        depth_f,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        min_pixels=200,
        min_footprint_fraction=0.35,
        size_tolerance=float(cfg["size_tolerance"]),
        prepared=prepared,
    )
    assert pose is not None
    assert np.linalg.norm(pose.bottom_center[:2] - center[:2]) < 0.04


def test_field_corner_recovers_extrinsic(tmp_path):
    """已知外参下，原点出发的长边 2 m 和短边 1.2 m 应解回同一组 R、t。
    写入 yaml 再读回，数值不变。"""
    rotation = compose_R(0.2, -0.35, 0.05)
    translation = np.array([0.3, -0.2, 1.4])
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    plane = Plane(normal=normal, offset=float(normal @ translation))
    origin = world_to_camera(np.zeros(3), rotation, translation)
    long_end = world_to_camera(np.array([2.0, 0.0, 0.0]), rotation, translation)
    short_end = world_to_camera(np.array([0.0, 1.2, 0.0]), rotation, translation)
    pose = estimate_from_lines(
        [Segment(origin, long_end), Segment(origin, short_end)],
        plane,
        2.0,
        1.2,
    )
    assert pose is not None
    est_r, est_t = pose
    np.testing.assert_allclose(est_t, translation, atol=1e-6)
    np.testing.assert_allclose(est_r, rotation, atol=1e-6)
    path = tmp_path / "extrinsic.yaml"
    save_extrinsic(path, est_r, est_t)
    loaded_r, loaded_t = load_extrinsic(path)
    np.testing.assert_allclose(loaded_r, est_r)
    np.testing.assert_allclose(loaded_t, est_t)


def test_fuse_and_score_bands():
    """核对作业分档：4%、12%、25% 分别是 4 分、2 分、0 分。
    八帧正常位姿加一帧离群点，融合结果应落在正常值上；三次满分的平均通过基础要求。"""
    assert band_score(0.04) == 4
    assert band_score(0.12) == 2
    assert band_score(0.25) == 0
    samples = [{"x": 1.0, "y": 0.5, "z": 0.0, "yaw": 0.2} for _ in range(8)]
    samples.append({"x": 4.0, "y": 4.0, "z": 0.0, "yaw": 2.5})
    fused = fuse_poses(samples)
    assert abs(fused.x - 1.0) < 1e-6
    assert abs(fused.yaw - 0.2) < 1e-6
    trial = score_trial(np.array([1.0, 0.0]), 0.0, np.array([1.02, 0.0]), 0.01)
    assert trial["position_score"] == 4
    summary = average_scores([trial, trial, trial])
    assert summary["basic_passed"]
    # 真值 y = 0 落在 0.15 m 以内，百分比会过敏，评分分母仍用这个下限。
    assert trial["sensitive_percent"]
    near = [{"x": 1.0, "y": 0.2, "z": 0.0, "yaw": 0.78, "quarter": True} for _ in range(4)]
    near += [{"x": 1.0, "y": 0.2, "z": 0.0, "yaw": -0.76, "quarter": True} for _ in range(4)]
    folded = fuse_poses(near)
    assert abs(angle_delta(folded.yaw, 0.77, True)) < np.deg2rad(3.0)


def test_face_fallback_on_single_face():
    """斜视时只画方块的正面。可见面检测应拒绝；贴地模型补救的平面位置误差小于 8 cm，
    折叠后的 yaw 误差小于 12°。"""
    intr = intrinsics_for(320, 240)
    cfg = load_config("config/default.yaml")
    rotation = compose_R(0.15, -0.7, 0.0)
    translation = np.array([0.12, 0.05, 0.7])
    center = np.array([0.22, 0.12, 0.0])
    yaw = 0.25
    color, depth = _render_front_face(intr, rotation, translation, center, yaw)
    gravity = rotation @ np.array([0.0, 0.0, -1.0])
    depth_f = filter_depth(depth, 0.3, 3.0, 0.08)
    frame = Frame(color, depth_f, gravity)
    basic = estimate_frame(frame, intr, cfg, rotation, translation)
    assert basic is None
    from ground.plane import sample_points
    plane = fit_ground(sample_points(depth_f, intr, stride=2), gravity)
    assert plane is not None
    pose = fit_when_face_missing(
        color,
        depth_f,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        rotation=rotation,
        translation=translation,
    )
    assert pose is not None
    from geom.transform import camera_to_world

    recovered = camera_to_world(pose.bottom_center, rotation, translation)
    assert np.linalg.norm(recovered[:2] - center[:2]) < 0.08
    assert abs(fold_yaw_quarter(pose.yaw - yaw)) < np.deg2rad(12)


def _render_front_face(intr, rotation, translation, center, yaw):
    """给补救测试用：世界系地面加上方块朝 +X 的那一个竖直面，不画顶面和其他侧面。"""
    from sim.scene import _face, _paint

    image = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan)
    xs = np.linspace(center[0] - 1.0, center[0] + 1.0, 60)
    ys = np.linspace(center[1] - 1.0, center[1] + 1.0, 60)
    gx, gy = np.meshgrid(xs, ys)
    ground_w = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    ground = world_to_camera(ground_w, rotation, translation)
    keep = ground[:, 2] > 0.2
    _paint(ground[keep], np.full((int(keep.sum()), 3), 40, np.uint8), intr, depth, image)
    size = 0.2
    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = rotation @ np.array([c, s, 0.0]) * size
    axis_y = rotation @ np.array([-s, c, 0.0]) * size
    axis_z = rotation @ np.array([0.0, 0.0, size])
    bottom = world_to_camera(center, rotation, translation)
    front = bottom + axis_x * 0.5
    pts, cols = _face(front, axis_y, axis_z, (255, 80, 40), samples=90)
    ok = pts[:, 2] > 0.2
    _paint(pts[ok], cols[ok], intr, depth, image)
    return image, depth
