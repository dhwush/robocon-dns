# 坐标链：印刷体旋转复合、世界/相机往返、两种畸变模型的投影往返、配置内参不被改写。

import numpy as np

from geom.transform import (
    Intrinsics,
    Rx,
    Ry,
    Rz,
    camera_to_world,
    compose_R,
    deproject_pixels,
    load_intrinsics,
    load_intrinsics_for_serial,
    project_linear,
    project_points,
    world_to_camera,
)


def test_printed_composition_matches_expansion():
    """R = Rz(φ) Ry(θ) Rx(ψ) 必须等于课件展开式，也等于三个单轴矩阵按该顺序相乘。"""
    phi, theta, psi = 0.4, -0.25, 0.7
    got = compose_R(phi, theta, psi)
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    expected = np.array(
        [
            [cph * cth, cph * sth * sps - sph * cps, cph * sth * cps + sph * sps],
            [sph * cth, sph * sth * sps + cph * cps, sph * sth * cps - cph * sps],
            [-sth, cth * sps, cth * cps],
        ]
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_allclose(got, Rz(phi) @ Ry(theta) @ Rx(psi), atol=1e-12)


def test_world_camera_roundtrip():
    """一组世界点变到相机系再变回来，应回到原点，误差只在浮点噪声内。"""
    rotation = compose_R(0.2, -0.3, 0.5)
    translation = np.array([0.1, -0.2, 0.4])
    points = np.array([[1.0, 0.2, 0.0], [0.0, 1.5, 0.3], [-0.4, 0.3, 1.2]])
    back = camera_to_world(world_to_camera(points, rotation, translation), rotation, translation)
    np.testing.assert_allclose(back, points, atol=1e-12)


def _sample_intrinsics(model: str, distorted: bool) -> Intrinsics:
    """构造一组固定内参。distorted 为真时带上非零的 k1、k2、p1、p2、k3。"""
    d = [0.08, -0.03, 0.001, -0.002, 0.01] if distorted else [0.0, 0.0, 0.0, 0.0, 0.0]
    return Intrinsics(640, 480, 600.0, 610.0, 320.0, 240.0, *d, distortion_model=model)


def test_projection_roundtrip_both_models():
    """Brown 和 Inverse Brown Conrady、有无畸变四种组合：投影到像素再用原深度反投影，
    相机系点误差小于 0.1 mm。"""
    points = np.array(
        [
            [0.10, -0.05, 1.2],
            [-0.20, 0.15, 1.8],
            [0.30, 0.20, 2.4],
            [0.0, 0.0, 1.0],
        ]
    )
    for model in ("brown", "inverse_brown_conrady"):
        for distorted in (False, True):
            intr = _sample_intrinsics(model, distorted)
            pixels = project_points(points, intr)
            depth = points[:, 2]
            back = deproject_pixels(pixels, depth, intr)
            np.testing.assert_allclose(back, points, atol=1e-4)


def test_linear_matrix_matches_pinhole_when_no_distortion():
    """无畸变时，课件的 M1 M2 矩阵投影应和先变到相机系再做小孔投影的结果一致。"""
    intr = _sample_intrinsics("none", False)
    rotation = compose_R(0.15, 0.05, -0.2)
    translation = np.array([0.0, 0.1, 0.0])
    world = np.array([[0.4, -0.2, 1.5], [-0.3, 0.25, 2.0]])
    cam = world_to_camera(world, rotation, translation)
    by_matrix = project_linear(world, intr, rotation, translation)
    by_pinhole = project_points(cam, intr)
    np.testing.assert_allclose(by_matrix, by_pinhole, atol=1e-9)


def test_intrinsics_follow_camera_serial():
    """四台相机各用自己的彩色内参。没接相机时按序列号查表，表外的号码不借用另一台。"""
    current = load_intrinsics_for_serial("244622071936")
    assert current.width == 1280
    assert abs(current.fx - 909.7513) < 1e-9
    assert abs(current.cy - 376.0774) < 1e-9
    other = load_intrinsics_for_serial("043322075459")
    assert abs(other.fx - 912.2286) < 1e-9
    assert abs(other.cx - 636.9600) < 1e-9
    assert abs(load_intrinsics_for_serial("213622078104").cy - 373.4455) < 1e-9
    assert abs(load_intrinsics_for_serial("244222071048").cx - 630.7516) < 1e-9
    try:
        load_intrinsics_for_serial("000000000000")
    except KeyError:
        return
    raise AssertionError("unknown serial should not fall back to another camera")


def test_config_intrinsics_are_the_only_source():
    """配置文件里的模型、分辨率和 fx 原样读入。用这组内参做一次投影再反投影，点能回来。"""
    intr = load_intrinsics("config/intrinsics.yaml")
    assert intr.distortion_model == "inverse_brown_conrady"
    assert intr.width == 1280
    assert abs(intr.fx - 909.7513) < 1e-9
    point = np.array([0.2, -0.1, 1.5])
    pixel = project_points(point, intr)[0]
    back = deproject_pixels(pixel, np.array([point[2]]), intr)[0]
    np.testing.assert_allclose(back, point, atol=1e-6)
