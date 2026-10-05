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
    project_linear,
    project_points,
    world_to_camera,
)


def test_printed_composition_matches_expansion():
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
    rotation = compose_R(0.2, -0.3, 0.5)
    translation = np.array([0.1, -0.2, 0.4])
    points = np.array([[1.0, 0.2, 0.0], [0.0, 1.5, 0.3], [-0.4, 0.3, 1.2]])
    back = camera_to_world(world_to_camera(points, rotation, translation), rotation, translation)
    np.testing.assert_allclose(back, points, atol=1e-12)


def _sample_intrinsics(model: str, distorted: bool) -> Intrinsics:
    d = [0.08, -0.03, 0.001, -0.002, 0.01] if distorted else [0.0, 0.0, 0.0, 0.0, 0.0]
    return Intrinsics(640, 480, 600.0, 610.0, 320.0, 240.0, *d, distortion_model=model)


def test_projection_roundtrip_both_models():
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
    intr = _sample_intrinsics("none", False)
    rotation = compose_R(0.15, 0.05, -0.2)
    translation = np.array([0.0, 0.1, 0.0])
    world = np.array([[0.4, -0.2, 1.5], [-0.3, 0.25, 2.0]])
    cam = world_to_camera(world, rotation, translation)
    by_matrix = project_linear(world, intr, rotation, translation)
    by_pinhole = project_points(cam, intr)
    np.testing.assert_allclose(by_matrix, by_pinhole, atol=1e-9)


def test_config_intrinsics_are_the_only_source():
    intr = load_intrinsics("config/intrinsics.yaml")
    assert intr.distortion_model == "inverse_brown_conrady"
    assert intr.width == 1280
    assert abs(intr.fx - 674.4192801798159) < 1e-9
    point = np.array([0.2, -0.1, 1.5])
    pixel = project_points(point, intr)[0]
    back = deproject_pixels(pixel, np.array([point[2]]), intr)[0]
    np.testing.assert_allclose(back, point, atol=1e-6)
