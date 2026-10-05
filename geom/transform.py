"""World, camera, image, and pixel transforms from the course slides.

Printed right-handed matrices are used. The handwritten Rx on the board
flips the sign of theta relative to the printed Rx; this module follows
the printed form. Distortion uses the forward radial polynomial. The slide
text swaps the names of the distorted and corrected coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def compose_R(phi: float, theta: float, psi: float) -> np.ndarray:
    """R = Rz(phi) Ry(theta) Rx(psi), matching the printed expansion."""
    return Rz(phi) @ Ry(theta) @ Rx(psi)


def world_to_camera(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = (rotation @ pts.T).T + np.asarray(translation, dtype=float)
    return out[0] if single else out


def camera_to_world(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = (rotation.T @ (pts - np.asarray(translation, dtype=float)).T).T
    return out[0] if single else out


def extrinsic_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    m2 = np.eye(4)
    m2[:3, :3] = rotation
    m2[:3, 3] = np.asarray(translation, dtype=float)
    return m2


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """M1, 3x4, as on the combined-projection slide."""
    return np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def projection_matrix(fx: float, fy: float, cx: float, cy: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return intrinsics_matrix(fx, fy, cx, cy) @ extrinsic_matrix(rotation, translation)


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    distortion_model: str

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=float)


def load_intrinsics(path: str | Path) -> Intrinsics:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    k1, k2, p1, p2, k3 = raw["D"]
    return Intrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        k1=float(k1),
        k2=float(k2),
        p1=float(p1),
        p2=float(p2),
        k3=float(k3),
        distortion_model=str(raw["distortion_model"]),
    )


def _distort(x: np.ndarray, y: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Forward map: ideal normalized coordinates to distorted ones."""
    r2 = x * x + y * y
    radial = 1.0 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2
    x_t = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
    y_t = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
    return x * radial + x_t, y * radial + y_t


def _undistort_iter(xd: np.ndarray, yd: np.ndarray, intr: Intrinsics, steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Invert the forward polynomial so distort(x, y) ~= (xd, yd)."""
    x = np.array(xd, dtype=float, copy=True)
    y = np.array(yd, dtype=float, copy=True)
    for _ in range(steps):
        r2 = x * x + y * y
        radial = 1.0 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2
        x_t = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
        y_t = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
        x = (xd - x_t) / radial
        y = (yd - y_t) / radial
    return x, y


def project_points(points_cam: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Camera points to pixels. Returns Nx2."""
    pts = np.atleast_2d(np.asarray(points_cam, dtype=float))
    z = pts[:, 2]
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    model = intr.distortion_model
    if model == "brown":
        xd, yd = _distort(x, y, intr)
    elif model == "inverse_brown_conrady":
        xd, yd = _undistort_iter(x, y, intr)
    elif model == "none":
        xd, yd = x, y
    else:
        raise ValueError(f"unknown distortion model {model}")
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return np.column_stack([u, v])


def deproject_pixels(pixels: np.ndarray, depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Pixels plus depth (meters) to camera-frame points. Returns Nx3."""
    pix = np.atleast_2d(np.asarray(pixels, dtype=float))
    z = np.asarray(depth, dtype=float).reshape(-1)
    xd = (pix[:, 0] - intr.cx) / intr.fx
    yd = (pix[:, 1] - intr.cy) / intr.fy
    model = intr.distortion_model
    if model == "brown":
        x, y = _undistort_iter(xd, yd, intr)
    elif model == "inverse_brown_conrady":
        x, y = _distort(xd, yd, intr)
    elif model == "none":
        x, y = xd, yd
    else:
        raise ValueError(f"unknown distortion model {model}")
    return np.column_stack([x * z, y * z, z])


def project_linear(points_world: np.ndarray, intr: Intrinsics, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Distortion-free slide projection s[u,v,1]^T = M1 M2 Xw."""
    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    m = projection_matrix(intr.fx, intr.fy, intr.cx, intr.cy, rotation, translation)
    proj = (m @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def fold_yaw_quarter(angle: float) -> float:
    """Fold a square's yaw into (-pi/4, pi/4]."""
    half = np.pi / 4.0
    period = np.pi / 2.0
    return float((angle + half) % period - half)


def yaw_about_z(direction_world: np.ndarray) -> float:
    d = np.asarray(direction_world, dtype=float)
    return float(np.arctan2(d[1], d[0]))
