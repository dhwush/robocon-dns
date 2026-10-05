"""Multi-frame median and the assignment's relative-error bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geom.transform import wrap_angle


@dataclass
class FusedPose:
    x: float
    y: float
    z: float
    yaw: float


def _circular_median(angles: np.ndarray) -> float:
    values = np.asarray(angles, dtype=float)
    best = values[0]
    best_cost = np.inf
    for candidate in values:
        delta = np.arctan2(np.sin(values - candidate), np.cos(values - candidate))
        cost = float(np.median(np.abs(delta)))
        if cost < best_cost:
            best_cost = cost
            best = candidate
    return float(best)


def fuse_poses(samples: list[dict], mad_scale: float = 3.0) -> FusedPose | None:
    if not samples:
        return None
    xyz = np.array([[s["x"], s["y"], s["z"]] for s in samples], dtype=float)
    yaw = np.array([s["yaw"] for s in samples], dtype=float)
    center = np.median(xyz, axis=0)
    offset = np.linalg.norm(xyz - center, axis=1)
    med = float(np.median(offset))
    mad = float(np.median(np.abs(offset - med))) + 1e-6
    yaw_ref = _circular_median(yaw)
    yaw_delta = np.abs(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref)))
    yaw_med = float(np.median(yaw_delta))
    yaw_mad = float(np.median(np.abs(yaw_delta - yaw_med))) + 1e-6
    keep = (offset <= med + mad_scale * mad) & (yaw_delta <= yaw_med + mad_scale * yaw_mad)
    if int(keep.sum()) < 1:
        keep = np.ones(len(samples), dtype=bool)
    xyz = xyz[keep]
    yaw = yaw[keep]
    fused_xyz = np.median(xyz, axis=0)
    return FusedPose(x=float(fused_xyz[0]), y=float(fused_xyz[1]), z=float(fused_xyz[2]), yaw=_circular_median(yaw))


def relative_position_error(est_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    gt = np.asarray(gt_xy, dtype=float)
    denom = float(np.linalg.norm(gt))
    if denom < 1e-6:
        denom = 1e-6
    return float(np.linalg.norm(np.asarray(est_xy, dtype=float) - gt) / denom)


def relative_yaw_error(est: float, gt: float) -> float:
    delta = abs(wrap_angle(est - gt))
    return float(delta / np.pi)


def band_score(error: float) -> int:
    if error < 0.05:
        return 4
    if error < 0.10:
        return 3
    if error < 0.15:
        return 2
    if error < 0.20:
        return 1
    return 0


def score_trial(est_xy: np.ndarray, est_yaw: float, gt_xy: np.ndarray, gt_yaw: float) -> dict:
    pos = relative_position_error(est_xy, gt_xy)
    yaw = relative_yaw_error(est_yaw, gt_yaw)
    return {"position_error": pos, "yaw_error": yaw, "position_score": band_score(pos), "yaw_score": band_score(yaw)}


def average_scores(trials: list[dict]) -> dict:
    pos = float(np.mean([t["position_score"] for t in trials]))
    yaw = float(np.mean([t["yaw_score"] for t in trials]))
    return {"position": pos, "yaw": yaw, "total": pos + yaw, "basic_passed": pos >= 4.0 and yaw >= 4.0}
