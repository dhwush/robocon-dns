"""多帧融合，以及作业的相对误差分档。

位置用中值。无标记方块的 yaw 先按 90° 展开成连续角再取中值，输出前折回主值区间，
避免 ±45° 两侧被当成两个朝向。离群帧用绝对中位差丢掉。
x、y 分开计相对误差。真值绝对值小于 0.15 m 时分母仍用 0.15 m，只避免百分比被小坐标放大，
并在结果里标出 sensitive_percent；这个下限不代表测量可以放宽。
<5% 为 4 分，之后每 5% 降 1 分。x/y 的得分再取平均。基础要求两项都到 4 分才算通过。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geom.transform import fold_yaw_quarter, wrap_angle


@dataclass
class FusedPose:
    x: float
    y: float
    z: float
    yaw: float


# --- 多帧中值 -----------------------------------------------------------------

def _circular_median(angles: np.ndarray) -> float:
    # 普通中值会把 -179° 和 179° 算成差很远。这里在每个样本上量环形距离，取中位代价最小的那个。
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


def _align_period(angles: np.ndarray, period: float) -> np.ndarray:
    # 以环形中值为参考，把每个角加减若干个周期，使它们落在同一段连续角上。
    ref = _circular_median(angles)
    turns = np.round((angles - ref) / period)
    return angles - turns * period


def fuse_poses(samples: list[dict], mad_scale: float = 3.0) -> FusedPose | None:
    if not samples:
        return None
    xyz = np.array([[s["x"], s["y"], s["z"]] for s in samples], dtype=float)
    yaw = np.array([s["yaw"] for s in samples], dtype=float)
    quarter = any(bool(s.get("quarter")) for s in samples)
    # 位置离群：到中位点的距离，再用这组距离的绝对中位差做门槛。
    center = np.median(xyz, axis=0)
    offset = np.linalg.norm(xyz - center, axis=1)
    med = float(np.median(offset))
    mad = float(np.median(np.abs(offset - med))) + 1e-6
    if quarter:
        # 无标记方块每 90° 重合。先展开再比偏差，±45° 两侧才不会互相把对方当成离群。
        aligned = _align_period(yaw, np.pi / 2.0)
        yaw_delta = np.abs(aligned - float(np.median(aligned)))
    else:
        aligned = yaw
        yaw_ref = _circular_median(yaw)
        yaw_delta = np.abs(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref)))
    yaw_med = float(np.median(yaw_delta))
    yaw_mad = float(np.median(np.abs(yaw_delta - yaw_med))) + 1e-6
    keep = (offset <= med + mad_scale * mad) & (yaw_delta <= yaw_med + mad_scale * yaw_mad)
    if int(keep.sum()) < 1:
        keep = np.ones(len(samples), dtype=bool)
    xyz = xyz[keep]
    fused_xyz = np.median(xyz, axis=0)
    if quarter:
        fused_yaw = fold_yaw_quarter(float(np.median(aligned[keep])))
    else:
        fused_yaw = _circular_median(yaw[keep])
    return FusedPose(x=float(fused_xyz[0]), y=float(fused_xyz[1]), z=float(fused_xyz[2]), yaw=fused_yaw)


# --- 评分 ---------------------------------------------------------------------
# 基础要求要先达到 4 分，才进入提高要求。三次测量取得分的平均。

def _axis_error(est: float, gt: float, floor: float) -> float:
    # 真值接近 0 时，百分比会被深度噪声放大。0.15 m 只做分母下限，不放宽测量本身。
    return abs(float(est) - float(gt)) / max(abs(float(gt)), floor)


def angle_delta(est: float, gt: float, quarter: bool) -> float:
    # 正方形棱每 90° 重合。有朝向标记时不再折返，只把差折进 (-π, π]。
    if quarter:
        return min(abs(wrap_angle(est - gt - k * np.pi / 2.0)) for k in range(-2, 3))
    return abs(wrap_angle(est - gt))


def relative_yaw_error(est: float, gt: float, quarter: bool = True) -> float:
    # 无标记方块的分母是 90°。有朝向标记时分母是 180°。
    delta = angle_delta(est, gt, quarter)
    period = np.pi / 2.0 if quarter else np.pi
    return float(delta / period)


def band_score(error: float) -> int:
    # 作业分档：<5% 4 分，5–10% 3 分，10–15% 2 分，15–20% 1 分，其余 0 分。
    if error < 0.05:
        return 4
    if error < 0.10:
        return 3
    if error < 0.15:
        return 2
    if error < 0.20:
        return 1
    return 0


def score_trial(
    est_xy: np.ndarray,
    est_yaw: float,
    gt_xy: np.ndarray,
    gt_yaw: float,
    quarter: bool = True,
    axis_floor: float = 0.15,
) -> dict:
    est = np.asarray(est_xy, dtype=float)
    gt = np.asarray(gt_xy, dtype=float)
    error_x = _axis_error(est[0], gt[0], axis_floor)
    error_y = _axis_error(est[1], gt[1], axis_floor)
    yaw = relative_yaw_error(est_yaw, gt_yaw, quarter=quarter)
    score_x = band_score(error_x)
    score_y = band_score(error_y)
    abs_yaw = angle_delta(est_yaw, gt_yaw, quarter)
    # 0.15 m 只用于内部告警。真值更小时，5% 已经小于深度噪声，百分比会过敏。
    sensitive = abs(float(gt[0])) < axis_floor or abs(float(gt[1])) < axis_floor
    return {
        "position_error": max(error_x, error_y),
        "error_x": error_x,
        "error_y": error_y,
        "yaw_error": yaw,
        "position_score": 0.5 * (score_x + score_y),
        "yaw_score": band_score(yaw),
        "abs_x": abs(float(est[0] - gt[0])),
        "abs_y": abs(float(est[1] - gt[1])),
        "abs_yaw": float(abs_yaw),
        "sensitive_percent": sensitive,
    }


def average_scores(trials: list[dict]) -> dict:
    pos = float(np.mean([t["position_score"] for t in trials]))
    yaw = float(np.mean([t["yaw_score"] for t in trials]))
    return {"position": pos, "yaw": yaw, "total": pos + yaw, "basic_passed": pos >= 4.0 and yaw >= 4.0}
