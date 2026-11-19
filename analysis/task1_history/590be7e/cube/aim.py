"""方块只露出一部分，或已经在画面里但这一帧没有识别出来时，给出俯仰和原地转角。

俯仰绕相机右轴：down 是镜头下俯，up 是上仰。原地转角绕竖直轴，小车不平移：
right 是向右转，left 是向左转。两个角度都是还要再转的大小，单位弧度，方向另给。

已经识别出来时，只有红蓝掩膜贴到画面边才给出 pitch、yaw。完整留在画面里不再建议转动。
连续几帧没有识别出来之后，才用 seek_pitch、seek_yaw 做微调。转角每拍最多 2.5°。
俯仰低于大约 5° 机构执行不了，所以只要还要俯仰，这一拍就是 5°，不再给出更小的一步。
红蓝块不在光轴上时，方向仍是把它收到光轴上。
已经对准、没有红蓝，或连续几帧都是地面失败时，只微调俯仰，seek_yaw 保持 none。
地面还在、绿线连续停在一处、足迹却失败时，不走 seek_。贴边用 footprint_pitch、footprint_yaw
把绿线往光轴收；绿线在中间时两个轴都微调，转角没有左右偏差就按给定方向扫。
镜头已经明显朝下时不再继续下俯。累计转过大约 15° 仍认不出就停这一轴。
中间扫转角扫到这个上限，或绿线贴到左右边时，下一拍改为反方向。
被切掉的一侧按方块对角线补回去，再让补全后的中心落到光轴上。
俯仰能直接收起图像里的上下偏差。左右偏差要先绕竖直轴转到俯仰平面里，再俯仰对准。
正俯视时，画面右侧的方块会先给出接近 90° 的右转，这是绕竖直轴能把它送进俯仰平面的转角。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from cube.detect import _color_layers
from geom.transform import Intrinsics, deproject_pixels


_DEADBAND_RAD = np.deg2rad(0.6)
# 顶面正方形对角线。补到这个跨度，四个上角才有机会一起进画面。
_DIAGONAL = float(np.sqrt(2.0))
# 未识别时原地转角每拍最多这么多。再大容易把已经差不多能看见的方块转出画面。
_SEEK_YAW_STEP_RAD = np.deg2rad(2.5)
# 俯仰机构低于大约 5° 不动。只要这一拍还要俯仰，就给整整 5°。
_SEEK_PITCH_STEP_RAD = np.deg2rad(5.0)
# 光轴已经比这个更朝下时，地面失败不再继续下俯。其他未识别改为上仰一拍。
_SEEK_NADIR_RAD = np.deg2rad(50.0)
# 绿线连续这么多帧都落在这个范围里，才算停在一处。
_STABLE_PX = 20.0
_STABLE_AREA_FRAC = 0.30
# 已经进入足迹恢复后，相机自己转一拍，绿线可以挪这么多，仍算同一处。
_HOLD_JUMP_PX = 120.0


@dataclass(frozen=True)
class AxisMove:
    """一个轴上的调整。direction 为 none 时角度为 0。"""

    direction: str
    radians: float
    degrees: float
    signed_radians: float

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "radians": self.radians,
            "degrees": self.degrees,
            "signed_radians": self.signed_radians,
        }


def _none_move() -> AxisMove:
    return AxisMove("none", 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ViewAdjustment:
    """当前帧建议的俯仰和原地转角。ready 为真时两个轴都不用再转。

    pitch、yaw 是已经识别出来、只是被画面边切到时的正常调整。
    seek_pitch、seek_yaw 只在这一帧没有识别出来时有非零值。
    footprint_pitch、footprint_yaw 只在绿线稳定、地面足迹仍失败时出现。
    """

    visible: bool
    ready: bool
    reason: str
    pitch: AxisMove
    yaw: AxisMove
    clipped: dict
    gravity_assumed: bool
    seek_pitch: AxisMove = field(default_factory=_none_move)
    seek_yaw: AxisMove = field(default_factory=_none_move)
    footprint_pitch: AxisMove = field(default_factory=_none_move)
    footprint_yaw: AxisMove = field(default_factory=_none_move)
    sweeping: bool = False

    def as_dict(self) -> dict:
        body = {
            "visible": self.visible,
            "ready": self.ready,
            "reason": self.reason,
            "pitch": self.pitch.as_dict(),
            "yaw": self.yaw.as_dict(),
            "clipped": dict(self.clipped),
            "gravity_assumed": self.gravity_assumed,
        }
        if self.reason in ("unrecognized", "ground"):
            body["seek_pitch"] = self.seek_pitch.as_dict()
            body["seek_yaw"] = self.seek_yaw.as_dict()
        if self.reason in ("footprint_edge", "footprint_center"):
            body["footprint_pitch"] = self.footprint_pitch.as_dict()
            body["footprint_yaw"] = self.footprint_yaw.as_dict()
            body["sweeping"] = self.sweeping
        return body


def _move(kind: str, signed: float) -> AxisMove:
    # 小于死区就当成已经对准，避免在 0 附近来回给相反方向。
    if abs(signed) < _DEADBAND_RAD:
        return _none_move()
    if kind == "pitch":
        direction = "down" if signed > 0.0 else "up"
    else:
        direction = "right" if signed > 0.0 else "left"
    magnitude = float(abs(signed))
    return AxisMove(direction, magnitude, float(np.degrees(magnitude)), float(signed))


def _level_axes(gravity: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    # 重力在彩色相机系里指向下。没有重力时按光轴水平：图像向下就是重力方向。
    assumed = gravity is None
    down = np.array([0.0, 1.0, 0.0]) if gravity is None else np.asarray(gravity, dtype=float)
    norm = float(np.linalg.norm(down))
    if norm < 1e-6:
        down = np.array([0.0, 1.0, 0.0])
        assumed = True
    else:
        down = down / norm
    up = -down
    optical_x = np.array([1.0, 0.0, 0.0])
    right = optical_x - float(np.dot(optical_x, up)) * up
    if float(np.linalg.norm(right)) < 0.25:
        # 相机右轴几乎竖直时，改用光轴和向上方向定水平右方。
        right = np.cross(np.array([0.0, 0.0, 1.0]), up)
    right = right / np.linalg.norm(right)
    forward = np.cross(up, right)
    forward = forward / np.linalg.norm(forward)
    return right, forward, up, assumed


def _largest_mask(strict: np.ndarray, min_pixels: int) -> np.ndarray | None:
    # 红蓝分界会把同一块方块分成两团。只在瞄准这一步把缝合上，不改位姿用的掩膜。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = None
    best_area = min_pixels - 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best = label
            best_area = area
    if best is None:
        return None
    return np.where(labels == best, np.uint8(255), np.uint8(0))


def _clipped_sides(mask: np.ndarray, margin: int) -> dict:
    height, width = mask.shape
    band = max(int(margin), 1)
    on = mask > 0
    return {
        "left": bool(np.any(on[:, :band])),
        "right": bool(np.any(on[:, width - band :])),
        "top": bool(np.any(on[:band, :])),
        "bottom": bool(np.any(on[height - band :, :])),
    }


def _extend_side(start: float, end: float, clipped_low: bool, clipped_high: bool, expected: float, limit: float) -> tuple[float, float]:
    # 两侧都被切住时，方块比画面宽，往两边补会把中心留在原地。
    if clipped_low == clipped_high:
        return start, end
    span = end - start
    missing = max(0.0, expected - span)
    if missing == 0.0:
        # 可见部分已经不小于整块方块，边上仍有像素，说明外面还有一截。只往外推一小段。
        missing = 0.15 * expected
    missing = min(missing, limit)
    if clipped_high:
        return start, end + missing
    return start - missing, end


def _aim_pixel(mask: np.ndarray, depth: np.ndarray, intr: Intrinsics, cube_size: float, zmin: float, zmax: float, margin: int) -> tuple[float, float, dict, float | None]:
    ys, xs = np.nonzero(mask)
    clipped = _clipped_sides(mask, margin)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    distance = float(np.median(z[valid])) if int(valid.sum()) >= 10 else None
    u0 = float(xs.min())
    u1 = float(xs.max())
    v0 = float(ys.min())
    v1 = float(ys.max())
    if distance is None:
        expect_u = 2.0 * max(u1 - u0, 1.0)
        expect_v = 2.0 * max(v1 - v0, 1.0)
    else:
        expect_u = _DIAGONAL * cube_size * float(intr.fx) / distance
        expect_v = _DIAGONAL * cube_size * float(intr.fy) / distance
    u0, u1 = _extend_side(u0, u1, clipped["left"], clipped["right"], expect_u, float(intr.width))
    v0, v1 = _extend_side(v0, v1, clipped["top"], clipped["bottom"], expect_v, float(intr.height))
    return (u0 + u1) * 0.5, (v0 + v1) * 0.5, clipped, distance


def _angles_toward(point: np.ndarray, gravity: np.ndarray | None) -> tuple[float, float, bool]:
    # 正值：俯仰是下俯，原地转角是向右。点在水平面上的方位决定转角，相对水平面的俯角决定俯仰。
    right, forward, up, assumed = _level_axes(gravity)
    x_l = float(np.dot(point, right))
    y_l = float(np.dot(point, forward))
    z_l = float(np.dot(point, up))
    yaw = float(np.arctan2(x_l, y_l))
    down = -up
    current_down = float(np.arcsin(np.clip(float(down[2]), -1.0, 1.0)))
    desired_down = float(np.arctan2(-z_l, np.hypot(x_l, y_l)))
    return desired_down - current_down, yaw, assumed


def _empty(reason: str, assumed: bool) -> ViewAdjustment:
    blank = {"left": False, "right": False, "top": False, "bottom": False}
    return ViewAdjustment(False, True, reason, _none_move(), _none_move(), blank, assumed)


def _down_angle(gravity: np.ndarray | None) -> float:
    # 光轴相对水平面的下俯角。正值表示镜头朝下。
    down = np.array([0.0, 1.0, 0.0]) if gravity is None else np.asarray(gravity, dtype=float)
    norm = float(np.linalg.norm(down))
    if norm < 1e-6:
        down = np.array([0.0, 1.0, 0.0])
    else:
        down = down / norm
    return float(np.arcsin(np.clip(float(down[2]), -1.0, 1.0)))


def _seek_pitch_when_centered(gravity: np.ndarray | None, *, stop_when_low: bool) -> float | None:
    # 正值是下俯。已经低过头时，地面失败就停；其他情况改为上仰一拍。
    if _down_angle(gravity) > _SEEK_NADIR_RAD:
        if stop_when_low:
            return None
        return -_SEEK_PITCH_STEP_RAD
    return _SEEK_PITCH_STEP_RAD


def _pitch_step(signed: float, room: float | None) -> AxisMove:
    # 剩余额度不够 5° 时这一拍不俯仰。给出 3° 机构也不会动。
    if abs(signed) < _DEADBAND_RAD:
        return _none_move()
    if room is not None and float(room) + 1e-6 < _SEEK_PITCH_STEP_RAD:
        return _none_move()
    return _move("pitch", float(np.copysign(_SEEK_PITCH_STEP_RAD, signed)))


def _micro(kind: str, signed: float, room: float | None) -> AxisMove:
    # 俯仰固定 5°。转角每拍重新按当前画面计算，再限制到 2.5°。room 是这一轴还允许累计的弧度。
    if kind == "pitch":
        return _pitch_step(signed, room)
    if room is not None and room <= _DEADBAND_RAD:
        return _none_move()
    limit = _SEEK_YAW_STEP_RAD if room is None else min(_SEEK_YAW_STEP_RAD, float(room))
    step = float(np.clip(signed, -limit, limit))
    return _move(kind, step)


def _seek_view(
    reason: str,
    visible: bool,
    clipped: dict,
    assumed: bool,
    seek_pitch: AxisMove,
    seek_yaw: AxisMove,
) -> ViewAdjustment:
    # 正常的 pitch、yaw 保持 none。要执行的角度只放在 seek_ 前缀下。
    moving = seek_pitch.direction != "none" or seek_yaw.direction != "none"
    return ViewAdjustment(
        visible,
        not moving,
        reason,
        _none_move(),
        _none_move(),
        clipped,
        assumed,
        seek_pitch,
        seek_yaw,
    )


def color_blob(color_bgr: np.ndarray, colors: list[dict], min_pixels: int = 40) -> tuple[float, float, int] | None:
    """红蓝块的重心和像素数。没有成片的块时返回空。瞄准和“是否停在一处”用同一块。"""
    strict, _loose, _front = _color_layers(color_bgr, colors)
    mask = _largest_mask(strict, min_pixels)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), int(xs.size)


def contour_is_stable(
    samples: list,
    need: int = 6,
    tol_px: float = _STABLE_PX,
    area_frac: float = _STABLE_AREA_FRAC,
) -> bool:
    """最近 need 帧都有绿线，重心离中位数不超过 tol_px，面积相对中位数不超过 area_frac。"""
    if len(samples) < int(need):
        return False
    window = list(samples)[-int(need) :]
    if any(item is None for item in window):
        return False
    us = np.array([item[0] for item in window], dtype=float)
    vs = np.array([item[1] for item in window], dtype=float)
    areas = np.array([item[2] for item in window], dtype=float)
    if float(np.max(np.hypot(us - np.median(us), vs - np.median(vs)))) > float(tol_px):
        return False
    median_area = float(np.median(areas))
    if median_area <= 0.0:
        return False
    return bool(np.max(np.abs(areas - median_area) / median_area) <= float(area_frac))


def contour_jumped(previous, current, tol_px: float = _HOLD_JUMP_PX) -> bool:
    """相邻两帧重心差过 tol_px，或其中一帧没有绿线。足迹恢复被这一下打断。"""
    if previous is None or current is None:
        return True
    return float(np.hypot(float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1]))) > float(tol_px)


def update_footprint_yaw_sign(
    sign: float,
    clipped: dict | None,
    yaw_sum: float,
    limit: float,
    previous_reason: str | None = None,
    sweeping: bool = False,
) -> tuple[float, float]:
    """中间扫转角用的方向。贴左边之后改为向右，贴右边之后改为向左。

    只在上一拍是光轴上的 footprint_center 扫掠、并且这一轴累计到 limit 时反向并把累计清零。
    绿线还偏在一侧时不反向，累计到 limit 就停。返回新的方向和累计。正方向是向右。
    """
    sign = 1.0 if float(sign) >= 0.0 else -1.0
    on_side = False
    if clipped:
        if clipped.get("left") and not clipped.get("right"):
            sign = 1.0
            on_side = True
        elif clipped.get("right") and not clipped.get("left"):
            sign = -1.0
            on_side = True
    at_limit = abs(float(yaw_sum)) >= float(limit) - 1e-6
    if on_side and at_limit:
        # 扫到头又贴上左右边时，清掉累计，下一拍才能顺着离开这边的方向转回来。
        yaw_sum = 0.0
    elif sweeping and previous_reason == "footprint_center" and at_limit:
        sign = -sign
        yaw_sum = 0.0
    return sign, float(yaw_sum)


def _footprint_view(
    reason: str,
    clipped: dict,
    assumed: bool,
    footprint_pitch: AxisMove,
    footprint_yaw: AxisMove,
    sweeping: bool = False,
) -> ViewAdjustment:
    # 正常的 pitch、yaw 和 seek_ 都不出现。足迹恢复只用 footprint_ 前缀。
    moving = footprint_pitch.direction != "none" or footprint_yaw.direction != "none"
    return ViewAdjustment(
        True,
        not moving,
        reason,
        _none_move(),
        _none_move(),
        clipped,
        assumed,
        footprint_pitch=footprint_pitch,
        footprint_yaw=footprint_yaw,
        sweeping=sweeping,
    )


def _plan_footprint(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    cube_size: float,
    gravity: np.ndarray | None,
    zmin: float,
    zmax: float,
    edge_margin: int,
    pitch_room: float | None,
    yaw_room: float | None,
    yaw_bias: float,
) -> ViewAdjustment:
    # 贴边时方向跟已经识别出来的切边一样，把补全后的中心收到光轴上。幅度仍是一拍。
    # 没贴边时俯仰只换入射角，转角有左右偏差就跟着走，没有就按 yaw_bias 扫一拍。
    u, v, clipped, _distance = _aim_pixel(mask, depth, intr, cube_size, zmin, zmax, edge_margin)
    point = deproject_pixels(np.array([[u, v]]), np.array([1.0]), intr)[0]
    pitch_signed, yaw_signed, assumed = _angles_toward(point, gravity)
    if any(clipped.values()):
        return _footprint_view(
            "footprint_edge",
            clipped,
            assumed,
            _pitch_step(pitch_signed, pitch_room),
            _micro("yaw", yaw_signed, yaw_room),
        )
    step = _seek_pitch_when_centered(gravity, stop_when_low=False)
    pitch = _none_move() if step is None else _pitch_step(step, pitch_room)
    # 正俯视时，光轴附近的一点在水平面上方位不定，不能拿那个转角去扫。
    off_axis = float(np.arctan2(np.hypot(point[0], point[1]), point[2]))
    if off_axis >= _DEADBAND_RAD and abs(yaw_signed) >= _DEADBAND_RAD:
        yaw = _micro("yaw", yaw_signed, yaw_room)
        sweeping = False
    else:
        bias = 1.0 if float(yaw_bias) >= 0.0 else -1.0
        yaw = _micro("yaw", bias * _SEEK_YAW_STEP_RAD, yaw_room)
        sweeping = True
    return _footprint_view("footprint_center", clipped, assumed, pitch, yaw, sweeping)


def plan_view(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    colors: list[dict],
    cube_size: float = 0.2,
    gravity: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 3.0,
    min_pixels: int = 40,
    edge_margin: int = 4,
    recognized: bool = True,
    seek: bool | None = None,
    ground_lost: bool = False,
    footprint_lost: bool = False,
    pitch_room: float | None = None,
    yaw_room: float | None = None,
    yaw_bias: float = 1.0,
) -> ViewAdjustment:
    """根据当前画面给出俯仰和原地转角。

    recognized 为真、seek 为假时，没有红蓝块，或方块没有被画面边切到，都不建议转动。
    seek 为真时，调用方已经数过连续失败帧。转角每拍最多 2.5°，俯仰只要还要转就是 5°。
    连续失败还没到门限时，recognized 和 seek 都为假，这一帧不转。
    ground_lost 为真时不跟着颜色块转，只尝试下俯，让画面里多露出地面。
    footprint_lost 为真时地面还在，绿线也连续停在一处，只是足迹失败。
    贴边给出 footprint_edge，留在中间给出 footprint_center。这两个不用 seek_。
    """
    if seek is None:
        seek = not recognized
    strict, _loose, _front = _color_layers(color_bgr, colors)
    mask = _largest_mask(strict, min_pixels)
    _right, _forward, _up, assumed = _level_axes(gravity)
    blank = {"left": False, "right": False, "top": False, "bottom": False}
    if footprint_lost and mask is not None and not ground_lost:
        return _plan_footprint(
            mask,
            depth,
            intr,
            cube_size,
            gravity,
            zmin,
            zmax,
            edge_margin,
            pitch_room,
            yaw_room,
            yaw_bias,
        )
    if ground_lost:
        step = _seek_pitch_when_centered(gravity, stop_when_low=True)
        pitch = _none_move() if step is None else _micro("pitch", step, pitch_room)
        return _seek_view("ground", mask is not None, blank, assumed, pitch, _none_move())
    if not recognized and not seek:
        if mask is None:
            return _empty("no_cube", assumed)
        return ViewAdjustment(True, True, "waiting", _none_move(), _none_move(), blank, assumed)
    if mask is None:
        if recognized:
            return _empty("no_cube", assumed)
        step = _seek_pitch_when_centered(gravity, stop_when_low=False)
        pitch = _none_move() if step is None else _micro("pitch", step, pitch_room)
        return _seek_view("unrecognized", False, blank, assumed, pitch, _none_move())
    u, v, clipped, _distance = _aim_pixel(mask, depth, intr, cube_size, zmin, zmax, edge_margin)
    if recognized and not any(clipped.values()):
        return ViewAdjustment(True, True, "centered", _none_move(), _none_move(), clipped, assumed)
    # 深度只用来估计方块在画面上该有多大。瞄准方向跟距离无关，用 1 m 取出这条射线。
    point = deproject_pixels(np.array([[u, v]]), np.array([1.0]), intr)[0]
    pitch_signed, yaw_signed, assumed = _angles_toward(point, gravity)
    if not recognized:
        # 正俯视时，光轴附近的一点在水平面上方位不定，不能拿那个转角去转车。
        off_axis = float(np.arctan2(np.hypot(point[0], point[1]), point[2]))
        if off_axis < _DEADBAND_RAD:
            step = _seek_pitch_when_centered(gravity, stop_when_low=False)
            pitch = _none_move() if step is None else _micro("pitch", step, pitch_room)
            return _seek_view("unrecognized", True, clipped, assumed, pitch, _none_move())
        return _seek_view(
            "unrecognized",
            True,
            clipped,
            assumed,
            _micro("pitch", pitch_signed, pitch_room),
            _micro("yaw", yaw_signed, yaw_room),
        )
    pitch = _move("pitch", pitch_signed)
    yaw = _move("yaw", yaw_signed)
    if pitch.direction == "none" and yaw.direction == "none":
        ready = True
        reason = "fills_frame"
    else:
        ready = False
        reason = "partial"
    return ViewAdjustment(True, ready, reason, pitch, yaw, clipped, assumed)


def _load_cfg(path: str) -> dict:
    from pathlib import Path

    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    hsv_path = Path(cfg.get("hsv_path", "config/hsv.yaml"))
    if hsv_path.exists():
        with open(hsv_path, "r", encoding="utf-8") as handle:
            saved = yaml.safe_load(handle) or {}
        if saved.get("colors"):
            cfg["colors"] = saved["colors"]
    return cfg


def run_live(cfg: dict, intr: Intrinsics, serial: str | None, color_lock: dict | None) -> None:
    """实机每帧打印一行调整量。识别过程中只转不平移，这里不输出平移。"""
    import time

    from camera.source import Frame, RealSenseSource, filter_depth

    source = RealSenseSource(
        intr.width,
        intr.height,
        fps=int(cfg.get("fps", 30)),
        serial=serial,
        color_lock=color_lock,
    )
    period = 1.0 / float(cfg.get("output_hz", 10))
    last = 0.0
    try:
        while True:
            frame: Frame = source.read()
            depth = filter_depth(frame.depth, cfg["depth_min"], cfg["depth_max"], cfg["flying_pixel_m"])
            adjustment = plan_view(
                frame.color,
                depth,
                intr,
                cfg["colors"],
                cube_size=float(cfg["cube_size"]),
                gravity=frame.gravity,
                zmin=float(cfg["depth_min"]),
                zmax=float(cfg["depth_max"]),
            )
            now = time.monotonic()
            if now - last >= period:
                print(json.dumps(adjustment.as_dict(), ensure_ascii=False))
                last = now
    finally:
        source.close()


def main() -> None:
    import argparse

    from camera.source import color_lock_from_config, list_camera_serials
    from geom.transform import load_intrinsics, load_intrinsics_for_serial

    parser = argparse.ArgumentParser(description="方块只露出一部分时，输出俯仰和原地转角")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--light", default=None)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    if args.intrinsics:
        serial, intr = None, load_intrinsics(args.intrinsics)
    else:
        serial = args.serial
        if serial is None:
            found = list_camera_serials()
            if not found:
                raise SystemExit("没有接上相机，不能计算俯仰和转角")
            if len(found) > 1:
                raise SystemExit("接了多台相机，请用 --serial 指定: " + ", ".join(found))
            serial = found[0]
        try:
            intr = load_intrinsics_for_serial(str(serial))
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    run_live(cfg, intr, serial, color_lock_from_config(cfg, args.light))


if __name__ == "__main__":
    main()
