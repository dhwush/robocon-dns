# 方块只露出一部分时，俯仰和原地转角的方向、大小。

import numpy as np

from cube.aim import plan_view
from geom.transform import Intrinsics


COLORS = [
    {"name": "red", "hsv_low": [0, 80, 60], "hsv_high": [10, 255, 255], "front": False},
    {"name": "red_high", "hsv_low": [170, 80, 60], "hsv_high": [180, 255, 255], "front": False},
    {"name": "blue", "hsv_low": [100, 80, 60], "hsv_high": [130, 255, 255], "front": False},
]


def _intr() -> Intrinsics:
    return Intrinsics(320, 240, 300.0, 300.0, 160.0, 120.0, 0, 0, 0, 0, 0, "none")


def _scene(u0: int, u1: int, v0: int, v1: int, depth_m: float = 1.0):
    color = np.full((240, 320, 3), 40, dtype=np.uint8)
    depth = np.full((240, 320), np.nan, dtype=np.float32)
    color[v0:v1, u0:u1] = (0, 0, 255)
    depth[v0:v1, u0:u1] = depth_m
    return color, depth


def test_centered_cube_needs_no_motion():
    """方块在画面中间且没有被边切到时，俯仰和转角都是 none。"""
    color, depth = _scene(140, 180, 100, 140)
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=np.array([0.0, 1.0, 0.0]))
    assert adjustment.visible
    assert adjustment.ready
    assert adjustment.reason == "centered"
    assert adjustment.pitch.direction == "none"
    assert adjustment.yaw.direction == "none"
    assert adjustment.pitch.radians == 0.0
    assert adjustment.yaw.radians == 0.0


def test_right_edge_turns_right():
    """方块被右边界切掉。补上看不见的一截之后，原地转角向右，大小与瞄准点方位一致。"""
    color, depth = _scene(280, 320, 100, 140)
    gravity = np.array([0.0, 1.0, 0.0])
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=gravity)
    assert adjustment.reason == "partial"
    assert adjustment.clipped["right"]
    assert not adjustment.clipped["left"]
    assert adjustment.yaw.direction == "right"
    assert adjustment.pitch.direction == "none"
    # 可见宽度约 40 px，对角线约 85 px，瞄准点在右边界之外，方位约 0.45–0.55 rad。
    assert 0.35 < adjustment.yaw.radians < 0.7
    assert adjustment.yaw.signed_radians > 0.0
    assert adjustment.yaw.degrees == np.degrees(adjustment.yaw.radians)


def test_bottom_edge_pitches_down():
    """方块被下边界切掉。光轴水平时，下俯才能把底下那一截收进画面。"""
    color, depth = _scene(140, 180, 200, 240)
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=np.array([0.0, 1.0, 0.0]))
    assert adjustment.pitch.direction == "down"
    assert adjustment.yaw.direction == "none"
    assert adjustment.pitch.signed_radians > 0.0
    assert 0.25 < adjustment.pitch.radians < 0.6


def test_fully_visible_off_center_needs_no_motion():
    """已经识别出来、方块又完整留在画面里时，即使不在正中，也不给俯仰和转角。"""
    color, depth = _scene(40, 80, 40, 80)
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=np.array([0.0, 1.0, 0.0]))
    assert adjustment.reason == "centered"
    assert adjustment.ready
    assert adjustment.pitch.direction == "none"
    assert adjustment.yaw.direction == "none"
    assert "seek_pitch" not in adjustment.as_dict()
    assert not any(adjustment.clipped.values())


def test_unrecognized_in_frame_uses_seek_angles():
    """连续未识别、方块仍完整留在画面里时，seek_ 只给出 2.5° 的一步，方向朝向光轴。"""
    color, depth = _scene(40, 80, 40, 80)
    gravity = np.array([0.0, 1.0, 0.0])
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=gravity, recognized=False)
    body = adjustment.as_dict()
    assert adjustment.reason == "unrecognized"
    assert not adjustment.ready
    assert adjustment.visible
    assert not any(adjustment.clipped.values())
    assert body["pitch"]["direction"] == "none"
    assert body["yaw"]["direction"] == "none"
    # 块在左上方。整段大约左转 18°、上仰 11°，这一拍各只走 2.5°。
    assert body["seek_yaw"]["direction"] == "left"
    assert body["seek_pitch"]["direction"] == "up"
    assert abs(body["seek_yaw"]["degrees"] - 2.5) < 1e-6
    assert abs(body["seek_pitch"]["degrees"] - 2.5) < 1e-6
    assert body["seek_yaw"]["signed_radians"] < 0.0
    assert body["seek_pitch"]["signed_radians"] < 0.0


def test_unrecognized_on_axis_pitches_down():
    """方块已经在光轴上但仍未识别时，下俯 2.5°，换一个能看见顶面的角度。"""
    color, depth = _scene(140, 180, 100, 140)
    adjustment = plan_view(
        color, depth, _intr(), COLORS, gravity=np.array([0.0, 1.0, 0.0]), recognized=False
    )
    body = adjustment.as_dict()
    assert body["seek_pitch"]["direction"] == "down"
    assert body["seek_yaw"]["direction"] == "none"
    assert abs(body["seek_pitch"]["degrees"] - 2.5) < 1e-6
    assert body["pitch"]["direction"] == "none"


def test_unrecognized_while_looking_down_pitches_up():
    """已经接近正俯视却仍未识别时，改为上仰 2.5°，不再继续往下转。"""
    color, depth = _scene(140, 180, 100, 140)
    adjustment = plan_view(
        color, depth, _intr(), COLORS, gravity=np.array([0.0, 0.0, 1.0]), recognized=False
    )
    body = adjustment.as_dict()
    assert body["seek_pitch"]["direction"] == "up"
    assert abs(body["seek_pitch"]["degrees"] - 2.5) < 1e-6
    assert body["seek_yaw"]["direction"] == "none"


def test_short_miss_does_not_seek():
    """连续失败还没到门限时，即使方块偏在角落，也不给 seek_。"""
    color, depth = _scene(40, 80, 40, 80)
    adjustment = plan_view(
        color,
        depth,
        _intr(),
        COLORS,
        gravity=np.array([0.0, 1.0, 0.0]),
        recognized=False,
        seek=False,
    )
    body = adjustment.as_dict()
    assert body["reason"] == "waiting"
    assert adjustment.ready
    assert "seek_pitch" not in body
    assert body["pitch"]["direction"] == "none"
    assert body["yaw"]["direction"] == "none"


def test_ground_loss_pitches_down_without_yaw():
    """连续地面失败时只下俯 2.5°，不跟着颜色块做转角。"""
    color, depth = _scene(40, 80, 40, 80)
    adjustment = plan_view(
        color,
        depth,
        _intr(),
        COLORS,
        gravity=np.array([0.0, 1.0, 0.0]),
        recognized=False,
        ground_lost=True,
    )
    body = adjustment.as_dict()
    assert body["reason"] == "ground"
    assert body["seek_pitch"]["direction"] == "down"
    assert abs(body["seek_pitch"]["degrees"] - 2.5) < 1e-6
    assert body["seek_yaw"]["direction"] == "none"
    assert body["pitch"]["direction"] == "none"


def test_ground_loss_stops_when_already_low():
    """镜头已经明显朝下、地面仍不够时，不再继续下俯。"""
    color, depth = _scene(140, 180, 100, 140)
    adjustment = plan_view(
        color,
        depth,
        _intr(),
        COLORS,
        gravity=np.array([0.0, 0.0, 1.0]),
        recognized=False,
        ground_lost=True,
    )
    body = adjustment.as_dict()
    assert body["reason"] == "ground"
    assert body["seek_pitch"]["direction"] == "none"
    assert body["seek_yaw"]["direction"] == "none"
    assert adjustment.ready


def test_seek_stops_after_limit():
    """这一轴累计已经到 15° 时，本拍不再加。"""
    color, depth = _scene(40, 80, 40, 80)
    adjustment = plan_view(
        color,
        depth,
        _intr(),
        COLORS,
        gravity=np.array([0.0, 1.0, 0.0]),
        recognized=False,
        pitch_room=0.0,
        yaw_room=0.0,
    )
    body = adjustment.as_dict()
    assert body["seek_pitch"]["direction"] == "none"
    assert body["seek_yaw"]["direction"] == "none"


def test_nadir_right_edge_yaws_then_pitches_up():
    """正俯视且方块被右边界切掉时，先向右转，再略微上仰。"""
    color, depth = _scene(280, 320, 100, 140)
    gravity = np.array([0.0, 0.0, 1.0])
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=gravity)
    assert adjustment.clipped["right"]
    assert adjustment.yaw.direction == "right"
    assert adjustment.pitch.direction == "up"
    assert adjustment.yaw.degrees > 45.0


def test_on_axis_while_pitched_stays_still():
    """已经低头、方块又在光轴上时，不再给俯仰和转角。"""
    color, depth = _scene(140, 180, 100, 140)
    pitch = np.deg2rad(35.0)
    gravity = np.array([0.0, np.cos(pitch), np.sin(pitch)])
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=gravity)
    assert adjustment.ready
    assert adjustment.pitch.direction == "none"
    assert adjustment.yaw.direction == "none"
    assert not adjustment.gravity_assumed


def test_recognition_output_carries_pitch_and_turn():
    """识别结果里附上这一帧的俯仰和原地转角，不改已经融合好的位姿。"""
    from main import with_view

    pose = {"frame": "world", "x": 0.1, "y": 0.2, "yaw": 0.3}
    view = {
        "pitch": {"direction": "down", "radians": 0.07, "degrees": 4.0, "signed_radians": 0.07},
        "yaw": {"direction": "right", "radians": 0.1, "degrees": 6.0, "signed_radians": 0.1},
    }
    printed = with_view(pose, view)
    assert printed["x"] == 0.1
    assert printed["yaw"] == 0.3
    assert printed["view"]["pitch"]["direction"] == "down"
    assert printed["view"]["yaw"]["direction"] == "right"
    assert "view" not in pose
    assert with_view(None, view) == {"view": view}
    assert with_view({}, view) == {"view": view}


def test_unrecognized_without_color_still_pitches():
    """红蓝也没分出来、连续未识别时，仍下俯 2.5°，不把两个轴留在 none。"""
    color = np.full((240, 320, 3), 40, dtype=np.uint8)
    depth = np.full((240, 320), 1.0, dtype=np.float32)
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=np.array([0.0, 1.0, 0.0]), recognized=False)
    body = adjustment.as_dict()
    assert not adjustment.visible
    assert body["reason"] == "unrecognized"
    assert body["pitch"]["direction"] == "none"
    assert body["seek_pitch"]["direction"] == "down"
    assert abs(body["seek_pitch"]["degrees"] - 2.5) < 1e-6
    assert body["seek_yaw"]["direction"] == "none"


def test_seek_waits_for_consecutive_misses():
    """5 帧失败还不转。第 6 帧才允许微调；其中地面也连续失败时只俯仰。"""
    from main import seek_ready

    assert seek_ready(5, 5) == (False, False)
    assert seek_ready(6, 2) == (True, False)
    assert seek_ready(6, 6) == (True, True)


def test_failed_ground_reuses_last_plane_until_the_camera_turns():
    """拟合失败或原点跳开时，相机没转就沿用上一块平面。一开始转就丢掉。"""
    from ground.plane import Plane, reuse_stable_plane

    stable = Plane(np.array([0.0, 0.0, 1.0]), 1.0)
    jumped = Plane(np.array([0.0, 0.0, 1.0]), 1.2)
    used, kept, left = reuse_stable_plane(None, stable, 3, False, 0.03, 8)
    assert used is stable and kept is stable and left == 2
    used, kept, left = reuse_stable_plane(jumped, stable, 4, False, 0.03, 8)
    assert used is stable and left == 3
    used, kept, left = reuse_stable_plane(None, stable, 1, True, 0.03, 8)
    assert used is None and kept is None and left == 0
    close = Plane(np.array([0.0, 0.0, 1.0]), 1.01)
    used, kept, left = reuse_stable_plane(close, stable, 1, False, 0.03, 8)
    assert used is close and kept is close and left == 8


def test_no_cube_does_not_move():
    """已经按识别成功调用、画面里又没有红蓝块时不转动，并标明是假设光轴水平。"""
    color = np.full((240, 320, 3), 40, dtype=np.uint8)
    depth = np.full((240, 320), 1.0, dtype=np.float32)
    adjustment = plan_view(color, depth, _intr(), COLORS, gravity=None)
    assert not adjustment.visible
    assert adjustment.ready
    assert adjustment.reason == "no_cube"
    assert adjustment.gravity_assumed
    assert adjustment.as_dict()["pitch"]["direction"] == "none"
