# 起步相机跟踪：陀螺仪旋转、视觉刚体、场地边把漂移拉回去。

import cv2
import numpy as np

from field.motion import MotionTracker, align_points, rigid_transform
from geom.transform import Rz
from sim.scene import intrinsics_for


def test_rigid_transform_recovers_rotation_and_translation():
    """已知刚体作用在一组点上，对齐结果应回到原来的旋转和平移。"""
    rotation = Rz(0.4)
    translation = np.array([0.02, -0.01, 0.03])
    source = np.array(
        [
            [0.2, 0.0, 1.0],
            [-0.1, 0.2, 1.1],
            [0.0, -0.2, 0.9],
            [0.15, 0.1, 1.2],
        ]
    )
    target = (rotation @ source.T).T + translation
    estimated_r, estimated_t = rigid_transform(source, target)
    np.testing.assert_allclose(estimated_r, rotation, atol=1e-8)
    np.testing.assert_allclose(estimated_t, translation, atol=1e-8)


def test_align_points_ignores_outlier():
    """一个点被挪开 20 cm。内点阈值 2 cm 时，结果仍应贴着其余点的真值。"""
    rotation = np.eye(3)
    translation = np.array([0.05, 0.0, 0.0])
    source = np.array([[0.1, 0.2, 1.0], [-0.2, 0.1, 1.0], [0.0, -0.15, 1.2], [0.2, -0.1, 0.8], [0.05, 0.05, 1.1]])
    target = source + translation
    target = target.copy()
    target[0] += np.array([0.2, 0.0, 0.0])
    estimated_r, estimated_t = align_points(source, target, min_points=3, rng=np.random.default_rng(0))
    np.testing.assert_allclose(estimated_r, rotation, atol=1e-6)
    np.testing.assert_allclose(estimated_t, translation, atol=1e-6)


def test_gyro_quarter_turn_about_z():
    """绕相机 Z 转 90°。世界系 +X 上的点在新相机系里应落到 -Y。"""
    tracker = MotionTracker()
    blank = np.zeros((16, 16), dtype=np.uint8)
    tracker.start(blank, np.ones((16, 16)))
    tracker.begin_frame()
    omega = np.array([0.0, 0.0, np.pi / 2.0])
    # 陀螺样品间隔必须短于 0.1 s，和实机 200 Hz 一致。总转角仍是 90°。
    samples = [(i * 0.05, omega) for i in range(21)]
    tracker.integrate_gyro(samples)
    rotation, translation = tracker.pose()
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(translation, 0.0, atol=1e-6)
    assert tracker.source == "gyro"


def test_field_edge_removes_translation_drift():
    """先在真位姿上锁定场地角，再人为平移漂移。同一场地测量应把外参拉回锁定时的值。"""
    tracker = MotionTracker()
    blank = np.zeros((16, 16), dtype=np.uint8)
    tracker.start(blank, np.ones((16, 16)))
    field_rotation = Rz(0.3)
    field_translation = np.array([0.4, -0.2, 1.2])
    tracker.observe_field(field_rotation, field_translation)
    assert tracker.anchored
    np.testing.assert_allclose(tracker.rotation, np.eye(3), atol=1e-8)
    tracker.translation = np.array([0.08, -0.02, 0.01])
    tracker.observe_field(field_rotation, field_translation)
    rotation, translation = tracker.pose()
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(translation, 0.0, atol=1e-8)
    assert tracker.source == "field"


def test_visual_shift_of_a_textured_plane():
    """正对 1 m 的纹理平面在图像上左移。相机相当于向右平移，位移约等于像素位移乘深度再除以焦距。"""
    intr = intrinsics_for(160, 120, model="none")
    rng = np.random.default_rng(2)
    texture = rng.integers(0, 255, size=(50, 50), dtype=np.uint8)

    def frame(shift: int):
        color = np.zeros((120, 160), dtype=np.uint8)
        depth = np.full((120, 160), np.nan)
        color[30:80, 40 + shift : 90 + shift] = texture
        depth[30:80, 40 + shift : 90 + shift] = 1.0
        return cv2.cvtColor(color, cv2.COLOR_GRAY2BGR), depth

    tracker = MotionTracker(min_points=8)
    first_color, first_depth = frame(0)
    tracker.start(first_color, first_depth)
    tracker.begin_frame()
    status = tracker.observe_visual(*frame(-8), intr)
    assert status == "ok"
    _rotation, translation = tracker.pose()
    # 特征在图像上向左，相机系里的 X 变小，相当于相机向自己的右方移动。
    expected = -8.0 * 1.0 / intr.fx
    assert abs(translation[0] - expected) < 0.015
    assert abs(translation[1]) < 0.015
    assert abs(translation[2]) < 0.015


def test_tracking_stops_after_hold():
    """视觉连续失败超过保持时间后，结果不可再拿去输出。重新锁定后恢复。"""
    tracker = MotionTracker(hold_s=0.3)
    tracker.started = True
    tracker.note_lost(0.2)
    assert tracker.reliable
    tracker.note_lost(0.2)
    assert not tracker.reliable
    tracker.note_locked()
    assert tracker.reliable
    assert tracker.lost_s == 0.0
