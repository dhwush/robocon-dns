"""当前车身右前点坐标：X右、Y前、Z上；相机仅允许相对车身俯仰/横滚。"""
from __future__ import annotations

import numpy as np

# 水平相机的车身三轴在光学系中的方向，p_cam = R p_body + t。
LEVEL_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
DEFAULT_OPTICAL_CENTER = np.array([-.211, .05, .675])


def _vector(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(label + '必须为三个有限数值')
    return value


def body_extrinsic(gravity, cfg=None, plane=None, camera_height=None):
    """按本帧重力补偿相机倾斜，不累计车身yaw或平移。

    车身在水平地面、相机偏航与车头一致。IMU缺失时可用本帧地面法向；
    两者都不可用时拒绝输出，不能把相机假设为水平。
    camera_pivot_to_optical_m 是相机水平时转轴到光心的车身系向量。
    零值表示转轴经过光心。非零时相机转动会同时改变光心的车身坐标。
    """
    cfg = {} if cfg is None else cfg
    if gravity is None and plane is not None:
        gravity = -np.asarray(plane.normal, dtype=float)
    if gravity is None:
        raise ValueError('没有有效重力或地面法向，不能换算车身坐标')
    down = _vector(gravity, '重力')
    length = float(np.linalg.norm(down))
    if length < 1e-6:
        raise ValueError('重力方向无效')
    down = down / length
    up = -down
    right = np.array([1., 0., 0.])
    right = right - float(right @ up) * up
    length = float(np.linalg.norm(right))
    if length < .05:
        raise ValueError('相机倾斜接近奇异姿态，不能确定车身方向')
    right /= length
    forward = np.cross(up, right)
    rotation = np.column_stack((right, forward, up))
    optical = _vector(cfg.get('camera_optical_center_m', DEFAULT_OPTICAL_CENTER), '安装光心').copy()
    if camera_height is not None:
        optical[2] = float(camera_height)
    if not np.isfinite(optical).all():
        raise ValueError('光心高度无效')
    lever = _vector(cfg.get('camera_pivot_to_optical_m', [0., 0., 0.]), '转轴偏移')
    assembly_turn = rotation.T @ LEVEL_ROTATION
    optical += (assembly_turn - np.eye(3)) @ lever
    return rotation, -rotation @ optical
