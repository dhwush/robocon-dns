"""深度预处理与 D435i 取流。

几何计算始终用配置文件里的内参，不用设备当时读出的内参覆盖。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np


# --- 一帧数据 -----------------------------------------------------------------
# color 为 BGR，depth 单位是米，gravity 是相机系中的重力方向（静止时由加速度计得到）。

@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    gravity: np.ndarray | None = None
    # 彩色相机系角速度 (时间戳秒, rad/s)。没有陀螺仪时为空，跟踪只靠视觉和场地边。
    gyro: list | None = None


# --- 深度清洗 -----------------------------------------------------------------
# 先丢掉 0.3–3 m 以外的值，再去掉边缘飞点，最后做中值平滑。
# 稀疏深度不做中值：用 0 填空洞再模糊会把有效点抹掉。

def clip_range(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # D435i 有效距离大约 0.3–3 m。更近的视差不可靠，更远的深度噪声按距离平方变大。
    # 非法值统一写成 NaN，后面的拟合会跳过它们。
    out = np.array(depth, dtype=float, copy=True)
    invalid = ~np.isfinite(out) | (out < zmin) | (out > zmax) | (out <= 0.0)
    out[invalid] = np.nan
    return out


def remove_flying_pixels(depth: np.ndarray, thresh_m: float) -> np.ndarray:
    # 飞点出现在深度不连续的边缘：一个像素跳到前景或背景。
    # 看 3×3 邻域的最大深度差，超过阈值就丢掉该像素。邻域有效点少于 3 个时不判，避免把稀疏点误删。
    # 膨胀和腐蚀一次扫完整幅图，不再对每个邻域各拷贝一次。
    import cv2

    src = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(src)
    count = cv2.boxFilter(
        finite.astype(np.float32),
        -1,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    hi = np.where(finite, src, np.float32(-1.0e6))
    lo = np.where(finite, src, np.float32(1.0e6))
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(hi, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=-1.0e6)
    local_min = cv2.erode(lo, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1.0e6)
    jump = ((local_max - local_min) > thresh_m) & (count >= 3)
    out = np.array(depth, dtype=float, copy=True)
    out[jump] = np.nan
    return out


def median_smooth(depth: np.ndarray, ksize: int = 3) -> np.ndarray:
    # 中值能压散粒噪声，又比均值更能保住深度台阶。
    import cv2

    finite = np.isfinite(depth)
    # 有效像素不到七成时，空洞会被 0 填上，中值会把真深度拉向 0。这种情况直接返回。
    if float(finite.mean()) < 0.7:
        return np.array(depth, dtype=float, copy=True)
    filled = np.where(finite, depth, 0.0).astype(np.float32)
    smooth = cv2.medianBlur(filled, ksize)
    # 窗口里有效像素不足一半的位置仍视为空洞。
    mask = cv2.blur(finite.astype(np.float32), (ksize, ksize))
    out = smooth.astype(float)
    out[mask < 0.5] = np.nan
    return out


# --- 多帧深度 -----------------------------------------------------------------
# 没有 RealSense 时间滤波器时，用指数滑动平均补洞。

class TemporalDepthFilter:
    def __init__(self, alpha: float = 0.4):
        # alpha 越大越跟当前帧。还没有上一帧时，apply 先把这一帧当作状态。
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        # 两帧都有效时做指数平均；当前帧是洞、上一帧有值时沿用上一帧补闪烁。
        cur = np.array(depth, dtype=float, copy=True)
        if self._state is None or self._state.shape != cur.shape:
            self._state = cur
            return cur.copy()
        both = np.isfinite(cur) & np.isfinite(self._state)
        blended = self.alpha * cur + (1.0 - self.alpha) * self._state
        out = cur.copy()
        out[both] = blended[both]
        only_old = ~np.isfinite(cur) & np.isfinite(self._state)
        out[only_old] = self._state[only_old]
        self._state = out
        return out


def filter_depth(
    depth: np.ndarray,
    zmin: float = 0.3,
    zmax: float = 3.0,
    flying_pixel_m: float = 0.08,
    temporal: TemporalDepthFilter | None = None,
) -> np.ndarray:
    # 顺序不能颠倒：先截距离，再删飞点，否则飞点的大跳变会污染邻域判断。
    out = clip_range(depth, zmin, zmax)
    out = remove_flying_pixels(out, flying_pixel_m)
    out = median_smooth(out, 3)
    if temporal is not None:
        out = temporal.apply(out)
    return out


# --- 实机取流 -----------------------------------------------------------------
# 彩色与深度对齐。深度在视差域做空间边缘保持滤波和时间滤波。
# 空洞填充会把背景拉进方块边缘，这里关掉。彩色曝光和白平衡锁定。
# 加速度静止时指向“上”，重力取反方向，并旋到彩色相机坐标系。

class RealSenseSource:
    """彩色与深度对齐。几何用的内参始终来自配置文件，不用设备当时读出的数覆盖。"""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        # 彩色、深度和惯性一起开。对齐和滤波要在回调之前建好，否则第一帧还没有这些对象。
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        # 彩色、深度、加速度一起开。几何仍用配置内参，这里的分辨率只决定取流大小。
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending: tuple | None = None
        self._gyro: deque = deque(maxlen=400)
        self._accel: np.ndarray | None = None
        self._gyro_to_color = np.eye(3)
        self._accel_to_color = np.eye(3)
        self._gyro_ready = False
        self._video_ready = False
        self.depth_scale = 0.001
        # 对齐和滤波要在启动回调之前建好，否则第一帧进来时还没有这些对象。
        self.align = rs.align(rs.stream.color)
        self.to_disp = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        try:
            self.spatial.set_option(rs.option.holes_fill, 0)
        except Exception:
            pass
        self.temporal_rs = rs.temporal_filter()
        try:
            self.profile = self.pipeline.start(config, self._on_frame)
            self._gyro_on = True
        except Exception:
            # D435 没有陀螺仪时关掉该流再启动。视觉跟踪和场地边仍然可用。
            self.pipeline = rs.pipeline()
            config.disable_stream(rs.stream.gyro)
            self.profile = self.pipeline.start(config, self._on_frame)
            self._gyro_on = False
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self._accel_to_color = self._accel_rotation()
        if self._gyro_on:
            self._gyro_to_color = self._motion_rotation(rs.stream.gyro)
        self._video_ready = True
        self._gyro_ready = True
        self._lock_color()

    def _accel_rotation(self) -> np.ndarray:
        # 加速度计到彩色相机的旋转。重力要先变到彩色系，再交给地面拟合。
        return self._motion_rotation(self._rs.stream.accel)

    def _motion_rotation(self, stream) -> np.ndarray:
        # 读设备给出的惯性系到彩色相机的外参。读失败时退回单位阵，避免整路取流起不来。
        try:
            # 惯性系和彩色相机不一致，要用设备外参旋过来。失败时退回单位阵。
            motion = self.profile.get_stream(stream).as_motion_stream_profile()
            color = self.profile.get_stream(self._rs.stream.color)
            ext = motion.get_extrinsics_to(color)
            return np.array(ext.rotation, dtype=float).reshape(3, 3)
        except Exception:
            return np.eye(3)

    def _gravity_from_specific(self, specific: np.ndarray) -> np.ndarray | None:
        # 加速度计静止时指向“上”。旋到彩色系后取反，得到相机系里的重力方向。
        up = self._accel_to_color @ specific
        norm = np.linalg.norm(up)
        if norm <= 1.0:
            return None
        return -up / norm

    def _on_frame(self, frame) -> None:
        # 回调里把彩色帧和陀螺样品拆开。图像速率约 30 Hz，陀螺约 200 Hz，不能只留每个图像帧上的一颗陀螺。
        rs = self._rs
        if frame.is_frameset():
            if not self._video_ready:
                return
            aligned = self.align.process(frame.as_frameset())
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return
            filtered = self.to_depth.process(
                self.temporal_rs.process(self.spatial.process(self.to_disp.process(depth_frame)))
            )
            depth = np.asanyarray(filtered.get_data()).astype(float) * self.depth_scale
            color = np.asanyarray(color_frame.get_data()).copy()
            gravity = None
            # 对齐结果经常不带加速度。空帧对象不是 None，直接读会抛掉整帧彩色和深度。
            accel = aligned.first_or_default(rs.stream.accel)
            if accel:
                data = accel.as_motion_frame().get_motion_data()
                specific = np.array([data.x, data.y, data.z], dtype=float)
                gravity = self._gravity_from_specific(specific)
            if gravity is None:
                with self._lock:
                    if self._accel is not None:
                        gravity = self._accel.copy()
            with self._lock:
                gyro = list(self._gyro)
                self._gyro.clear()
                self._pending = (color, depth, gravity, gyro)
            self._event.set()
            return
        if not self._gyro_ready or not frame.is_motion_frame():
            return
        motion = frame.as_motion_frame()
        kind = motion.profile.stream_type()
        data = motion.get_motion_data()
        sample = np.array([data.x, data.y, data.z], dtype=float)
        if kind == rs.stream.accel:
            gravity = self._gravity_from_specific(sample)
            if gravity is not None:
                with self._lock:
                    self._accel = gravity
            return
        if kind != rs.stream.gyro:
            return
        omega = self._gyro_to_color @ sample
        with self._lock:
            self._gyro.append((float(motion.timestamp) * 1e-3, omega))

    def _set_clamped(self, sensor, option, value: float) -> None:
        # 不同固件的上下限不一样。超出范围就夹到设备允许的值，避免设置失败把整路取流打断。
        if not sensor.supports(option):
            return
        try:
            span = sensor.get_option_range(option)
            sensor.set_option(option, float(np.clip(value, span.min, span.max)))
        except Exception:
            return

    def _lock_color(self) -> None:
        # 自动曝光在平台移动时会连跳，颜色阈值跟着漂。锁定曝光和白平衡。
        # 关掉自动后若留着默认曝光和增益，白天会把红蓝冲成接近白色。
        rs = self._rs
        try:
            sensor = self.profile.get_device().first_color_sensor()
        except Exception:
            return
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        self._set_clamped(sensor, rs.option.exposure, 50)
        self._set_clamped(sensor, rs.option.gain, 16)
        self._set_clamped(sensor, rs.option.brightness, 0)
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 0)

    def read(self) -> Frame:
        # 等到回调写好一帧再取出。超时说明彩色或深度没来。
        if not self._event.wait(timeout=5.0):
            raise RuntimeError("missing color or depth frame")
        self._event.clear()
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            raise RuntimeError("missing color or depth frame")
        color, depth, gravity, gyro = pending
        return Frame(color=color, depth=depth, gravity=gravity, gyro=gyro)

    def close(self) -> None:
        # 停掉管线，释放相机，避免下次启动报设备忙。
        self.pipeline.stop()
