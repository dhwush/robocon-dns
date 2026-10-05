"""深度预处理与 D435i 取流。

几何计算始终用配置文件里的内参，不用设备当时读出的内参覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- 一帧数据 -----------------------------------------------------------------
# color 为 BGR，depth 单位是米，gravity 是相机系中的重力方向（静止时由加速度计得到）。

@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    gravity: np.ndarray | None = None


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
    out = np.array(depth, dtype=float, copy=True)
    local_max = np.full(out.shape, -np.inf)
    local_min = np.full(out.shape, np.inf)
    count = np.zeros(out.shape, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            # roll 会把对面一行卷过来，卷进来的边必须作废。
            if dy > 0:
                shifted[:dy, :] = np.nan
            elif dy < 0:
                shifted[dy:, :] = np.nan
            if dx > 0:
                shifted[:, :dx] = np.nan
            elif dx < 0:
                shifted[:, dx:] = np.nan
            ok = np.isfinite(shifted)
            local_max = np.where(ok, np.maximum(local_max, shifted), local_max)
            local_min = np.where(ok, np.minimum(local_min, shifted), local_min)
            count += ok
    jump = ((local_max - local_min) > thresh_m) & (count >= 3)
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
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        cur = np.array(depth, dtype=float, copy=True)
        if self._state is None or self._state.shape != cur.shape:
            self._state = cur
            return cur.copy()
        # 两帧都有值：按 alpha 混合。当前帧是洞、上一帧有值：沿用上一帧，用来补闪烁空洞。
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
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        # 彩色、深度、加速度一起开。几何仍用配置内参，这里的分辨率只决定取流大小。
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.accel)
        self.profile = self.pipeline.start(config)
        # 深度对齐到彩色，后面的像素才能和颜色掩膜共用同一套 (u, v)。
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        # 视差域里做空间/时间滤波更符合立体匹配的噪声形态，做完再变回深度。
        self.to_disp = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        # 空间滤波保留边缘。holes_fill 置 0，避免强空洞填充把方块边缘拉进背景。
        try:
            self.spatial.set_option(rs.option.holes_fill, 0)
        except Exception:
            pass
        self.temporal_rs = rs.temporal_filter()
        self._accel_to_color = self._accel_rotation()
        self._lock_color()

    def _accel_rotation(self) -> np.ndarray:
        rs = self._rs
        try:
            # 加速度计坐标系和彩色相机不一致，要用设备外参旋过来。失败时退回单位阵。
            accel = self.profile.get_stream(rs.stream.accel).as_motion_stream_profile()
            color = self.profile.get_stream(rs.stream.color)
            ext = accel.get_extrinsics_to(color)
            return np.array(ext.rotation, dtype=float).reshape(3, 3)
        except Exception:
            return np.eye(3)

    def _lock_color(self) -> None:
        # 自动曝光在平台移动时会连跳，颜色阈值跟着漂。锁定曝光和白平衡。
        rs = self._rs
        try:
            sensor = self.profile.get_device().first_color_sensor()
        except Exception:
            return
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 0)

    def read(self) -> Frame:
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("missing color or depth frame")
        # 深度 → 视差 → 空间滤波 → 时间滤波 → 视差回到深度。不补洞。
        filtered = self.to_depth.process(
            self.temporal_rs.process(self.spatial.process(self.to_disp.process(depth_frame)))
        )
        # z16 要乘深度尺度才是米。
        depth = np.asanyarray(filtered.get_data()).astype(float) * self.depth_scale
        color = np.asanyarray(color_frame.get_data())
        gravity = None
        accel = frames.first_or_default(self._rs.stream.accel)
        if accel is not None:
            data = accel.as_motion_frame().get_motion_data()
            specific = np.array([data.x, data.y, data.z], dtype=float)
            # 静止时加速度计测到的是支撑力，方向朝上。重力是它的反向单位向量。
            up = self._accel_to_color @ specific
            norm = np.linalg.norm(up)
            if norm > 1.0:
                gravity = -up / norm
        return Frame(color=color, depth=depth, gravity=gravity)

    def close(self) -> None:
        self.pipeline.stop()
