"""Depth filtering and an optional D435i source.

Live capture uses pyrealsense2 when a camera is opened. Pixel geometry
still uses the intrinsics loaded from config, not the device profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    gravity: np.ndarray | None = None


def clip_range(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    out = np.array(depth, dtype=float, copy=True)
    invalid = ~np.isfinite(out) | (out < zmin) | (out > zmax) | (out <= 0.0)
    out[invalid] = np.nan
    return out


def remove_flying_pixels(depth: np.ndarray, thresh_m: float) -> np.ndarray:
    """Drop pixels whose 3x3 neighborhood jumps by more than thresh_m."""
    out = np.array(depth, dtype=float, copy=True)
    local_max = np.full(out.shape, -np.inf)
    local_min = np.full(out.shape, np.inf)
    count = np.zeros(out.shape, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
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
    import cv2

    finite = np.isfinite(depth)
    if float(finite.mean()) < 0.7:
        return np.array(depth, dtype=float, copy=True)
    filled = np.where(finite, depth, 0.0).astype(np.float32)
    smooth = cv2.medianBlur(filled, ksize)
    mask = cv2.blur(finite.astype(np.float32), (ksize, ksize))
    out = smooth.astype(float)
    out[mask < 0.5] = np.nan
    return out


class TemporalDepthFilter:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
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
    out = clip_range(depth, zmin, zmax)
    out = remove_flying_pixels(out, flying_pixel_m)
    out = median_smooth(out, 3)
    if temporal is not None:
        out = temporal.apply(out)
    return out


class RealSenseSource:
    """Aligned color and depth. Intrinsics for geometry stay in the config file."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.accel)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.to_disp = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.temporal_rs = rs.temporal_filter()
        self.hole = rs.hole_filling_filter()
        self._accel_to_color = self._accel_rotation()

    def _accel_rotation(self) -> np.ndarray:
        rs = self._rs
        try:
            accel = self.profile.get_stream(rs.stream.accel).as_motion_stream_profile()
            color = self.profile.get_stream(rs.stream.color)
            ext = accel.get_extrinsics_to(color)
            return np.array(ext.rotation, dtype=float).reshape(3, 3)
        except Exception:
            return np.eye(3)

    def read(self) -> Frame:
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("missing color or depth frame")
        filtered = self.to_depth.process(
            self.hole.process(self.temporal_rs.process(self.spatial.process(self.to_disp.process(depth_frame))))
        )
        depth = np.asanyarray(filtered.get_data()).astype(float) * self.depth_scale
        color = np.asanyarray(color_frame.get_data())
        gravity = None
        accel = frames.first_or_default(self._rs.stream.accel)
        if accel is not None:
            data = accel.as_motion_frame().get_motion_data()
            specific = np.array([data.x, data.y, data.z], dtype=float)
            up = self._accel_to_color @ specific
            norm = np.linalg.norm(up)
            if norm > 1.0:
                gravity = -up / norm
        return Frame(color=color, depth=depth, gravity=gravity)

    def close(self) -> None:
        self.pipeline.stop()
