"""从相机起步位置连续跟踪，并用场地边校正漂移。

世界原点是第一帧的相机光心，轴与当时的相机轴重合。之后每帧：

- 陀螺仪只在视觉跟踪失败时维持朝向，平移先保持不动。
- 深度视觉里程计成功时，用它更新从上一帧到这一帧的旋转和平移。
- 场地边能解出时，把场地角在起步坐标系里的位置锁定一次，以后每次看见边都把跟踪拉回这个绝对位姿。

场地边定义的是场地上的固定参照，不是另一套世界原点。p_C = R p_W + t 的约定与外参模块相同。
"""

from __future__ import annotations

import cv2
import numpy as np

from geom.transform import deproject_pixels


def _skew(vector: np.ndarray) -> np.ndarray:
    # 反对称矩阵。旋转向量的指数映射要用它把轴写成叉乘。
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    # 旋转向量的方向是轴，长度是转角。小角度直接用一阶，避免除以接近 0 的角。
    theta = float(np.linalg.norm(rotation_vector))
    if theta < 1e-8:
        return np.eye(3) + _skew(rotation_vector)
    axis = np.asarray(rotation_vector, dtype=float) / theta
    skew = _skew(axis)
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _rotation_error(rotation: np.ndarray) -> float:
    # 相对单位阵的转角。接近 0 说明这一步陀螺几乎没转，不必改写位姿。
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求 p_target = R p_source + t。点数少于 3 时不能唯一确定刚体。"""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    # H = Σ (p)(q)^T。右奇异向量在 target 一侧，所以 R = V U^T。
    covariance = (src - src_mean).T @ (dst - dst_mean)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t = right_t.copy()
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = dst_mean - rotation @ src_mean
    return rotation, translation


def align_points(
    source: np.ndarray,
    target: np.ndarray,
    thresh_m: float = 0.02,
    iterations: int = 40,
    min_points: int = 12,
    rotation: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC 刚体。给定 rotation 时只估计平移，用来在视觉模糊时仍可采用陀螺仪朝向。"""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    count = len(src)
    if count < 3 or (rotation is None and count < min_points):
        return None
    generator = rng if rng is not None else np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_count = 0
    draws = 1 if rotation is not None else 3
    for _ in range(iterations):
        choice = generator.choice(count, size=min(draws, count), replace=False)
        if rotation is None:
            if len(choice) < 3:
                continue
            candidate_r, candidate_t = rigid_transform(src[choice], dst[choice])
        else:
            candidate_r = rotation
            candidate_t = dst[choice[0]] - rotation @ src[choice[0]]
        residual = np.linalg.norm((candidate_r @ src.T).T + candidate_t - dst, axis=1)
        inliers = residual < thresh_m
        inlier_count = int(inliers.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers
    if best_inliers is None or best_count < min_points:
        return None
    if rotation is None:
        return rigid_transform(src[best_inliers], dst[best_inliers])
    translation = np.mean(dst[best_inliers] - (rotation @ src[best_inliers].T).T, axis=0)
    return rotation, translation


class MotionTracker:
    """起步相机系下的外参。reliable 为假时调用方应停止输出，避免把跟丢后的冻结平移当成新位置。"""

    def __init__(self, hold_s: float = 0.3, vo_thresh_m: float = 0.02, min_points: int = 12):
        # 起步外参是单位旋转和零平移。hold_s 是跟丢后仍允许输出的时间。
        self.hold_s = float(hold_s)
        self.vo_thresh_m = float(vo_thresh_m)
        self.min_points = int(min_points)
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.initial_up: np.ndarray | None = None
        self.started = False
        self.reliable = True
        self.lost_s = 0.0
        self.source = "start"
        self.anchored = False
        self._field_rotation: np.ndarray | None = None
        self._field_translation: np.ndarray | None = None
        self._rotation_prev = np.eye(3)
        self._translation_prev = np.zeros(3)
        self._prev_gray: np.ndarray | None = None
        self._prev_depth: np.ndarray | None = None
        self._last_gyro_stamp: float | None = None

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        # 返回副本，避免调用方改到跟踪器内部的 R、t。
        return self.rotation.copy(), self.translation.copy()

    def set_initial_up(self, up_camera: np.ndarray) -> None:
        # 只在起步时记一次。以后地面法向变回这个方向，用来确认跟踪没有把世界轴拧斜。
        if self.initial_up is not None:
            return
        vector = np.asarray(up_camera, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return
        self.initial_up = vector / norm

    def start(self, color_bgr: np.ndarray, depth: np.ndarray, up_camera: np.ndarray | None = None) -> None:
        # 第一帧就是世界原点。这一帧之前的陀螺样品丢掉，避免把上电时刻算进位移。
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self._remember(color_bgr, depth)
        self.started = True
        self.reliable = True
        self.lost_s = 0.0
        self.source = "start"
        if up_camera is not None:
            self.set_initial_up(up_camera)

    def begin_frame(self) -> None:
        # 陀螺和视觉都相对这一帧开始时的位姿积分，所以要先记下当前值。
        self._rotation_prev = self.rotation.copy()
        self._translation_prev = self.translation.copy()

    def integrate_gyro(self, samples: list[tuple[float, np.ndarray]]) -> None:
        """samples 是彩色相机系角速度，单位 rad/s，按时间升序。没有样品时不改位姿。"""
        if not samples:
            return
        rotation_step = np.eye(3)
        stamp = self._last_gyro_stamp
        for sample_stamp, omega in samples:
            if stamp is None:
                stamp = float(sample_stamp)
                continue
            dt = float(sample_stamp) - stamp
            stamp = float(sample_stamp)
            # 丢样或时间回绕时跳过，避免一次把朝向转过一大圈。
            if dt <= 0.0 or dt > 0.1:
                continue
            # 角速度在当前机体系。固定点在新机体系里的坐标要乘 exp(-ω dt)。
            rotation_step = _so3_exp(-np.asarray(omega, dtype=float) * dt) @ rotation_step
        self._last_gyro_stamp = stamp
        if _rotation_error(rotation_step) < 1e-8:
            return
        # 纯旋转时，世界原点在相机里的向量跟着机体系一起转。平移留到视觉或场地边。
        self.rotation = rotation_step @ self._rotation_prev
        self.translation = rotation_step @ self._translation_prev
        self.source = "gyro"

    def observe_visual(self, color_bgr: np.ndarray, depth: np.ndarray, intr) -> str:
        """相对上一帧的视觉运动。返回 init、ok 或 lost。lost 时保留陀螺仪已经写入的预测。"""
        if self._prev_gray is None or self._prev_depth is None:
            self._remember(color_bgr, depth)
            return "init"
        motion = self._visual_motion(color_bgr, depth, intr)
        self._remember(color_bgr, depth)
        if motion is None:
            return "lost"
        rotation_step, translation_step = motion
        self.rotation = rotation_step @ self._rotation_prev
        self.translation = rotation_step @ self._translation_prev + translation_step
        self.source = "visual"
        return "ok"

    def observe_field(self, rotation_field: np.ndarray, translation_field: np.ndarray) -> None:
        """场地边外参 p_C = R_f p_F + t_f。第一次看见时把场地记入起步系，之后用它消除漂移。"""
        rotation_field = np.asarray(rotation_field, dtype=float)
        translation_field = np.asarray(translation_field, dtype=float)
        if not self.anchored:
            # 当前跟踪位姿把这次测量换算到起步系。以后不再用漂移后的跟踪重写这个锚点。
            self._field_rotation = self.rotation.T @ rotation_field
            self._field_translation = self.rotation.T @ (translation_field - self.translation)
            self.anchored = True
            self.source = "field"
            return
        rotation = rotation_field @ self._field_rotation.T
        translation = translation_field - rotation @ self._field_translation
        self.rotation = rotation
        self.translation = translation
        self.source = "field"

    def note_locked(self) -> None:
        # 视觉或场地边成功时清掉跟丢计时，输出可以继续。
        self.lost_s = 0.0
        self.reliable = True

    def note_lost(self, dt: float) -> None:
        # 累计跟丢时间。超过 hold_s 后 reliable 变假，调用方应停止输出。
        self.lost_s += max(float(dt), 0.0)
        self.reliable = self.lost_s <= self.hold_s

    def _remember(self, color_bgr: np.ndarray, depth: np.ndarray) -> None:
        # 留下灰度和深度，下一帧光流和反投影都相对这一帧。
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY) if color_bgr.ndim == 3 else np.asarray(color_bgr)
        self._prev_gray = gray
        self._prev_depth = np.asarray(depth, dtype=np.float32)

    def _visual_motion(self, color_bgr: np.ndarray, depth: np.ndarray, intr) -> tuple[np.ndarray, np.ndarray] | None:
        # 光流跟上特征后再配深度求刚体增量。点不够或对齐失败就返回空，朝向留给陀螺仪。
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY) if color_bgr.ndim == 3 else np.asarray(color_bgr)
        features = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=120, qualityLevel=0.01, minDistance=16)
        if features is None or len(features) < self.min_points:
            return None
        found, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            features,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if found is None:
            return None
        status = status.reshape(-1).astype(bool)
        previous = features.reshape(-1, 2)[status]
        current = found.reshape(-1, 2)[status]
        src, dst = self._pair_points(previous, self._prev_depth, current, depth, intr)
        if len(src) < self.min_points:
            return None
        return align_points(src, dst, thresh_m=self.vo_thresh_m, min_points=self.min_points)

    def _pair_points(self, previous_px, previous_depth, current_px, current_depth, intr):
        # 只保留两帧都有有效深度的特征，并反投影成相机系点对。
        src_px = []
        src_z = []
        dst_px = []
        dst_z = []
        height, width = previous_depth.shape[:2]
        for (u0, v0), (u1, v1) in zip(previous_px, current_px):
            x0, y0 = int(round(float(u0))), int(round(float(v0)))
            x1, y1 = int(round(float(u1))), int(round(float(v1)))
            if not (0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height):
                continue
            z0 = float(previous_depth[y0, x0])
            z1 = float(current_depth[y1, x1])
            if not (np.isfinite(z0) and np.isfinite(z1) and z0 > 0.2 and z1 > 0.2):
                continue
            src_px.append([u0, v0])
            src_z.append(z0)
            dst_px.append([u1, v1])
            dst_z.append(z1)
        if len(src_px) < 3:
            return np.zeros((0, 3)), np.zeros((0, 3))
        source = deproject_pixels(np.asarray(src_px), np.asarray(src_z), intr)
        target = deproject_pixels(np.asarray(dst_px), np.asarray(dst_z), intr)
        return source, target
