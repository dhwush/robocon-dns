"""视觉里程计鲁棒性：单独定坐标系，以及和场地边识别交替时的误差。

场景是起步相机正对的一块带纹理平面，平面上有一个已知直角。
相机沿起步系 +X 平移并绕 Z 缓慢转动。场地边在中间一段不画，检测应失败，只剩视觉。
"""

from __future__ import annotations

import numpy as np

from field.extrinsic import calibrate_from_image
from field.motion import MotionTracker
from geom.transform import Rz, project_points
from ground.plane import fit_ground, sample_points
from sim.scene import intrinsics_for


FIELD_LENGTH = 0.55
FIELD_WIDTH = 0.40
PLANE_Z = 1.0


def trajectory(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # 每帧平移 6 mm、绕起步系 Z 转 0.008 rad。p_C = R p_W + t。
    poses = []
    for index in range(count):
        yaw = index * 0.008
        camera_in_world = np.array([index * 0.006, 0.0, 0.0])
        rotation_wc = Rz(yaw)
        rotation = rotation_wc.T
        translation = -rotation @ camera_in_world
        poses.append((rotation, translation))
    return poses


def _shade(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 2 cm 一格的随机灰度。格子边只有几个像素，短于霍夫的 30 像素，不会被当成场地边。
    ix = np.floor(x / 0.02).astype(np.int64)
    iy = np.floor(y / 0.02).astype(np.int64)
    mixed = (ix * 73856093) ^ (iy * 19349663)
    return (120 + (np.abs(mixed) % 40)).astype(np.uint8)


def render(intr, rotation: np.ndarray, translation: np.ndarray, show_edges: bool, noise_m: float, seed: int):
    # 每个像素的射线交到 Z = 1 的平面上，避免平面轮廓被霍夫当成场地边。
    height, width = intr.height, intr.width
    rows, cols = np.mgrid[0:height, 0:width]
    ray_x = (cols - intr.cx) / intr.fx
    ray_y = (rows - intr.cy) / intr.fy
    ray = np.stack([ray_x, ray_y, np.ones((height, width))], axis=-1)
    ray_world = ray @ rotation
    offset = rotation.T @ translation
    denom = ray_world[:, :, 2]
    scale = (PLANE_Z + offset[2]) / np.where(np.abs(denom) < 1e-8, np.nan, denom)
    valid = np.isfinite(scale) & (scale > 0.3)
    world = scale[:, :, None] * ray_world - offset
    gray = np.full((height, width), 40, dtype=np.uint8)
    depth = np.full((height, width), np.nan, dtype=float)
    gray[valid] = _shade(world[:, :, 0][valid], world[:, :, 1][valid])
    depth[valid] = scale[valid]
    # 纹理模糊后梯度低于 Canny 阈值，霍夫只留下后面画上的场地边。角点仍够光流跟踪。
    import cv2

    gray[:] = cv2.GaussianBlur(gray, (5, 5), 0)
    if show_edges:
        _paint_edges(gray, intr, rotation, translation)
    if noise_m > 0.0:
        finite = np.isfinite(depth)
        depth[finite] += np.random.default_rng(seed).normal(0.0, noise_m, size=int(finite.sum()))
    color = np.repeat(gray[:, :, None], 3, axis=2)
    return color, depth


def _paint_edges(gray: np.ndarray, intr, rotation: np.ndarray, translation: np.ndarray) -> None:
    # 在纹理平面上画两条直角边。不画的帧应让场地边检测失败，只剩视觉跟踪。
    import cv2

    corner = np.array([0.0, 0.0, PLANE_Z])
    ends = (np.array([FIELD_LENGTH, 0.0, PLANE_Z]), np.array([0.0, FIELD_WIDTH, PLANE_Z]))
    for end in ends:
        samples = np.linspace(0.0, 1.0, 40)[:, None]
        world = corner * (1.0 - samples) + end * samples
        camera = (rotation @ world.T).T + translation
        if np.any(camera[:, 2] <= 0.3):
            continue
        pixels = np.rint(project_points(camera, intr)).astype(int)
        cv2.line(gray, tuple(pixels[0]), tuple(pixels[-1]), 0, 5, cv2.LINE_AA)


def field_pose(color, depth, intr, gravity: np.ndarray):
    # 这一帧能解出场地角就返回外参，边不在画面里返回空。
    plane = fit_ground(sample_points(depth, intr, stride=3), gravity, thresh=0.01, max_angle_deg=8.0)
    if plane is None:
        return None
    return calibrate_from_image(color, intr, plane, FIELD_LENGTH, FIELD_WIDTH)


def _rotation_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    # 两个旋转差多少度。用来看视觉和场地边校正之后还剩多少朝向误差。
    delta = estimated.T @ truth
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run_case(intr, frames, gravities, field_frames: set[int] | None) -> dict:
    # field_frames 为空时只跑视觉。给出帧号时，这些帧能看见边就用场地角拉回漂移。
    tracker = MotionTracker(hold_s=0.3, min_points=8)
    rows = []
    for index, ((color, depth), (rotation, translation), gravity) in enumerate(zip(frames, gravities[0], gravities[1])):
        if index == 0:
            tracker.start(color, depth, up_camera=-gravity)
        else:
            tracker.begin_frame()
            visual = tracker.observe_visual(color, depth, intr)
            recognized = None
            if field_frames is not None and index in field_frames:
                recognized = field_pose(color, depth, intr, gravity)
                if recognized is not None:
                    tracker.observe_field(*recognized)
                    tracker.note_locked()
                elif visual == "ok":
                    tracker.note_locked()
                else:
                    tracker.note_lost(1.0 / 30.0)
            elif visual == "ok":
                tracker.note_locked()
            else:
                tracker.note_lost(1.0 / 30.0)
        estimated_r, estimated_t = tracker.pose()
        rows.append(
            {
                "frame": index,
                "source": tracker.source,
                "reliable": tracker.reliable,
                "field_seen": field_frames is not None and index in field_frames and tracker.source == "field",
                "trans_mm": float(np.linalg.norm(estimated_t - translation) * 1000.0),
                "rot_deg": _rotation_deg(estimated_r, rotation),
            }
        )
    return {"rows": rows, "anchored": tracker.anchored}


def _summary(result: dict) -> dict:
    # 终点误差、过程中的最大误差，以及有多少帧真正锁上了场地边。
    rows = result["rows"]
    trans = np.array([row["trans_mm"] for row in rows])
    rot = np.array([row["rot_deg"] for row in rows])
    lost = sum(1 for row in rows if row["source"] not in ("start", "visual", "field") or not row["reliable"])
    field_hits = sum(1 for row in rows if row["field_seen"])
    return {
        "final_trans_mm": trans[-1],
        "final_rot_deg": rot[-1],
        "max_trans_mm": float(trans.max()),
        "max_rot_deg": float(rot.max()),
        "field_hits": field_hits,
        "lost_frames": lost,
        "anchored": result["anchored"],
    }


def build_sequence(intr, show_edges_at: set[int], noise_m: float = 0.002):
    # 同一条轨迹渲染 24 帧。只有 show_edges_at 里的帧画场地边。
    poses = trajectory(24)
    frames = []
    gravities = []
    truths = []
    for index, (rotation, translation) in enumerate(poses):
        gravity = rotation @ np.array([0.0, 0.0, 1.0])
        frames.append(render(intr, rotation, translation, index in show_edges_at, noise_m, seed=1000 + index))
        gravities.append(gravity)
        truths.append((rotation, translation))
    return frames, (truths, gravities)


def test_vo_alone_and_with_field_gaps():
    """慢速平移加小转角时，单独视觉的终点误差停留在数毫米。

    第 8–15 帧不画场地边，这些帧的来源应是 visual。边重新出现后要能再次解出。
    场地边解出的角点比这段视觉累加更散，配合之后的终点误差会大于单独视觉。
    """
    intr = intrinsics_for(320, 240, model="none")
    visible = set(range(24)) - set(range(8, 16))
    frames, packed = build_sequence(intr, visible)
    alone = run_case(intr, frames, packed, field_frames=None)
    paired_frames, paired_packed = build_sequence(intr, visible)
    paired = run_case(intr, paired_frames, paired_packed, field_frames=visible)
    alone_summary = _summary(alone)
    paired_summary = _summary(paired)
    assert alone_summary["lost_frames"] == 0
    assert alone_summary["final_trans_mm"] < 15.0
    assert alone_summary["final_rot_deg"] < 1.0
    gap = [row for row in paired["rows"] if 8 <= row["frame"] <= 15]
    assert all(row["source"] == "visual" for row in gap)
    assert paired_summary["field_hits"] >= 10
    assert paired["rows"][16]["field_seen"]
    assert paired_summary["final_trans_mm"] > alone_summary["final_trans_mm"]


def main() -> None:
    # 直接运行时打印单独视觉和穿插场地边两路的逐帧误差，方便对照。
    intr = intrinsics_for(320, 240, model="none")
    visible = set(range(24)) - set(range(8, 16))
    frames, packed = build_sequence(intr, visible)
    alone = run_case(intr, frames, packed, field_frames=None)
    paired_frames, paired_packed = build_sequence(intr, visible)
    paired = run_case(intr, paired_frames, paired_packed, field_frames=visible)
    print("frame  alone_mm  alone_deg  paired_mm  paired_deg  paired_source")
    for left, right in zip(alone["rows"], paired["rows"]):
        print(
            f"{left['frame']:5d}  {left['trans_mm']:8.2f}  {left['rot_deg']:9.3f}  "
            f"{right['trans_mm']:9.2f}  {right['rot_deg']:10.3f}  {right['source']}"
        )
    print("alone", _summary(alone))
    print("paired", _summary(paired))


if __name__ == "__main__":
    main()
