"""三次静止摆放：观察满最短时间且窗口稳定后，才记下世界系 x、y 和 yaw。

丢帧、相邻帧地面中心跳过 sample_jump_m，或出现不小于 10° 的转动指令时，稳定窗清空。
计时不因丢帧清零。第一次从 begin 起算；写出之后要等方块离开，下一次才重新计时。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fuse.median import _align_period, fuse_poses

_MOVE_KEYS = ("pitch", "yaw", "seek_pitch", "seek_yaw", "footprint_pitch", "footprint_yaw")
NEXT_POSITION = "识别完成，换下一个位置"
FINISHED = "识别完成"


def command_blocks_commit(view: dict, limit_deg: float = 10.0) -> bool:
    """单条达到 limit_deg 的俯仰或原地转角会让这一窗不能提交。小于该角度的指令不挡。"""
    for key in _MOVE_KEYS:
        move = view.get(key) or {}
        if not isinstance(move, dict) or move.get("direction") in (None, "none"):
            continue
        if abs(float(move.get("degrees", 0.0))) + 1e-6 >= float(limit_deg):
            return True
    return False


def _aligned_yaw(samples: list[dict]) -> np.ndarray:
    # 与融合同一套展开。无标记方块在 ±45° 两侧仍是同一个朝向，不能把标准差算大。
    yaw = np.array([float(sample["yaw"]) for sample in samples], dtype=float)
    if any(bool(sample.get("quarter")) for sample in samples):
        return _align_period(yaw, np.pi / 2.0)
    center = yaw[0]
    return np.arctan2(np.sin(yaw - center), np.cos(yaw - center))


@dataclass(frozen=True)
class TrialRecord:
    """一次已经提交的位姿。文本只含计分用的 x、y 和弧度 yaw。"""

    index: int
    x: float
    y: float
    yaw: float

    def text(self) -> str:
        return f"trial {self.index}\nx_m: {self.x:.3f}\ny_m: {self.y:.3f}\nyaw_rad: {self.yaw:.3f}\n"


class TrialSession:
    """收集、提交、等待挪动、结束。preview 在未提交时是当前稳定估计，提交后是冻结值。"""

    def __init__(
        self,
        trial_count: int = 3,
        min_trial_s: float = 15.0,
        stable_frames: int = 15,
        stable_xy_m: float = 0.010,
        stable_yaw_deg: float = 1.5,
        trial_move_m: float = 0.08,
        trial_move_frames: int = 6,
        trial_lost_s: float = 1.0,
        sample_jump_m: float = 0.03,
        motion_deg: float = 10.0,
    ):
        self.trial_count = int(trial_count)
        self.min_trial_s = float(min_trial_s)
        self.stable_frames = int(stable_frames)
        self.stable_xy_m = float(stable_xy_m)
        self.stable_yaw_deg = float(stable_yaw_deg)
        self.trial_move_m = float(trial_move_m)
        self.trial_move_frames = int(trial_move_frames)
        self.trial_lost_s = float(trial_lost_s)
        self.sample_jump_m = float(sample_jump_m)
        self.motion_deg = float(motion_deg)
        self.phase = "collect"
        self.status = ""
        self.observe_since: float | None = None
        self.samples: list[dict] = []
        self.completed = 0
        self.frozen: dict | None = None
        self._pending: dict | None = None
        self._committed_xy: np.ndarray | None = None
        self._away_frames = 0
        self._lost_since: float | None = None
        self._seen_long_loss = False

    @classmethod
    def from_config(cls, cfg: dict) -> TrialSession:
        return cls(
            trial_count=int(cfg.get("trial_count", 3)),
            min_trial_s=float(cfg.get("min_trial_s", 15)),
            stable_frames=int(cfg.get("stable_frames", 15)),
            stable_xy_m=float(cfg.get("stable_xy_m", 0.010)),
            stable_yaw_deg=float(cfg.get("stable_yaw_deg", 1.5)),
            trial_move_m=float(cfg.get("trial_move_m", 0.08)),
            trial_move_frames=int(cfg.get("trial_move_frames", 6)),
            trial_lost_s=float(cfg.get("trial_lost_s", 1.0)),
            sample_jump_m=float(cfg.get("sample_jump_m", 0.03)),
            motion_deg=float(cfg.get("commit_motion_deg", 10)),
        )

    def begin(self, now: float) -> None:
        self.observe_since = float(now)

    def drop_window(self) -> None:
        # 地面或外参跳了。这一窗作废，本次观察的 15 秒继续走。
        self.samples.clear()
        self._pending = None
        if self.phase == "collect":
            self.status = ""

    def preview(self) -> dict | None:
        if self.frozen is not None:
            return dict(self.frozen)
        if self._pending is not None:
            return dict(self._pending)
        return None

    def update(self, sample: dict | None, now: float, motion_blocked: bool) -> TrialRecord | None:
        """新鲜检测才传入 sample。回放的旧结果传 None。返回值只在新提交的那一帧非空。"""
        if self.phase == "done":
            return None
        if self.observe_since is None:
            self.observe_since = float(now)
        if self.phase == "wait" and not self._left_last_pose(sample, now):
            return None
        if self.phase == "wait":
            self._arm(now)
        if motion_blocked or sample is None:
            self.samples.clear()
            self._pending = None
            if self.phase == "collect":
                self.status = ""
            return None
        if self.samples:
            previous = self.samples[-1]
            jumped = np.hypot(float(sample["x"]) - float(previous["x"]), float(sample["y"]) - float(previous["y"]))
            if float(jumped) > self.sample_jump_m:
                self.samples.clear()
        self.samples.append(sample)
        self.samples[:] = self.samples[-self.stable_frames :]
        fused = self._ready()
        if fused is None:
            self._pending = None
            self.status = ""
            return None
        self._pending = {"x": fused.x, "y": fused.y, "yaw": fused.yaw}
        elapsed = float(now) - float(self.observe_since)
        if elapsed < self.min_trial_s:
            remain = self.min_trial_s - elapsed
            self.status = f"还需 {max(1, int(np.ceil(remain - 1e-9)))} s"
            return None
        return self._commit(fused)

    def _left_last_pose(self, sample: dict | None, now: float) -> bool:
        # 离开 8 cm 并持续若干帧，或丢失超过 1 秒后再出现，才开始下一次。
        if sample is None:
            if self._lost_since is None:
                self._lost_since = float(now)
            if float(now) - self._lost_since >= self.trial_lost_s:
                self._seen_long_loss = True
            return False
        if self._lost_since is not None and float(now) - self._lost_since >= self.trial_lost_s:
            self._seen_long_loss = True
        if self._seen_long_loss:
            return True
        self._lost_since = None
        assert self._committed_xy is not None
        distance = float(np.hypot(float(sample["x"]) - self._committed_xy[0], float(sample["y"]) - self._committed_xy[1]))
        if distance > self.trial_move_m:
            self._away_frames += 1
        else:
            self._away_frames = 0
        return self._away_frames >= self.trial_move_frames

    def _arm(self, now: float) -> None:
        self.phase = "collect"
        self.observe_since = float(now)
        self.samples.clear()
        self._pending = None
        self.frozen = None
        self.status = ""
        self._away_frames = 0
        self._lost_since = None
        self._seen_long_loss = False

    def _ready(self):
        if len(self.samples) < self.stable_frames:
            return None
        window = self.samples[-self.stable_frames :]
        xy = np.array([[sample["x"], sample["y"]] for sample in window], dtype=float)
        center = np.median(xy, axis=0)
        if float(np.max(np.abs(xy - center))) > self.stable_xy_m:
            return None
        if float(np.degrees(np.std(_aligned_yaw(window)))) > self.stable_yaw_deg:
            return None
        return fuse_poses(window)

    def _commit(self, fused) -> TrialRecord:
        self.completed += 1
        record = TrialRecord(self.completed, float(fused.x), float(fused.y), float(fused.yaw))
        self.samples.clear()
        self._pending = None
        self.frozen = {"x": record.x, "y": record.y, "yaw": record.yaw}
        self._committed_xy = np.array([record.x, record.y], dtype=float)
        self._away_frames = 0
        self._lost_since = None
        self._seen_long_loss = False
        if self.completed >= self.trial_count:
            self.phase = "done"
            self.status = FINISHED
        else:
            self.phase = "wait"
            self.status = NEXT_POSITION
        return record


def open_trial_file(path: str | Path):
    """每次启动清空，避免上一次的三次结果留在文件里。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "w", encoding="utf-8")
    handle.write("")
    handle.flush()
    return handle


def append_trial(handle, record: TrialRecord) -> str:
    text = record.text()
    if record.index > 1:
        handle.write("\n")
    handle.write(text)
    handle.flush()
    return text
