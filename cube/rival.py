"""绿线在 2 到 3 个物体之间跳时，按地面中心把方块留住。

每一帧最多 3 个已经通过足迹的候选。地面中心相对稳定位置跳过 15 cm，
这一帧就不把跳过去的那一块送进中值。稳定位置附近还有候选时留它。
稳定块连续缺席 3 帧，才承认绿线已经换到另一件物体上。
相机或地面自己跳了 30 mm 时由调用方 reset，不在这里当成杂物。
"""

from __future__ import annotations

import numpy as np


JUMP_M = 0.15
MAX_OBJECTS = 3
MISS_FRAMES = 3


def ground_xy(sample: dict) -> np.ndarray:
    return np.array([float(sample["x"]), float(sample["y"])], dtype=float)


def shape_cost(sample: dict, cube_size: float) -> tuple[int, float, float]:
    # 短边太窄的线材排最后。长边和高度峰值更接近边长、足迹更满的排前面。
    short = float(sample.get("short_side", 0.0))
    long_side = float(sample.get("long_side", cube_size))
    peak = float(sample.get("peak", cube_size))
    fraction = float(sample.get("footprint_fraction", 0.0))
    thin = 0 if short >= 0.45 * cube_size else 1
    return (thin, abs(long_side - cube_size) + abs(peak - cube_size), -fraction)


def _cluster(candidates: list[dict], jump_m: float, cube_size: float) -> list[dict]:
    # 15 cm 以内当成同一件物体，只留形状更好的那一个。最多 3 件。
    ordered = sorted(candidates, key=lambda item: shape_cost(item, cube_size))
    kept: list[dict] = []
    for item in ordered:
        xy = ground_xy(item)
        if any(float(np.linalg.norm(xy - ground_xy(other))) <= jump_m for other in kept):
            continue
        kept.append(item)
        if len(kept) >= MAX_OBJECTS:
            break
    return kept


class RivalPicker:
    """记住上一块稳定方块的地面中心，绿线跳到旁边的物体上时不跟着走。"""

    def __init__(self, jump_m: float = JUMP_M, cube_size: float = 0.2):
        self.jump_m = float(jump_m)
        self.cube_size = float(cube_size)
        self.stable_xy: np.ndarray | None = None
        self.misses = 0
        self.holding = False

    def reset(self) -> None:
        self.stable_xy = None
        self.misses = 0
        self.holding = False

    def choose(self, candidates: list[dict]) -> dict | None:
        self.holding = False
        unique = _cluster(candidates, self.jump_m, self.cube_size)
        if not unique:
            return None
        if self.stable_xy is None:
            chosen = min(unique, key=lambda item: shape_cost(item, self.cube_size))
            self._take(chosen)
            return chosen
        near = [
            item
            for item in unique
            if float(np.linalg.norm(ground_xy(item) - self.stable_xy)) <= self.jump_m
        ]
        if near:
            chosen = min(
                near,
                key=lambda item: (
                    float(np.linalg.norm(ground_xy(item) - self.stable_xy)),
                    *shape_cost(item, self.cube_size),
                ),
            )
            self._take(chosen)
            return chosen
        self.misses += 1
        if self.misses < MISS_FRAMES:
            # 绿线跳走了，但还没连续缺席够久。调用方沿用上一块方块，不要据此去转相机。
            self.holding = True
            return None
        chosen = min(unique, key=lambda item: shape_cost(item, self.cube_size))
        self._take(chosen)
        return chosen

    def _take(self, chosen: dict) -> None:
        self.stable_xy = ground_xy(chosen)
        self.misses = 0
