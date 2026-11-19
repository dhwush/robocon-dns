"""只读复现当前反馈冲突：从仓库根目录运行本脚本，不连接相机。"""
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from angle_feedback import RelativeStop
from fuse.trial import command_blocks_commit


def gravity(degrees):
    angle = math.radians(degrees)
    return np.array([0., math.cos(angle), math.sin(angle)])


def view(key, degrees, reason="unrecognized"):
    axis = "pitch" if key.endswith("pitch") else "yaw"
    direction = (("down" if degrees > 0 else "up") if axis == "pitch"
                 else ("right" if degrees > 0 else "left"))
    return dict(visible=True, ready=False, reason=reason, gravity_assumed=False,
                **{key: dict(direction=direction, degrees=abs(degrees),
                             signed_radians=math.radians(degrees))})


results = {}
feedback = RelativeStop(clock=lambda: 0.)
results["boundary_chatter"] = [
    feedback.apply(view("seek_pitch", -5), gravity(angle))["motion_status"]
    for angle in [0, -2.1, -1.6, -2.2]
]

feedback = RelativeStop(clock=lambda: 0.)
feedback.apply(view("seek_pitch", -5), gravity(0))
results["target_disappears"] = feedback.apply(
    dict(visible=False, ready=True, reason="no_cube"), gravity(0))

feedback = RelativeStop(clock=lambda: 0.)
feedback.apply(view("yaw", 15, "partial"), gravity(0),
               [(0., np.zeros(3)), (.1, np.zeros(3))])
results["planner_left20_feedback_right15"] = feedback.apply(
    view("yaw", -20, "partial"), gravity(0), [(.2, np.zeros(3))])

feedback = RelativeStop(clock=lambda: 0.)
mixed = view("seek_pitch", -10)
mixed["seek_yaw"] = view("seek_yaw", 10)["seek_yaw"]
results["mixed_missing_feedback"] = feedback.apply(mixed, gravity(0))

feedback = RelativeStop(clock=lambda: 0.)
output = feedback.apply(view("seek_pitch", -5), gravity(45))
results["sub10_command_blocks_commit"] = dict(
    status=output["motion_status"], blocks_commit=command_blocks_commit(output))

print(json.dumps(results, ensure_ascii=False, indent=2))
