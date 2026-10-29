# 一次摆放要观察满最短时间，并且窗口稳定，才写下 x、y、yaw。

import numpy as np
import pytest

from fuse.trial import (
    FINISHED,
    NEXT_POSITION,
    TrialSession,
    append_trial,
    command_blocks_commit,
    open_trial_file,
)


def _pose(x=0.2, y=0.5, yaw=0.1):
    return {"x": x, "y": y, "z": 0.1, "yaw": yaw, "quarter": True}


def _fill(session, sample, now, frames=15, blocked=False):
    record = None
    for _ in range(frames):
        got = session.update(sample, now, blocked)
        if got is not None:
            record = got
    return record


def test_stable_pose_waits_for_minimum_time():
    """15 帧已经稳定时，未满 15 秒不提交，画面只提示剩余时间。满 15 秒才写下第一次。"""
    session = TrialSession(min_trial_s=15, stable_frames=15)
    session.begin(0.0)
    assert _fill(session, _pose(), 1.0) is None
    assert session.status == "还需 14 s"
    assert session.preview()["x"] == 0.2
    record = session.update(_pose(), 15.0, False)
    assert record is not None
    assert record.index == 1
    assert record.text() == "trial 1\nx_m: 0.200\ny_m: 0.500\nyaw_rad: 0.100\n"
    assert session.status == NEXT_POSITION
    assert session.preview()["yaw"] == record.yaw


def test_spread_and_jump_and_motion_block_commit():
    """位置散开、相邻帧跳过 3 cm，或有不小于 10° 的指令时，到时也不提交。"""
    loose = TrialSession(min_trial_s=15, stable_frames=15)
    loose.begin(0.0)
    for index in range(15):
        offset = 0.012 if index % 2 else 0.0
        assert loose.update(_pose(x=0.2 + offset), 20.0, False) is None
    assert loose.status == ""

    jumped = TrialSession(min_trial_s=15, stable_frames=15)
    jumped.begin(0.0)
    assert _fill(jumped, _pose(), 20.0, frames=14) is None
    assert jumped.update(_pose(x=0.25), 20.0, False) is None
    assert jumped.update(_pose(x=0.25), 20.0, False) is None

    blocked = TrialSession(min_trial_s=15, stable_frames=15)
    blocked.begin(0.0)
    assert _fill(blocked, _pose(), 20.0, frames=14) is None
    assert blocked.update(_pose(), 20.0, True) is None
    assert blocked.status == ""
    assert _fill(blocked, _pose(), 21.0, frames=14) is None
    assert blocked.update(_pose(), 21.0, False) is not None


def test_commands_below_ten_degrees_do_not_block():
    """小于 10° 的指令不执行，也不挡住提交。达到 10° 的挡住。"""
    assert command_blocks_commit({"yaw": {"direction": "right", "degrees": 32.6}})
    assert command_blocks_commit({"pitch": {"direction": "down", "degrees": 10.0}})
    assert not command_blocks_commit({"pitch": {"direction": "down", "degrees": 4.9}})
    assert not command_blocks_commit({"seek_yaw": {"direction": "left", "degrees": 2.5}})
    assert not command_blocks_commit({})


def test_quarter_yaw_near_fold_is_stable():
    """±45° 两侧是同一个朝向。交替出现时仍然可以提交。"""
    session = TrialSession(min_trial_s=15, stable_frames=15)
    session.begin(0.0)
    record = None
    for index in range(15):
        yaw = 0.78 if index % 2 else -0.76
        record = session.update(_pose(yaw=yaw), 20.0, False)
    assert record is not None
    assert abs(record.yaw) < np.deg2rad(46)


def test_same_place_does_not_write_again_until_three_moves():
    """提交后原地抖动不产生下一次。离开 8 cm 满 6 帧并再观察 15 秒，才写下一次。第三次后停止。"""
    session = TrialSession(min_trial_s=15, stable_frames=15, trial_count=3)
    session.begin(0.0)
    places = (0.2, 0.4, 0.7, 1.0)
    first = _fill(session, _pose(x=places[0]), 15.0)
    assert first.index == 1
    assert _fill(session, _pose(x=0.22), 30.0, frames=20) is None
    assert session.phase == "wait"

    written = [first]
    for place in places[1:3]:
        for step in range(6):
            session.update(_pose(x=place), 40.0 + step, False)
        assert session.phase == "collect"
        started = session.observe_since
        assert _fill(session, _pose(x=place), started + 14.0, frames=20) is None
        assert session.status.startswith("还需")
        written.append(_fill(session, _pose(x=place), started + 15.0, frames=20))
    assert [item.index for item in written] == [1, 2, 3]
    assert session.status == FINISHED
    for step in range(6):
        assert session.update(_pose(x=places[3]), 100.0 + step, False) is None
    assert _fill(session, _pose(x=places[3]), 200.0) is None
    assert session.completed == 3


def test_long_loss_starts_the_next_clock_even_at_the_same_place():
    """丢失满 1 秒后再出现，即使还在原地，也作为下一次并重新计 15 秒。短暂丢失不换次。"""
    session = TrialSession(min_trial_s=15, stable_frames=15)
    session.begin(0.0)
    assert _fill(session, _pose(), 15.0).index == 1
    session.update(None, 16.0, False)
    session.update(_pose(), 16.4, False)
    assert session.phase == "wait"
    session.update(None, 17.0, False)
    session.update(None, 18.0, False)
    assert session.phase == "wait"
    assert session.update(_pose(), 18.1, False) is None
    assert session.phase == "collect"
    assert session.frozen is None
    assert _fill(session, _pose(), 18.1 + 10.0) is None
    assert session.status.startswith("还需")
    second = _fill(session, _pose(), 18.1 + 15.0)
    assert second.index == 2


def test_drop_window_does_not_restart_the_clock():
    """地面跳变清空稳定窗，已经走过的观察时间保留。"""
    session = TrialSession(min_trial_s=15, stable_frames=15)
    session.begin(0.0)
    assert _fill(session, _pose(), 10.0) is None
    assert session.status.startswith("还需")
    session.drop_window()
    assert session.status == ""
    assert _fill(session, _pose(), 10.0) is None
    assert session.status.startswith("还需")
    assert session.update(_pose(), 15.0, False) is not None


def test_notice_is_drawn_on_the_preview():
    """测试画面底部能写出“识别完成，换下一个位置”，不是空字。"""
    from view_cube_center import paint_notice

    canvas = np.zeros((120, 1280, 3), dtype=np.uint8)
    image = canvas[:, :640]
    paint_notice(image, "识别完成，换下一个位置")
    if int(canvas.sum()) == 0:
        pytest.skip("系统没有可用的中文字体")
    assert int(canvas.sum()) > 1000


def test_result_file_keeps_only_scored_fields(tmp_path):
    """文件里每次只有 x、y 和弧度 yaw，两次之间空一行。"""
    session = TrialSession(min_trial_s=15, stable_frames=15, trial_count=2)
    session.begin(0.0)
    first = _fill(session, _pose(), 15.0)
    for step in range(6):
        session.update(_pose(x=0.5), 16.0 + step, False)
    second = _fill(session, _pose(x=0.5), session.observe_since + 15.0)
    handle = open_trial_file(tmp_path / "cube_trials.txt")
    append_trial(handle, first)
    append_trial(handle, second)
    handle.close()
    text = (tmp_path / "cube_trials.txt").read_text(encoding="utf-8")
    assert text == (
        "trial 1\n"
        "x_m: 0.200\n"
        "y_m: 0.500\n"
        "yaw_rad: 0.100\n"
        "\n"
        "trial 2\n"
        "x_m: 0.500\n"
        "y_m: 0.500\n"
        "yaw_rad: 0.100\n"
    )
