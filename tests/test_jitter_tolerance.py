"""短时抖动可以收敛，真实换位、运动和陈旧数据不能冒充稳定结果。"""
import math

import numpy as np
import pytest

from fuse.trial import TrialSession
from ground.plane import PlaneChangeGuard


def pose(x=.2, y=.5, yaw=.1):
    return dict(x=x, y=y, z=.1, yaw=yaw, quarter=True)


def test_four_fps_with_one_miss_every_ten_frames_eventually_commits():
    session = TrialSession(trial_count=1)
    session.begin(0.)
    records = []
    for i in range(101):
        sample = None if i % 10 == 9 else pose()
        record = session.update(sample, i*.25, False)
        if record:
            assert i*.25 >= 15.
            records.append(record)
    assert len(records) == 1
    assert records[0].x == pytest.approx(.2)


def test_two_fps_can_collect_a_recent_stable_window():
    session = TrialSession(min_trial_s=0)
    for i in range(9):
        assert session.update(pose(), i*.5, False) is None
    assert session.update(pose(), 4.5, False) is not None


def test_two_short_misses_preserve_window_but_never_commit_on_missing_frame():
    session = TrialSession(min_trial_s=0)
    for i in range(9):
        assert session.update(pose(), i*.25, False) is None
    for t in [2.25, 2.5]:
        assert session.update(None, t, False) is None
        assert len(session.samples) == 9
        assert session.preview() is None
    assert session.update(pose(), 2.75, False) is not None


def test_third_consecutive_miss_clears_old_samples():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    for t in [.25, .5, .75]:
        assert session.update(None, t, False) is None
    assert not session.samples
    assert session.update(pose(), 1., False) is None


def test_long_gap_clears_even_without_explicit_missing_frame():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    assert session.update(pose(), 1.1, False) is None
    assert len(session.samples) == 1


def test_samples_older_than_six_seconds_cannot_pad_the_window():
    session = TrialSession(min_trial_s=0)
    for i in range(30):
        assert session.update(pose(), i*.8, False) is None
    assert len(session.samples) < 10


def test_relaxed_xy_and_yaw_jitter_can_converge():
    session = TrialSession(min_trial_s=0)
    record = None
    for i in range(10):
        sample = pose(x=.2 + (.025 if i % 2 else 0.), yaw=.1+math.radians(2 if i % 2 else -2))
        record = session.update(sample, i*.25, False)
    assert record is not None
    assert record.x == pytest.approx(.2125)


def test_two_outliers_are_excluded_from_final_pose():
    session = TrialSession(min_trial_s=0)
    record = None
    for i in range(10):
        sample = pose(x=.235) if i == 2 else pose(yaw=.6) if i == 6 else pose()
        record = session.update(sample, i*.25, False)
    assert record is not None
    assert (record.x, record.y, record.yaw) == pytest.approx((.2, .5, .1))


def test_three_outliers_do_not_pass_as_stable():
    session = TrialSession(min_trial_s=0)
    for i in range(10):
        assert session.update(pose(x=.235 if i in (2, 4, 6) else .2), i*.25, False) is None


def test_newest_outlier_does_not_publish_an_old_stable_window():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    assert session.update(pose(x=.235), .25, False) is None
    assert session.preview() is None
    assert session.update(pose(), .5, False) is not None


def test_isolated_large_jump_does_not_destroy_good_samples():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    assert session.update(pose(x=.5), .25, False) is None
    assert len(session.samples) == 9
    record = session.update(pose(), .5, False)
    assert record is not None and record.x == pytest.approx(.2)


def test_confirmed_new_position_discards_all_old_position_samples():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    for i in range(3):
        assert session.update(pose(x=.5), (i+1)*.25, False) is None
    assert len(session.samples) == 3
    assert all(sample['x'] == .5 for sample in session.samples)
    for i in range(6):
        assert session.update(pose(x=.5), 1.+i*.25, False) is None
    record = session.update(pose(x=.5), 2.5, False)
    assert record is not None and record.x == pytest.approx(.5)


def test_actual_motion_still_resets_the_window():
    session = TrialSession(min_trial_s=0)
    for _ in range(9):
        session.update(pose(), 0., False)
    assert session.update(pose(), .25, True) is None
    assert not session.samples
    assert session.update(pose(), .5, False) is None


def test_plane_guard_ignores_single_spikes_but_confirms_persistent_change():
    guard = PlaneChangeGuard()
    assert not guard.update(np.zeros(3))
    for _ in range(5):
        assert not guard.update(np.array([0., 0., .08]))
        assert not guard.update(np.zeros(3))
    assert not guard.update(np.array([0., 0., .08]))
    assert not guard.update(np.array([0., 0., .081]))
    assert guard.update(np.array([0., 0., .079]))
    assert not guard.update(np.array([0., 0., .08]))


def test_live_small_pitch_jitter_does_not_clear_the_window(monkeypatch, tmp_path, capsys):
    import main
    from camera.source import Frame
    from field.body import body_extrinsic
    from ground.plane import Plane
    from sim.scene import intrinsics_for

    def gravity(degrees):
        angle = math.radians(degrees)
        return np.array([0., math.cos(angle), math.sin(angle)])

    # 此抖动会使相机系t相邻变化超过旧3cm门限，车身系目标与地面却没移动。
    _, t0 = body_extrinsic(gravity(45))
    _, t1 = body_extrinsic(gravity(47.6))
    assert np.linalg.norm(t1-t0) > .03
    reads = []
    class Source:
        def __init__(self, *args, **kwargs): pass
        def read(self):
            if len(reads) >= 12:
                raise KeyboardInterrupt
            g = gravity(45 if len(reads) % 2 else 47.6)
            reads.append(g)
            return Frame(np.zeros((48, 64, 3), np.uint8), np.ones((48, 64), np.float32), g, [])
        def close(self): pass

    mask = np.zeros((48, 64), np.uint8)
    mask[15:35, 22:44] = 255
    monkeypatch.setattr(main, 'RealSenseSource', Source)
    monkeypatch.setattr(main, '_acquire_plane', lambda *a: Plane(-reads[-1], -.675))
    monkeypatch.setattr(main, 'reuse_stable_plane', lambda p, *a: (p, p, 0))
    monkeypatch.setattr(main, 'estimate_candidates', lambda *a, **k: [dict(
        **pose(), inliers=100, plane_origin=np.zeros(3), corners_cam=None, mask=mask)])
    monkeypatch.setattr(main, '_view_for_frame', lambda *a, **k: dict(
        visible=True, ready=True, reason='centered', gravity_assumed=False))
    cfg = main.load_config('config/default.yaml')
    cfg.update(min_trial_s=0., trial_count=1, result_path=str(tmp_path/'trials.txt'))
    main.run_live(cfg, intrinsics_for(64, 48), 'basic', False, False, False, headless=True)
    output = capsys.readouterr().out
    assert output.count('第1次识别完成') == 1
    assert (tmp_path/'trials.txt').read_text().count('trial 1') == 1
