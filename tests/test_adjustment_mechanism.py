"""真实冲突回归：视觉换向、到位抖动、反馈缺失、搜索超时及实际转角额度。"""
import math

import numpy as np
import pytest

from angle_feedback import KEYS, PitchStop, RelativeStop
from tests.test_angle_stop import gravity, move


def stopped(output):
    return all((output.get(key) or {}).get('direction') in (None, 'none') for key in KEYS)


@pytest.mark.parametrize('axis', ['pitch', 'yaw'])
def test_visual_instruction_follows_current_frame_even_if_imu_has_not_moved(axis):
    feedback = RelativeStop(clock=lambda: 0.)
    feedback.apply(move(axis, 15), gravity(0))
    output = feedback.apply(move(axis, -20), gravity(0))
    assert output[axis]['direction'] == ('up' if axis == 'pitch' else 'left')
    assert output[axis]['degrees'] == pytest.approx(20.)
    assert not feedback.goals


@pytest.mark.parametrize('key', ['pitch', 'seek_pitch', 'footprint_pitch'])
def test_target_disappears_cancels_old_goal(key):
    feedback = RelativeStop(clock=lambda: 0.)
    feedback.apply(move(key, -5), gravity(0))
    output = feedback.apply(dict(visible=False, ready=True, reason='no_cube'), gravity(0))
    assert stopped(output) and not feedback.goals
    assert '继续' not in output['motion_status']
    assert not output['motion_arrived'] and not output['ready']


def test_one_missing_axis_feedback_produces_one_consistent_stop():
    feedback = RelativeStop(clock=lambda: 0.)
    command = move('seek_pitch', -10)
    command['seek_yaw'] = move('seek_yaw', 10)['seek_yaw']
    output = feedback.apply(command, gravity(0))
    assert stopped(output) and output['motion_active']
    assert '反馈缺失' in output['motion_status']
    assert '继续' not in output['motion_status'] and '可以了' not in output['motion_status']
    assert not feedback.goals  # 恢复反馈后再建立目标，不能追着未确认的旧起点走。


@pytest.mark.parametrize('axis', ['pitch', 'yaw'])
def test_visual_deadzone_holds_through_small_jitter_and_resumes_at_four(axis):
    feedback = RelativeStop(clock=lambda: 0.)
    for error, arrived in [(5., False), (3., True), (3.4, True), (2.8, True), (4., False)]:
        output = feedback.apply(move(axis, error), gravity(0))
        assert output['motion_arrived'] == arrived
        assert stopped(output) == arrived


def test_hysteresis_survives_transition_between_partial_and_fills_frame():
    feedback = RelativeStop(clock=lambda: 0.)
    command = move('pitch', 0)
    command['pitch']['direction'] = 'none'
    command['reason'] = 'fills_frame'
    feedback.apply(command, gravity(0))
    command = move('pitch', 3.4)
    command['reason'] = 'partial'
    output = feedback.apply(command, gravity(0))
    assert stopped(output) and output['motion_arrived']


def test_absolute_pitch_stop_uses_same_hysteresis():
    feedback = PitchStop(clock=lambda: 0.)
    goal = math.radians(20)
    for error, arrived in [(5., False), (3., True), (3.4, True), (4., False)]:
        feedback.command(lambda: goal, goal-math.radians(error))
        assert feedback.arrived == arrived


def test_search_cannot_rebase_target_after_timeout_until_context_changes():
    now = [0.]
    feedback = RelativeStop(clock=lambda: now[0], timeout_s=2.)
    command = move('seek_pitch', 5)
    feedback.apply(command, gravity(0))
    now[0] = 2.
    output = feedback.apply(command, gravity(0))
    assert stopped(output) and '超时' in output['motion_status']
    now[0] = 3.
    output = feedback.apply(command, gravity(0))
    assert stopped(output) and '超时' in output['motion_status']
    output = feedback.apply(command, gravity(0), target_changed=True)
    assert output['motion_started'] and not stopped(output)


def test_search_mode_change_replaces_old_pitch_direction():
    feedback = RelativeStop(clock=lambda: 0.)
    command = move('seek_pitch', -10)
    command['reason'] = 'unrecognized'
    feedback.apply(command, gravity(0))
    command = move('seek_pitch', 5)
    command['reason'] = 'ground'
    output = feedback.apply(command, gravity(0))
    assert output['seek_pitch']['direction'] == 'down'
    assert output['seek_pitch']['degrees'] == pytest.approx(5.)


def test_inertial_gap_stops_search_instead_of_claiming_complete_integration():
    feedback = RelativeStop(clock=lambda: 0.)
    g = gravity(0)
    feedback.observe(g, [(0., np.zeros(3)), (.1, np.zeros(3))])
    command = move('seek_yaw', 10)
    feedback.apply(command, g)
    output = feedback.apply(command, g, [(1., np.zeros(3)), (1.1, np.zeros(3))])
    assert stopped(output) and not feedback.goals
    assert '中断' in output['motion_status'] and not output['motion_arrived']


def test_feedback_loss_revokes_an_already_active_search_goal():
    feedback = RelativeStop(clock=lambda: 0.)
    command = move('seek_pitch', 10)
    feedback.apply(command, gravity(0))
    assert feedback.goals
    output = feedback.apply(command, None)
    assert stopped(output) and not feedback.goals
    assert '反馈缺失' in output['motion_status']


def test_observation_keeps_actual_motion_separate_from_command_and_is_not_integrated_twice():
    feedback = RelativeStop(clock=lambda: 0.)
    g = gravity(0)
    feedback.observe(g, [(0., np.zeros(3)), (.1, np.zeros(3))])
    command = move('seek_yaw', 10)
    feedback.apply(command, g)
    observation = feedback.observe(g, [(.2, g*math.radians(20.))])
    assert observation['delta']['yaw'] == pytest.approx(math.radians(2.))
    output = feedback.apply(command, g, observation=observation)
    assert output['seek_yaw']['degrees'] == pytest.approx(8.)
    assert feedback.yaw == pytest.approx(math.radians(2.))


def test_live_search_budget_uses_measured_motion_and_headless_has_heartbeat(monkeypatch, tmp_path, capsys):
    import main
    from camera.source import Frame
    from sim.scene import intrinsics_for

    index = [-1]
    class Source:
        def __init__(self, *args, **kwargs): pass
        def read(self):
            index[0] += 1
            if index[0] == 7:
                raise KeyboardInterrupt
            # 前五帧不动，第六帧才实际下俯2°。
            g = gravity(2 if index[0] >= 5 else 0)
            return Frame(np.zeros((48, 64, 3), np.uint8), np.ones((48, 64), np.float32), g, [])
        def close(self): pass

    rooms = []
    def plan(*args, **kwargs):
        rooms.append(kwargs['pitch_room'])
        command = move('seek_pitch', 5)
        command['reason'] = 'unrecognized'
        return command

    monkeypatch.setattr(main, 'RealSenseSource', Source)
    monkeypatch.setattr(main.time, 'monotonic', lambda: 100.+index[0]*.6)
    monkeypatch.setattr(main, '_acquire_plane', lambda *a: None)
    monkeypatch.setattr(main, 'estimate_candidates', lambda *a, **k: [])
    monkeypatch.setattr(main, '_view_for_frame', plan)
    cfg = main.load_config('config/default.yaml')
    cfg['result_path'] = str(tmp_path/'trials.txt')
    main.run_live(cfg, intrinsics_for(64, 48), 'basic', False, False, False, headless=True)
    assert rooms[:5] == pytest.approx([math.radians(15.)]*5)
    assert rooms[5] == pytest.approx(math.radians(13.))
    output = capsys.readouterr().out
    assert output.count('俯仰向下继续调整，剩余5.0°') == 2
    assert '俯仰可以了' in output
