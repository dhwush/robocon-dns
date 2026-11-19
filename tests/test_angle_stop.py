import math
import numpy as np
import pytest
from angle_feedback import within_angle, PitchStop, RelativeStop


def gravity(degrees):
    a=math.radians(degrees)
    return np.array([0., math.cos(a), math.sin(a)])


def move(key, degrees):
    axis='pitch' if key.endswith('pitch') else 'yaw'
    direction=('down' if degrees>0 else 'up') if axis=='pitch' else ('right' if degrees>0 else 'left')
    return {key: dict(direction=direction,degrees=abs(degrees),signed_radians=math.radians(degrees)),
            'visible':True,'ready':False,'gravity_assumed':False}


@pytest.mark.parametrize('error,arrived', [(2.99,True),(3.,True),(3.01,False),(-3.,True),(-3.01,False)])
def test_three_degree_inclusive_boundary(error,arrived):
    assert within_angle(math.radians(20),math.radians(20+error)) == arrived
    stop=PitchStop(clock=lambda:0.)
    stop.command(lambda: math.radians(20), math.radians(20+error))
    assert stop.arrived == arrived
    assert ('可以了' in stop.message) == arrived


def test_pitch_stop_keeps_arrived_goal_before_asking_next():
    now=[0.]
    stop=PitchStop(clock=lambda:now[0])
    goal=math.radians(20)
    assert stop.command(lambda:goal,0.) == goal
    now[0]=.1
    stop.command(lambda:goal,math.radians(17))
    now[0]=.5
    assert stop.command(lambda: pytest.fail('停止提示期间不能请求下一目标'),math.radians(18)) == goal
    now[0]=1.2
    assert stop.command(lambda:math.radians(40),math.radians(20)) == math.radians(40)
    assert not stop.arrived


@pytest.mark.parametrize('key', ['seek_pitch','footprint_pitch'])
def test_relative_pitch_target_is_fixed_until_three_degrees(key):
    now=[0.]
    stop=RelativeStop(clock=lambda:now[0])
    first=stop.apply(move(key,10),gravity(0))
    assert first['motion_active'] and first[key]['degrees']==pytest.approx(10)
    # 每帧又建议转10度，已发出的目标不能从10变成17。
    now[0]=.1
    arrived=stop.apply(move(key,10),gravity(7))
    assert arrived['motion_arrived'] and '可以了' in arrived['motion_status']
    assert arrived[key]['direction']=='none'
    now[0]=.2
    drifted=stop.apply(move(key,10),gravity(6))
    assert not drifted['motion_arrived']
    assert drifted[key]['degrees']==pytest.approx(4)


@pytest.mark.parametrize('key', ['seek_yaw','footprint_yaw'])
@pytest.mark.parametrize('sign', [-1,1])
def test_rotation_uses_gyro_feedback_and_stops_at_three_degrees(key,sign):
    now=[0.]
    stop=RelativeStop(clock=lambda:now[0])
    g=gravity(0)
    stop.apply({},g,[(1.,np.zeros(3)),(1.1,np.zeros(3))])
    first=stop.apply(move(key,10*sign),g,[(1.2,np.zeros(3))])
    assert first['motion_active']
    now[0]=.1
    arrived=stop.apply(move(key,10*sign),g,[(1.3,g*math.radians(70*sign))])
    assert arrived['motion_arrived'] and '可以了' in arrived['motion_status']
    assert arrived[key]['direction']=='none'


def test_missing_feedback_does_not_claim_arrival():
    stop=RelativeStop(clock=lambda:0.)
    out=stop.apply(move('seek_yaw',10),None,[])
    assert not out['motion_arrived']
    assert '反馈缺失' in out['motion_status'] and '可以了' not in out['motion_status']


def test_pending_small_motion_blocks_trial_commit_until_confirmed():
    from fuse.trial import command_blocks_commit
    assert command_blocks_commit({'motion_active': True, 'yaw': {'direction':'right','degrees':3.01}})
    assert not command_blocks_commit({'motion_active': False, 'motion_arrived': True})


@pytest.mark.parametrize('key', ['pitch', 'seek_pitch', 'footprint_pitch'])
@pytest.mark.parametrize('error', [-3.01, -3., -2.99, 2.99, 3., 3.01])
def test_relative_pitch_deadzone_includes_both_three_degree_boundaries(key, error):
    stop = RelativeStop(clock=lambda: 0.)
    stop.apply(move(key, 10), gravity(0))
    output = stop.apply(move(key, error if key == 'pitch' else 10), gravity(10-error))
    arrived = abs(error) <= 3.
    assert output['motion_arrived'] == arrived
    assert ('俯仰可以了' in output['motion_status']) == arrived
    assert (output[key]['direction'] == 'none') == arrived


@pytest.mark.parametrize('reason', ['ground', 'unrecognized', 'waiting'])
def test_no_pitch_command_without_alignment_does_not_claim_pitch_arrival(reason):
    output = RelativeStop(clock=lambda: 0.).apply(
        dict(visible=True, ready=True, reason=reason, gravity_assumed=False,
             pitch=dict(direction='none', degrees=0.)), gravity(0))
    assert '俯仰可以了' not in output['motion_status']
