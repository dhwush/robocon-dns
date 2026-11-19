"""录屏回归：红地板误瞄准、低帧率陀螺仪漏积累、完成提示被覆盖。"""
from collections import deque
from types import SimpleNamespace
import math
import threading
import numpy as np
import pytest
from camera.source import RealSenseSource
from cube.aim import plan_view
from angle_feedback import RelativeStop
from fuse.trial import TrialRecord, TrialSession
from main import completed_view, trial_result_line, paint_progress_notice
from geom.transform import Intrinsics

COLORS=[dict(name='red',hsv_low=[0,80,60],hsv_high=[10,255,255],front=False)]


def test_aim_uses_detected_cube_not_frame_filling_red_floor():
    color=np.full((240,320,3),(0,0,130),np.uint8)
    depth=np.ones((240,320),np.float32)
    mask=np.zeros((240,320),np.uint8)
    mask[90:170,110:190]=255
    intr=Intrinsics(320,240,300.,300.,160.,120.,0,0,0,0,0,'none')
    wrong=plan_view(color,depth,intr,COLORS,gravity=np.array([0.,1.,0.]))
    assert all(wrong.clipped.values())  # 原来拿到了铺满视场的红地板。
    correct=plan_view(color,depth,intr,COLORS,gravity=np.array([0.,1.,0.]),target_mask=mask)
    assert correct.reason=='centered' and correct.ready
    assert not any(correct.clipped.values())
    assert correct.pitch.direction==correct.yaw.direction=='none'


def test_visible_cube_cancels_obsolete_rotation_goal():
    feedback=RelativeStop(clock=lambda:0.)
    feedback.goals={'yaw':('yaw',math.radians(20),math.radians(20))}
    view=dict(visible=True,ready=True,reason='centered',gravity_assumed=False)
    output=feedback.apply(view,np.array([0.,1.,0.]))
    assert not feedback.goals
    assert not output['motion_active'] and '可以了' in output['motion_status']


def source_without_device():
    source=RealSenseSource.__new__(RealSenseSource)
    source._rs=SimpleNamespace(stream=SimpleNamespace(accel='accel'))
    source._video_ready=True
    source._lock=threading.Lock()
    source._event=threading.Event()
    source._gyro=deque(maxlen=400)
    source._pending=None
    source._accel=np.array([0.,1.,0.])
    source.depth_scale=.001
    source.align=SimpleNamespace(process=lambda frame:frame)
    source.temporal_rs=SimpleNamespace(process=lambda frame:frame)
    return source


def video_frame(stamp):
    depth=SimpleNamespace(get_data=lambda:np.ones((2,2),np.uint16)*1000)
    color=SimpleNamespace(timestamp=stamp*1000,get_data=lambda:np.zeros((2,2,3),np.uint8))
    frame=SimpleNamespace(is_frameset=lambda:True,get_depth_frame=lambda:depth,
                          get_color_frame=lambda:color,first_or_default=lambda stream:None)
    frame.as_frameset=lambda:frame
    return frame


def test_video_overwrite_keeps_all_unread_gyro_and_leaves_future_samples():
    source=source_without_device()
    for i in range(81):
        source._gyro.append((i*.005,np.array([0.,1.,0.])))
        if i%6==0:
            source._on_frame(video_frame(i*.005))
    # 消费者0.4秒才read一次；相机此前已覆盖了多帧图像。
    source._on_frame(video_frame(.395))
    frame=source.read()
    assert len(frame.gyro)==80
    assert frame.gyro[0][0]==0 and frame.gyro[-1][0]==pytest.approx(.395)
    assert len(source._gyro)==1 and source._gyro[0][0]==pytest.approx(.4)
    source._on_frame(video_frame(.4))
    assert source.read().gyro[0][0]==pytest.approx(.4)
    assert not source._gyro


def test_two_fps_rotation_integrates_whole_interval_and_reaches_three_degrees():
    source=source_without_device()
    feedback=RelativeStop(clock=lambda:0.)
    g=np.array([0.,1.,0.])
    source._gyro.extend([(0.,np.zeros(3)),(.005,np.zeros(3))])
    source._on_frame(video_frame(.005))
    frame=source.read()
    feedback.apply({},g,frame.gyro)
    command=dict(visible=True,ready=False,gravity_assumed=False,
                 seek_yaw=dict(direction='right',degrees=10.,signed_radians=math.radians(10.)))
    feedback.apply(command,g,[])
    for i in range(2,102):
        stamp=i*.005
        source._gyro.append((stamp,g*math.radians(14.)))
        if i%6==0:
            source._on_frame(video_frame(stamp))
    source._on_frame(video_frame(.505))
    frame=source.read()
    result=feedback.apply(command,g,frame.gyro)
    assert math.degrees(feedback.yaw)==pytest.approx(7.)
    assert result['motion_arrived'] and '可以了' in result['motion_status']


def test_completion_is_explicit_and_stops_further_aiming():
    session=TrialSession()
    session.phase='wait'
    session.status='识别完成，换下一个位置'
    view=completed_view(session)
    assert not view['motion_active'] and '识别完成' in view['motion_status']
    text=trial_result_line(TrialRecord(1,-.145,.331,-.137),session.status)
    assert '识别完成' in text and 'X=-0.145' in text and 'Y=+0.331' in text and 'yaw=-7.8°' in text


def test_completion_and_stop_have_separate_visible_notice_lines(monkeypatch):
    calls=[]
    monkeypatch.setattr('view_cube_center.paint_notice',lambda image,text:calls.append((image.shape,text)))
    paint_progress_notice(np.zeros((360,1280,3),np.uint8),'可以了，停止','识别完成，换下一个位置')
    assert [item[1] for item in calls]==['可以了，停止','识别完成，换下一个位置']
    assert calls[0][0][0]>calls[1][0][0]


def test_live_completion_prints_coordinates_once_and_does_not_restart_rotation(monkeypatch,tmp_path,capsys):
    import main
    from camera.source import Frame
    from sim.scene import intrinsics_for
    reads=[]
    class Source:
        def __init__(self,*a,**k): pass
        def read(self):
            if len(reads)==5:
                raise KeyboardInterrupt
            reads.append(True)
            return Frame(np.full((48,64,3),(0,0,150),np.uint8),np.ones((48,64),np.float32),np.array([0.,1.,0.]),[])
        def close(self): pass
    mask=np.zeros((48,64),np.uint8)
    mask[15:35,22:44]=255
    def candidates(*args,**kwargs):
        return [dict(x=-.145,y=.331,z=.1,yaw=-.137,inliers=100,quarter=True,
                     corners_cam=None,plane_origin=np.zeros(3),mask=mask)]
    monkeypatch.setattr(main,'RealSenseSource',Source)
    monkeypatch.setattr(main,'_acquire_plane',lambda *a: __import__('ground.plane',fromlist=['Plane']).Plane(np.array([0.,-1.,0.]),-.675))
    monkeypatch.setattr(main,'reuse_stable_plane',lambda p,*a:(p,p,0))
    monkeypatch.setattr(main,'estimate_candidates',candidates)
    calls=[]
    original=main._view_for_frame
    def aim(*a,**k):
        calls.append(k.get('target_mask'))
        return original(*a,**k)
    monkeypatch.setattr(main,'_view_for_frame',aim)
    cfg=main.load_config('config/default.yaml')
    cfg.update(result_path=str(tmp_path/'trials.txt'),trial_count=1,min_trial_s=0.,stable_frames=1,min_samples=1)
    main.run_live(cfg,intrinsics_for(64,48),'basic',False,False,False,headless=True)
    output=capsys.readouterr().out
    assert output.count('trial 1')==1
    assert output.count('第1次识别完成')==1
    assert 'X=-0.145 m' in output and 'Y=+0.331 m' in output
    assert len(calls)==1 and calls[0] is mask  # 后四帧不再下发自动调整。
    assert 'trial 1' in (tmp_path/'trials.txt').read_text()
