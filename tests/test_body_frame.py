"""俯仰前后，当前车身右前点的坐标与yaw保持一致。"""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from field.body import body_extrinsic, DEFAULT_OPTICAL_CENTER
from geom.transform import camera_to_world, world_to_camera
from main import pose_to_world
from ground.plane import Plane


def expected(pitch, lever=None):
    c, s = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[1., 0., 0.], [0., s, -c], [0., c, s]])
    level = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    optical = DEFAULT_OPTICAL_CENTER.copy()
    if lever is not None:
        optical += (rotation.T @ level - np.eye(3)) @ lever
    return rotation, -rotation @ optical


@pytest.mark.parametrize('degrees', [-55., -20., 0., 15., 40.])
@pytest.mark.parametrize('lever', [np.zeros(3), np.array([0.02, 0.07, 0.03])])
def test_same_body_point_and_heading_survive_camera_pitch(degrees, lever):
    pitch = math.radians(degrees)
    r, t = expected(pitch, lever)
    gravity = -r[:, 2]
    measured_r, measured_t = body_extrinsic(gravity, {'camera_pivot_to_optical_m': lever})
    points = np.array([[0., 0., 0.], [.2, .6, .1], [-.25, .75, .8]])
    camera = world_to_camera(points, r, t)
    recovered = camera_to_world(camera, measured_r, measured_t)
    np.testing.assert_allclose(recovered, points, atol=1e-12)
    edge_body = np.array([math.cos(.31), math.sin(.31), 0.])
    edge_recovered = measured_r.T @ (r @ edge_body)
    assert math.atan2(edge_recovered[1], edge_recovered[0]) == pytest.approx(.31)
    np.testing.assert_allclose(measured_r.T @ measured_r, np.eye(3), atol=1e-12)


def test_plane_fallback_matches_imu_and_absence_is_rejected():
    r, t = expected(-.4)
    plane = Plane(r[:, 2], float(r[:, 2] @ t))
    imu = body_extrinsic(-plane.normal)
    fallback = body_extrinsic(None, plane=plane)
    np.testing.assert_allclose(imu[0], fallback[0])
    np.testing.assert_allclose(imu[1], fallback[1])
    for bad in (None, [0.,0.,0.], [float('nan'),1.,0.]):
        with pytest.raises(ValueError):
            body_extrinsic(bad)


def test_pose_center_uses_body_frame_after_pitch():
    r, t = expected(-.5)
    measured_r, measured_t = body_extrinsic(-r[:, 2])
    height = .2
    bottom_body = np.array([.2, .65, 0.])
    pose = SimpleNamespace(bottom_center=world_to_camera(bottom_body, r, t), yaw=.2,
                           inliers=100, front_color=False, corners=None,
                           layers=None, top_color=None, pad_corners=None)
    plane = Plane(r[:, 2], float(r[:, 2] @ t))
    # TASK_LIFT
    result = pose_to_world(pose, measured_r, measured_t, plane, height)
    np.testing.assert_allclose([result['x'],result['y'],result['z']], [.2,.65,.1], atol=1e-12)

@pytest.mark.parametrize('mode', ['basic', 'field'])
def test_live_modes_use_current_body_transform_without_tracking(monkeypatch, tmp_path, mode):
    import main
    from camera.source import Frame
    from sim.scene import intrinsics_for
    current = {}
    seen = []
    target = np.array([.2, .65, .1])
    class Source:
        index = 0
        closed = False
        def __init__(self, *a, **k):
            current['source'] = self
        def read(self):
            if self.index >= 3:
                raise KeyboardInterrupt
            r, t = expected(math.radians([-35., 0., 25.][self.index]))
            self.index += 1
            current['r'], current['t'] = r, t
            return Frame(np.zeros((48,64,3), np.uint8), np.ones((48,64), np.float32), -r[:,2], [])
        def close(self):
            self.closed = True
    monkeypatch.setattr(main, 'RealSenseSource', Source)
    monkeypatch.setattr(main, 'render_color_standard', lambda *a, **k: None)
    monkeypatch.setattr(main, 'show_color_standard', lambda *a, **k: None)
    def plane_now(*args, **kwargs):
        r, t = current['r'], current['t']
        return Plane(r[:,2], float(r[:,2] @ t))
    def sample(r, t):
        recovered = camera_to_world(world_to_camera(target, current['r'], current['t']), r, t)
        seen.append(recovered)
        return dict(x=float(recovered[0]),y=float(recovered[1]),z=float(recovered[2]),yaw=.2,
                    inliers=100,quarter=True,corners_cam=None,plane_origin=np.zeros(3))
    cfg = main.load_config('config/default.yaml')
    cfg['min_samples'] = 1
    cfg['result_path'] = str(tmp_path/'results.txt')
    monkeypatch.setattr(main, '_acquire_plane', plane_now)
    monkeypatch.setattr(main, 'reuse_stable_plane', lambda p,*a: (p,p,0))
    monkeypatch.setattr(main, 'estimate_candidates', lambda frame,intr,cfg,r,t,*a,**k: [sample(r,t)])
    monkeypatch.setattr(main, '_view_for_frame', lambda *a,**k: {})
    main.run_live(cfg, intrinsics_for(64,48), mode, False, False, False, headless=True)
    assert current['source'].closed
    assert len(seen) == 3
    np.testing.assert_allclose(seen, np.tile(target, (3,1)), atol=1e-12)
