import cv2
import numpy as np
import main
from camera.source import Frame
from sim.scene import intrinsics_for


def test_headless_processes_frame_without_gui_and_closes(monkeypatch, tmp_path):
    reads, closed = [], []
    class Source:
        def __init__(self, *args, **kwargs): pass
        def read(self):
            if reads:
                raise KeyboardInterrupt
            reads.append(True)
            return Frame(np.zeros((48,64,3), np.uint8), np.full((48,64), np.nan, np.float32),
                         np.array([0.,1.,0.]), [])
        def close(self): closed.append(True)
    def forbidden(*args, **kwargs):
        raise AssertionError('SSH无窗口模式不允许绘图或窗口调用')
    monkeypatch.setattr(main, 'RealSenseSource', Source)
    for name in ('imshow','waitKey','namedWindow','destroyAllWindows'):
        monkeypatch.setattr(cv2, name, forbidden)
    for name in ('render_color_standard','show_color_standard','_show_test_view'):
        monkeypatch.setattr(main, name, forbidden)
    cfg = main.load_config('config/default.yaml')
    cfg['result_path'] = str(tmp_path / 'results.txt')
    main.run_live(cfg, intrinsics_for(64,48), 'basic', False, False, False, headless=True)
    assert reads and closed == [True]


def test_cli_passes_headless_option(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(sys, 'argv', ['main.py', '--headless'])
    monkeypatch.setattr(main, 'live_intrinsics', lambda serial: ('camera', SimpleNamespace()))
    calls = []
    monkeypatch.setattr(main, 'run_live', lambda *a, **k: calls.append(k))
    main.main()
    assert calls[0]['headless'] is True
