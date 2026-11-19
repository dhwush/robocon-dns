"""视觉实时对准与IMU搜索反馈；3°到位、4°恢复，故障时统一停止。"""
import math
import time
import numpy as np

TOL_DEG = 3.0
TOL_RAD = math.radians(TOL_DEG)
RESUME_RAD = math.radians(4.0)
STOP_TEXT = '可以了，停止！'
KEYS = ('pitch', 'yaw', 'seek_pitch', 'seek_yaw', 'footprint_pitch', 'footprint_yaw')


def within_angle(target, current):
    return math.isfinite(float(target)) and math.isfinite(float(current)) and abs(float(target)-float(current)) <= TOL_RAD + 1e-12


class PitchStop:
    """俯仰的绝对目标：到位后保持一秒，不让规划跳到下一角度掩盖停止提示。"""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.target = None
        self.ack = False
        self.until = None
        self.arrived = False
        self.message = ''

    def command(self, provider, current):
        now = self.clock()
        if self.target is not None and not self.ack and within_angle(self.target, current):
            self.ack, self.until = True, now + 1.
        if self.until is not None and now < self.until:
            target = self.target
        else:
            self.until = None
            target = float(provider())
            if self.target is None or abs(target-self.target) > 1e-9:
                self.target, self.ack = target, False
                self.arrived = False
            if not self.ack and within_angle(target, current):
                self.ack, self.until = True, now + 1.
        self.arrived = within_angle(target, current) or (
            self.arrived and math.isfinite(float(current))
            and abs(target-current) < RESUME_RAD-1e-12)
        self.message = STOP_TEXT + '俯仰已到位，保持当前角度' if self.arrived else '继续调整俯仰，进入目标±3°后停止'
        return target


class RelativeStop:
    """可见目标按本帧图像对准；搜索按IMU完成单步，失效时整机停止。"""
    def __init__(self, clock=None, timeout_s=10.0):
        self.clock = time.monotonic if clock is None else clock
        self.timeout_s = float(timeout_s)
        self.goals = {}
        self.yaw = 0.
        self.stamp = None
        self.yaw_valid = False
        self.yaw_received = None
        self.until = None
        self._pitch = None
        self._context = None
        self._stopped = {}
        self._active_since = None
        self._timed_out = False

    def cancel(self):
        """撤销动作和到位锁存，不重置惯性时间基准。"""
        self.goals.clear()
        self.until = None
        self._context = None
        self._stopped.clear()
        self._active_since = None
        self._timed_out = False

    def observe(self, gravity, gyro=None):
        """每帧读取惯性反馈，包括等待/完成期间；返回实测角度增量。"""
        now = self.clock()
        down = None
        if gravity is not None:
            g = np.asarray(gravity, dtype=float)
            if g.shape == (3,) and np.isfinite(g).all() and np.linalg.norm(g) > 1e-6:
                down = g / np.linalg.norm(g)
        pitch = None if down is None else math.asin(float(np.clip(down[2], -1., 1.)))
        pitch_delta = None if pitch is None or self._pitch is None else pitch-self._pitch
        self._pitch = pitch
        yaw_before = self.yaw
        integrated, gap = False, False
        for stamp, omega in gyro or []:
            stamp = float(stamp)
            if not math.isfinite(stamp) or (self.stamp is not None and stamp <= self.stamp):
                continue
            dt = None if self.stamp is None else stamp-self.stamp
            w = np.asarray(omega, dtype=float)
            if dt is not None and down is not None and 0 < dt <= .2 and w.shape == (3,) and np.isfinite(w).all():
                self.yaw += float(w @ down)*dt
                self.yaw_valid = True
                self.yaw_received = now
                integrated = True
            elif dt is not None:
                self.yaw_valid = False
                gap = True
            self.stamp = stamp
        yaw = self.yaw if self.yaw_valid and self.yaw_received is not None and now-self.yaw_received <= .3 else None
        return dict(current={'pitch': pitch, 'yaw': yaw},
                    delta={'pitch': pitch_delta, 'yaw': self.yaw-yaw_before if integrated and not gap else None},
                    gap=gap)

    @staticmethod
    def _signed(move):
        if move.get('direction') in (None, 'none'):
            return 0.
        return float(move.get('signed_radians', math.radians(float(move.get('degrees', 0.)))*
                     (-1 if move.get('direction') in ('up', 'left') else 1)))

    def _arrived(self, axis, error):
        # 进入死区立即到位；到位后达到4°才重新调整。
        arrived = abs(error) <= TOL_RAD + 1e-12 or (
            self._stopped.get(axis, False) and abs(error) < RESUME_RAD-1e-12)
        self._stopped[axis] = arrived
        return arrived

    def _finish(self, out, message, *, active=False, arrived=False, started=False):
        out.update(ready=arrived, motion_active=active, motion_arrived=arrived,
                   motion_started=started, motion_status=message)
        return out

    def apply(self, view, gravity, gyro=None, *, observation=None, target_changed=False):
        now = self.clock()
        observed = self.observe(gravity, gyro) if observation is None else observation
        current = observed['current']
        out = dict(view)
        self._clear(out)
        reason = view.get('reason')
        search_keys = [key for key in KEYS if key.startswith(('seek_', 'footprint_'))
                       and key in view]
        # reason是正式规划器的状态；无reason的调用仍可按动作字段区分。
        visual = (reason in ('partial', 'fills_frame') or
                  (reason is None and not search_keys)) and bool(view.get('visible'))
        context = ('visual' if visual else reason, bool(view.get('visible')))
        if target_changed or context != self._context:
            self.cancel()
            self._context = context
        if reason == 'centered' and view.get('visible') and not view.get('gravity_assumed'):
            self.cancel()
            return self._finish(out, STOP_TEXT+'保持当前角度', arrived=True)
        if self._timed_out:
            return self._finish(out, '停止！调整超时，请检查机构或重新放置目标', active=True)

        errors = {}
        started = False
        if visual and not view.get('gravity_assumed') and current['pitch'] is not None:
            # 不让旧的惯性目标覆盖当前目标的视觉方向。
            self.goals.clear()
            self.until = None
            for axis in ('pitch', 'yaw'):
                if axis in view:
                    errors[axis] = (axis, self._signed(view[axis]))
        elif search_keys and not view.get('gravity_assumed'):
            moves = {('pitch' if key.endswith('pitch') else 'yaw'): (key, self._signed(view[key]))
                     for key in search_keys if view[key].get('direction') not in (None, 'none')}
            if not moves:
                self.goals.clear()
                self.until = None
                self._active_since = None
                return self._finish(out, '停止，当前搜索已无可执行调整，请检查目标和搜索范围')
            if observed['gap']:
                self.goals.clear()
                self.until = None
                self._active_since = None
                return self._finish(out, '停止！角度反馈中断，重新确认后再调整', active=True)
            for axis in list(self.goals):
                if axis not in moves:
                    del self.goals[axis]
                    self._stopped.pop(axis, None)
            if self.until is not None and now >= self.until:
                self.goals.clear()
                self._stopped.clear()
                self.until = None
                self._active_since = None
            # 搜索规划不提供真实剩余角，必须用固定一步的IMU目标。
            if not self.goals:
                if any(current[axis] is None for axis in moves):
                    return self._finish(out, '停止！角度反馈缺失，暂不能确认到位', active=True)
                self.goals = {axis: (key, current[axis]+signed, signed)
                              for axis, (key, signed) in moves.items()}
                started = True
            if any(current[axis] is None for axis in self.goals):
                self.goals.clear()
                self._stopped.clear()
                self.until = None
                self._active_since = None
                return self._finish(out, '停止！角度反馈缺失，暂不能确认到位', active=True)
            for axis, (key, target, _) in self.goals.items():
                error = target-current[axis]
                if axis == 'yaw':
                    error = math.atan2(math.sin(error), math.cos(error))
                errors[axis] = (key, error)
        else:
            self.cancel()
            return self._finish(out, '停止，等待有效目标和角度反馈')

        if not errors or any(not math.isfinite(error) for _, error in errors.values()):
            self.cancel()
            return self._finish(out, '停止，等待有效目标和角度反馈', active=True)
        messages, pending = [], False
        for axis, (key, error) in errors.items():
            label = '俯仰' if axis == 'pitch' else '旋转'
            if self._arrived(axis, error):
                messages.append(label+STOP_TEXT)
            else:
                pending = True
                direction = (('down' if error > 0 else 'up') if axis == 'pitch'
                             else ('right' if error > 0 else 'left'))
                out[key] = dict(direction=direction, radians=abs(error),
                                degrees=math.degrees(abs(error)), signed_radians=error)
                messages.append(label+{'down':'向下', 'up':'向上', 'right':'向右', 'left':'向左'}[direction]+
                                '继续调整，剩余%.1f°' % math.degrees(abs(error)))
        if pending:
            self.until = None
            if self._active_since is None:
                self._active_since = now
            if now-self._active_since >= self.timeout_s:
                self._timed_out = True
                self.goals.clear()
                self._clear(out)
                return self._finish(out, '停止！调整超时，请检查机构或重新放置目标', active=True)
        else:
            self._active_since = None
            if self.goals and self.until is None:
                self.until = now+1.
        return self._finish(out, '；'.join(messages), active=pending, arrived=not pending, started=started)

    @staticmethod
    def _clear(view):
        for key in KEYS:
            if key in view:
                view[key] = dict(direction='none', radians=0., degrees=0., signed_radians=0.)
