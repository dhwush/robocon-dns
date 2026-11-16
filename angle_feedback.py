"""角度指令使用实际反馈判定；±3度以内先明确停车，再允许下一动作。"""
import math
import time
import numpy as np

TOL_DEG = 3.0
TOL_RAD = math.radians(TOL_DEG)
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
            if not self.ack and within_angle(target, current):
                self.ack, self.until = True, now + 1.
        self.arrived = within_angle(target, current)
        self.message = STOP_TEXT + '俯仰已到位（误差不超过3°）' if self.arrived else '继续调整俯仰，进入目标±3°后停止'
        return target


class RelativeStop:
    """任务一相对转角固定为一条目标；IMU判定到位后，停一秒再接受新目标。"""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.goals = {}
        self.yaw = 0.
        self.stamp = None
        self.yaw_valid = False
        self.yaw_received = None
        self.until = None

    def cancel(self):
        """测量已完成或当前识别证明无需调整，撤销旧目标，不重置惯性时间基准。"""
        self.goals.clear()
        self.until = None

    def apply(self, view, gravity, gyro=None):
        now = self.clock()
        out = dict(view)
        down = None
        if gravity is not None:
            g = np.asarray(gravity, dtype=float)
            if g.shape == (3,) and np.isfinite(g).all() and np.linalg.norm(g) > 1e-6:
                down = g / np.linalg.norm(g)
        pitch = None if down is None else math.asin(float(np.clip(down[2], -1., 1.)))
        for stamp, omega in gyro or []:
            stamp = float(stamp)
            if self.stamp is not None and stamp <= self.stamp:
                continue
            dt = None if self.stamp is None else stamp-self.stamp
            if dt is not None and down is not None and 0 < dt <= .2:
                w = np.asarray(omega, dtype=float)
                if w.shape == (3,) and np.isfinite(w).all():
                    self.yaw += float(w @ down)*dt
                    self.yaw_valid = True
                    self.yaw_received = now
            elif dt is not None:
                self.yaw_valid = False
            self.stamp = stamp
        current = {'pitch': pitch, 'yaw': self.yaw if self.yaw_valid and self.yaw_received is not None and now-self.yaw_received <= .3 else None}
        out['motion_started'] = False
        if view.get('visible') and view.get('ready') and view.get('reason') == 'centered':
            self.cancel()
        if self.until is not None:
            if now < self.until:
                still_arrived = all(target is not None and current[axis] is not None and
                    abs(math.atan2(math.sin(target-current[axis]), math.cos(target-current[axis]))) <= TOL_RAD + 1e-12
                    for axis, (_, target, _) in self.goals.items())
                if still_arrived:
                    self._clear(out)
                    out.update(ready=True, motion_active=False, motion_arrived=True, motion_status=STOP_TEXT+'保持车身和相机不动')
                    return out
            else:
                self.goals.clear()
            self.until = None
        if not self.goals:
            for key in KEYS:
                move = view.get(key) or {}
                if move.get('direction') in (None, 'none'):
                    continue
                axis = 'pitch' if key.endswith('pitch') else 'yaw'
                if axis in self.goals:
                    continue
                signed = float(move.get('signed_radians', math.radians(float(move.get('degrees', 0.)))*(-1 if move.get('direction') in ('up','left') else 1)))
                self.goals[axis] = (key, None if current[axis] is None else current[axis]+signed, signed)
            out['motion_started'] = bool(self.goals)
        self._clear(out)
        messages, pending, unknown = [], False, False
        for axis, (key, target, signed) in self.goals.items():
            value = current[axis]
            if target is None and value is not None:
                target = value + signed
                self.goals[axis] = (key, target, signed)
            # 普通对准可以用同一画面的剩余角；搜索动作没有目标图像，必须依靠IMU。
            if target is not None and value is not None:
                error = target-value
                if axis == 'yaw':
                    error = math.atan2(math.sin(error), math.cos(error))
            elif key in ('pitch','yaw') and view.get('visible') and not view.get('gravity_assumed'):
                error = float((view.get(key) or {}).get('signed_radians', 0.))
            else:
                unknown = True
                messages.append('停止！角度反馈缺失，暂不能确认到位')
                continue
            label = '俯仰' if axis == 'pitch' else '旋转'
            if abs(error) <= TOL_RAD + 1e-12:
                messages.append(label+'可以了，停止！')
            else:
                pending = True
                direction = ('down' if error > 0 else 'up') if axis == 'pitch' else ('right' if error > 0 else 'left')
                out[key] = dict(direction=direction, radians=abs(error), degrees=math.degrees(abs(error)), signed_radians=error)
                messages.append(label+({'down':'向下','up':'向上','right':'向右','left':'向左'}[direction])+'继续调整，剩余%.1f°' % math.degrees(abs(error)))
        if unknown:
            self._clear(out)
        active = pending or unknown
        arrived = bool(self.goals) and not active
        if arrived:
            self.until = now+1.
        if not messages:
            arrived = bool(view.get('visible') and view.get('ready') and not view.get('gravity_assumed') and view.get('reason') in ('centered','fills_frame'))
            messages = [STOP_TEXT+'保持当前角度' if arrived else '停止，等待有效目标和角度反馈']
        out.update(ready=not active, motion_active=active, motion_arrived=arrived, motion_status='；'.join(messages))
        return out

    @staticmethod
    def _clear(view):
        for key in KEYS:
            if key in view:
                view[key] = dict(direction='none',radians=0.,degrees=0.,signed_radians=0.)
