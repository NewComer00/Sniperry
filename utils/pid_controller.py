import enum
import numpy


class PIDController:
    class PIDType(enum.Enum):
        POSITIONAL = enum.auto(),
        INCREMENTAL = enum.auto()

    _register = {}

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "_pid_type"):
            raise ValueError(cls.__name__ + ' has no _pid_type')
        PIDController._register.update({pt: cls for pt in cls._pid_type})

    @classmethod
    def get_pid_controller(cls, pid_type):
        return cls._register.get(pid_type)

    def __init__(self, kp, ki, kd):
        self._kp = kp
        self._ki = ki
        self._kd = kd

    def update(self, current_signal, target_signal):
        pass

    def reset(self):
        pass


class PIDIncremental(PIDController):
    _pid_type = [PIDController.PIDType.INCREMENTAL]

    def __init__(self, kp, ki, kd, signal_size):
        super().__init__(kp, ki, kd)
        self._err = numpy.zeros(signal_size)
        self._err_pre = numpy.zeros(signal_size)
        self._err_pre_pre = numpy.zeros(signal_size)

    def update(self, current_signal, target_signal):
        self._err = current_signal - target_signal
        err_diff = self._err - self._err_pre
        err_diff_pre = self._err_pre - self._err_pre_pre
        output_signal = \
            self._kp * err_diff + self._ki * self._err + self._kd * (err_diff - err_diff_pre)

        self._err_pre_pre = self._err_pre
        self._err_pre = self._err
        return output_signal

    def reset(self):
        self._err.fill(0)
        self._err_pre.fill(0)
        self._err_pre_pre.fill(0)


class PIDPositional(PIDController):
    _pid_type = [PIDController.PIDType.POSITIONAL]

    def __init__(self, kp, ki, kd, signal_size):
        super().__init__(kp, ki, kd)
        self._err = numpy.zeros(signal_size)
        self._err_integral = numpy.zeros(signal_size)

    def update(self, current_signal, target_signal):
        err_pre = self._err
        self._err = current_signal - target_signal
        self._err_integral += self._err
        # print(self._err_integral)
        err_diff = self._err - err_pre
        output_signal = \
            self._kp * self._err + self._ki * self._err_integral + self._kd * err_diff
        return output_signal

    def reset(self):
        self._err.fill(0)
        self._err_integral.fill(0)
