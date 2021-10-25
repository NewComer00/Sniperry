import numpy


class PIDController:
    def __init__(self, kp, ki, kd, signal_size):
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._err = numpy.zeros(signal_size)
        self._err_integral = numpy.zeros(signal_size)

    def update(self, current_signal, target_signal):
        err_pre = self._err
        self._err = current_signal - target_signal
        self._err_integral = self._err + self._err_integral
        err_diff = self._err - err_pre
        output_signal = \
            self._kp * self._err + self._ki * self._err_integral + self._kd * err_diff
        return output_signal

    def reset(self):
        self._err.fill(0)
        self._err_integral.fill(0)


