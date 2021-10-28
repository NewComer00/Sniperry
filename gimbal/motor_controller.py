import struct
import time
import keyboard  # using module keyboard
import serial


class MotorController:
    def __init__(self, port, baudrate, limited_angles=False):
        self.ser = serial.Serial(port=port, baudrate=baudrate)
        self.limited_angles = limited_angles

        self.pitch = 0
        self.roll = 0
        self.yaw = 0

    def set_angles(self, pitch, roll, yaw):
        """
        set the pitch, roll and yaw angles of the gimbal
        :param pitch:   float, in degree
        :param roll:    float, in degree
        :param yaw:     float, in degree
        :return:
        """

        # the command byte string template
        COMMAND_SETANGLES = b'\xFA\x0E\x11%s%s%s%s\x00\x33\x34'
        FLAG_LIMITED = b'\x07'
        FLAG_UNLIMITED = b'\x00'

        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

        # decode float degrees into IEEE754 binary form in little-endian
        pitch_hex = struct.pack('<f', pitch)
        roll_hex = struct.pack('<f', roll)
        yaw_hex = struct.pack('<f', yaw)
        flag_is_limited = FLAG_LIMITED if self.limited_angles else FLAG_UNLIMITED

        # construct the command and send it through serial
        command_to_send = COMMAND_SETANGLES % (
            pitch_hex,
            roll_hex,
            yaw_hex,
            flag_is_limited
        )
        self.ser.write(command_to_send)

    def reset_angles(self):
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.set_angles(0, 0, 0)

    def increase_angles(self, pitch_diff, roll_diff, yaw_diff):
        self.pitch += pitch_diff
        self.roll += roll_diff
        self.yaw += yaw_diff
        self.set_angles(self.pitch, self.roll, self.yaw)

    def keep_angles(self):
        self.set_angles(self.pitch, self.roll, self.yaw)


if __name__ == '__main__':
    # connect to the gimbal serial
    motor = MotorController('COM4', 115200)

    while True:
        if keyboard.is_pressed('a'):
            print('-yaw')
            motor.set_angles(0, 0, -30)

        if keyboard.is_pressed('d'):
            print('+yaw')
            motor.set_angles(0, 0, 30)

        if keyboard.is_pressed('w'):
            print('-pitch')
            motor.set_angles(-20, 0, 0)

        if keyboard.is_pressed('s'):
            print('+pitch')
            motor.set_angles(20, 0, 0)

        if keyboard.is_pressed('e'):
            print('-roll')
            motor.set_angles(0, -20, 0)

        if keyboard.is_pressed('q'):
            print('+roll')
            motor.set_angles(0, 20, 0)

        time.sleep(0.1)
        motor.set_angles(0, 0, 0)
