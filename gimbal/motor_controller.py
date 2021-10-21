import struct
import time
import keyboard  # using module keyboard
import serial


def set_angles(serial, pitch, roll, yaw, is_limited=True):
    """
    set the pitch, roll and yaw angles of the gimbal
    :param serial:  serial
    :param pitch:   float, in degree
    :param roll:    float, in degree
    :param yaw:     float, in degree
    :param is_limited: bool, whether the input angles are limited
    :return:
    """

    # the command byte string template
    COMMAND_SETANGLES = b'\xFA\x0E\x11%s%s%s%s\x00\x33\x34'
    FLAG_LIMITED = b'\x07'
    FLAG_UNLIMITED = b'\x00'

    # decode float degrees into IEEE754 binary form in little-endian
    pitch_hex = struct.pack('<f', pitch)
    roll_hex = struct.pack('<f', roll)
    yaw_hex = struct.pack('<f', yaw)
    flag_is_limited = FLAG_LIMITED if is_limited else FLAG_UNLIMITED

    # construct the command and send it through serial
    command_to_send = COMMAND_SETANGLES % (
        pitch_hex,
        roll_hex,
        yaw_hex,
        flag_is_limited
    )
    serial.write(command_to_send)


if __name__ == '__main__':
    # connect to the gimbal serial
    ser = serial.Serial('COM5', baudrate=115200, timeout=1)

    while ser.is_open:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown

            if keyboard.is_pressed('a'):
                print('yaw')
                set_angles(ser, 0, 0, -30)

            if keyboard.is_pressed('d'):
                print('yaw')
                set_angles(ser, 0, 0, 30)

            if keyboard.is_pressed('w'):
                print('pitch')
                set_angles(ser, -20, 0, 0)

            if keyboard.is_pressed('s'):
                print('pitch')
                set_angles(ser, 20, 0, 0)

            if keyboard.is_pressed('e'):
                print('roll')
                set_angles(ser, 0, -20, 0)

            if keyboard.is_pressed('q'):
                print('roll')
                set_angles(ser, 0, 20, 0)

            time.sleep(0.1)
            set_angles(ser, 0, 0, 0)
        except:
            break  # if user pressed a key other than the given key the loop will break
