from __future__ import annotations
import struct
import time
import serial
from gimbal.motor_controller import MotorController

# Implementation of crc16 (CRC-16-XMODEM) in python. Adapted from
# https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6?permalink_comment_id=3943460#gistcomment-3943460


def crc16(data: bytes):
    '''
    CRC-16 (XMODEM) implemented with a precomputed lookup table
    '''
    table = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7, 0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6, 0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485, 0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4, 0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823, 0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12, 0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41, 0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70, 0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F, 0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E, 0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D, 0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C, 0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB, 0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A, 0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9, 0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8, 0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
    ]
    # important, not 0xFFFF
    crc = 0x0000
    for byte in data:
        crc = (crc << 8) ^ table[(crc >> 8) ^ byte]
        # important, crc must stay 16bits all the way through
        crc &= 0xFFFF
    return crc


class A8miniController(MotorController):
    def __init__(self, port, baudrate, timeout=0.1):
        self.ser = serial.Serial(port=port,
                                 baudrate=baudrate,
                                 timeout=timeout)

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

        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

        # the command byte string template
        COMMAND_SETANGLES = b'\x55\x66\x01\x04\x00\x00\x00\x0e%s%s%s'

        # limit the angle range
        pitch_max = 25.0
        pitch_min = -90.0
        yaw_max = 135.0
        yaw_min = -135.0
        pitch = min(max(pitch, pitch_min), pitch_max)
        yaw = min(max(yaw, yaw_min), yaw_max)

        # decode int((float degrees)*10) into int16_t binary form in little-endian
        pitch_hex = struct.pack('<h', int(pitch*10))
        yaw_hex = struct.pack('<h', int(yaw*10))
        data_segmentation = COMMAND_SETANGLES[:-2] % (yaw_hex, pitch_hex)
        crc16_hex = struct.pack('<H', crc16(data_segmentation))

        # construct the command and send it through serial
        command_to_send = COMMAND_SETANGLES % (
            yaw_hex,
            pitch_hex,
            crc16_hex
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
        self.turn_to_direction(0, 0)

    def turn_to_direction(self, force_pitch: int, force_yaw: int):
        COMMAND_SETDIRECTION = b'\x55\x66\x01\x02\x00\x00\x00\x07%s%s%s'

        # limit force value into [-100, +100]
        force_pitch = min(max(force_pitch, -100), 100)
        force_yaw = min(max(force_yaw, -100), 100)

        force_pitch_hex = struct.pack('<b', force_pitch)
        force_yaw_hex = struct.pack('<b', force_yaw)
        data_segmentation = COMMAND_SETDIRECTION[:-2] % (
            force_yaw_hex, force_pitch_hex)
        crc16_hex = struct.pack('<H', crc16(data_segmentation))

        # construct the command and send it through serial
        command_to_send = COMMAND_SETDIRECTION % (
            force_yaw_hex,
            force_pitch_hex,
            crc16_hex
        )

        self.ser.write(command_to_send)

    def read_angles(self) -> tuple | None:
        COMMAND_READANG = b'\x55\x66\x01\x00\x00\x00\x00\x0d\xe8\x05'
        # clear the input buffer
        self.ser.read(self.ser.in_waiting)
        self.ser.write(COMMAND_READANG)

        while True:
            frame = self.get_dataframe()
            if frame is None:
                return None
            if frame["CMD_ID"] == 0x0d:
                data = frame["DATA"]
                yaw, pitch, roll = tuple(
                    i/10 for i in struct.unpack('<hhh', data[0:6]))
                return (pitch, roll, yaw)

    def get_dataframe(self) -> dict | None:
        # read until meet an valid ack frame
        ACK_HEADER = b'\x55\x66\x02'
        self.ser.read_until(expected=ACK_HEADER)

        try:
            data_len_hex = self.ser.read(size=2)
            data_len = struct.unpack('<H', data_len_hex)[0]

            frame_seq_hex = self.ser.read(size=2)
            frame_seq = struct.unpack('<H', frame_seq_hex)[0]

            cmd_id_hex = self.ser.read(size=1)
            cmd_id = struct.unpack('<B', cmd_id_hex)[0]

            data = self.ser.read(size=data_len)
            crc16_recv = struct.unpack('<H', self.ser.read(size=2))[0]
        except struct.error as err:
            print(f"[Error]: {err}. get_dataframe is called too frequently.")
            return None

        if crc16_recv == crc16(ACK_HEADER
                               + data_len_hex
                               + frame_seq_hex
                               + cmd_id_hex
                               + data):
            pass
        else:
            print(
                "[Warning]: CRC16 check failed, the received data is corrupted.")

        info = {
            "SEQ": frame_seq,
            "CMD_ID": cmd_id,
            "DATA": data,
        }
        return info


if __name__ == '__main__':
    import keyboard

    # connect to the gimbal serial
    motor = A8miniController('/dev/ttyS0', 115200)
    motor.reset_angles()
    motor_speed = 100
    try:
        while True:
            perform_action = False
            # time.sleep(0.01)

            if keyboard.is_pressed('a'):
                print('-yaw')
                motor.turn_to_direction(0, -motor_speed)
                perform_action = True

            if keyboard.is_pressed('d'):
                print('+yaw')
                motor.turn_to_direction(0, motor_speed)
                perform_action = True

            if keyboard.is_pressed('w'):
                print('-pitch')
                motor.turn_to_direction(motor_speed, 0)
                perform_action = True

            if keyboard.is_pressed('s'):
                print('+pitch')
                motor.turn_to_direction(-motor_speed, 0)
                perform_action = True

            if keyboard.is_pressed('space'):
                print(motor.read_angles())

            if not perform_action:
                motor.turn_to_direction(0, 0)

            time.sleep(0.1)
    except KeyboardInterrupt:
        motor.reset_angles()

