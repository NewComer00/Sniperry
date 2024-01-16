import enum
import cv2
import numpy
import keyboard
from gimbal import siyi_controller
from tensorflow_lite import lite_detector
from utils import pid_controller, rtsp_cap


class Status(enum.Enum):
    DETECTING = enum.auto(),
    TRACKING = enum.auto()


# mouse call-back function
def on_mouse(event, x, y, flags, param):
    global status

    global frame
    global detect_locations
    global tracker_init_bbox
    global tracker_init_frame

    global tracker
    global pid
    global motor
    global tracker_frame_ctr

    # left click to select a detected target
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(0, len(detect_locations)):
            tl = tuple(detect_locations[i][[0, 1]])
            br = tuple(detect_locations[i][[2, 3]])
            if tl[0] < x < br[0] and tl[1] < y < br[1]:
                # store vars for tracker
                tracker_init_bbox = (detect_locations[i][0],
                                     detect_locations[i][1],
                                     detect_locations[i][2] -
                                     detect_locations[i][0],
                                     detect_locations[i][3] - detect_locations[i][1])
                tracker_init_frame = frame
                status = Status.TRACKING
                break

    # right click to stop tracking
    elif event == cv2.EVENT_RBUTTONDOWN:
        motor.keep_angles()
        tracker_frame_ctr = 0
        tracker = cv2.legacy.TrackerMOSSE_create()
        pid.reset()
        status = Status.DETECTING

    # middle click to reset position
    elif event == cv2.EVENT_MBUTTONDOWN:
        motor.reset_angles()


if __name__ == '__main__':

    motor_port = '/dev/ttyS0'
    motor_baudrate = 115200
    motor = siyi_controller.A8miniController(motor_port, motor_baudrate)
    motor.reset_angles()

    motor_kp = 0.003
    motor_ki = 0
    motor_kd = 0.01
    # we use positional pid without the intergral term
    pid_type = pid_controller.PIDController.PIDType.POSITIONAL
    pid = pid_controller.PIDController.get_pid_controller(pid_type)(
        motor_kp, motor_ki, motor_kd, 2)

    detector_threshold = 0.6
    detector_model_file = "./tensorflow_lite/data/mobnet_v3_coco_official.tflite"
    detector_label_file = "./tensorflow_lite/data/labelmap.txt"

    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker_frame_ctr = 0

    window_name = "caps"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
    fresh = rtsp_cap.FreshestFrame(cap)
    FRAME_WIDTH, FRAME_HEIGHT = (640, 480)

    status = Status.DETECTING
    while True:
        _, frame = fresh.read()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT),
                           interpolation=cv2.INTER_AREA)

        # object detection
        if status is Status.DETECTING:

            # enable keyboard control during the detecting stage
            turning_force = 100
            force_pitch = force_yaw = 0
            if keyboard.is_pressed('a'):
                force_yaw = -turning_force
            elif keyboard.is_pressed('d'):
                force_yaw = turning_force
            elif keyboard.is_pressed('w'):
                force_pitch = -turning_force
            elif keyboard.is_pressed('s'):
                force_pitch = turning_force
            # negative yaw for natural control
            motor.turn_to_direction(force_pitch, force_yaw)

            # do object detection first
            detect_scores, detect_labels, detect_locations = \
                lite_detector.detect(frame,
                                     detector_model_file,
                                     detector_label_file,
                                     threshold=detector_threshold)

            # mark every detected objects
            for i in range(0, len(detect_scores)):
                top_left = tuple(detect_locations[i][[0, 1]])
                bottom_right = tuple(detect_locations[i][[2, 3]])
                color = (255, 0, 0)
                thickness = 3
                info = "%s: %.2f" % (detect_labels[i], detect_scores[i])
                frame = cv2.rectangle(
                    frame, top_left, bottom_right, color, thickness)
                cv2.putText(frame, info, top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness - 1, cv2.LINE_AA)

        # object tracking
        if status is Status.TRACKING:
            # initialize the tracker after we select a detected target
            if tracker_frame_ctr == 0:
                tracker.init(tracker_init_frame, tracker_init_bbox)

            # feed the next frame to the tracker
            is_tracked, bbox = tracker.update(frame)

            # if the target is being tracked
            if is_tracked:
                # draw the target rectangle
                bbox = tuple(int(elem) for elem in bbox)  # elements to int
                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                color = (0, 0, 255)
                thickness = 3
                info = "[TARGET LOCKED]"
                frame = cv2.rectangle(
                    frame, top_left, bottom_right, color, thickness)
                cv2.putText(frame, info, top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness + 1, cv2.LINE_AA)

                # pid control the motor to follow the target
                bbox_center = numpy.array(
                    [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                screen_center = numpy.array([FRAME_WIDTH/2, FRAME_HEIGHT/2])
                pid_signal = pid.update(bbox_center, screen_center)
                turning_force = (int(-pid_signal[1]*100), int(+pid_signal[0]*100))
                motor.turn_to_direction(*turning_force)
                # print(f"{bbox_center}\t{screen_center}\t{pid_signal}\t{turning_force}")

            # if target is lost ...
            else:
                color = (0, 0, 255)
                thickness = 3
                info = "[TARGET LOST]"
                cv2.putText(frame, info, (FRAME_WIDTH//2, FRAME_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness + 1, cv2.LINE_AA)

            tracker_frame_ctr += 1

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)  # for imshow() to take effect
        # DO NOT DELETE! To avoid frame from being frozen when holding a key.
        cv2.waitKey(1)

    fresh.release()
    cv2.destroyAllWindows()

