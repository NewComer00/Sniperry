import enum
import serial
import cv2
import numpy
from gimbal import motor_controller
from tensorflow_lite import lite_detector
from utils import pid_controller


class Status(enum.Enum):
    DETECTING = enum.auto(),
    TRACKING = enum.auto()


# mouse call-back function
def on_mouse(event, x, y, flags, param):
    global status

    if event == cv2.EVENT_LBUTTONDOWN:
        global frame
        global detect_locations
        global tracker_init_bbox
        global tracker_init_frame

        for i in range(0, len(detect_locations)):
            tl = tuple(detect_locations[i][[0, 1]])
            br = tuple(detect_locations[i][[2, 3]])
            if tl[0] < x < br[0] and tl[1] < y < br[1]:
                # store vars for tracker
                tracker_init_bbox = (detect_locations[i][0],
                                     detect_locations[i][1],
                                     detect_locations[i][2] - detect_locations[i][0],
                                     detect_locations[i][3] - detect_locations[i][1])
                tracker_init_frame = frame
                status = Status.TRACKING
                break

    elif event == cv2.EVENT_RBUTTONDOWN:
        global tracker
        global pid
        global motor_serial
        global tracker_frame_ctr

        motor_controller.set_angles(motor_serial, 0, 0, 0)
        tracker_frame_ctr = 0
        tracker = cv2.legacy.TrackerMOSSE_create()
        pid.reset()
        status = Status.DETECTING


if __name__ == '__main__':

    motor_port = 'COM4'
    motor_baudrate = 115200
    motor_serial = serial.Serial(motor_port, baudrate=motor_baudrate)
    motor_controller.set_angles(motor_serial, 0, 0, 0)
    motor_kp = 0.15
    motor_ki = 0.006
    motor_kd = 0.25
    pid = pid_controller.PIDController(motor_kp, motor_ki, motor_kd, 2)

    detector_threshold = 0.6
    detector_model_file = "./tensorflow_lite/data/mobnet_v3_coco_official.tflite"
    detector_label_file = "./tensorflow_lite/data/labelmap.txt"

    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker_frame_ctr = 0

    window_name = "caps"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape

    status = Status.DETECTING
    while True:
        _, frame = cap.read()

        # object detection
        if status is Status.DETECTING:
            detect_scores, detect_labels, detect_locations = \
                lite_detector.detect(frame,
                                     detector_model_file,
                                     detector_label_file,
                                     threshold=detector_threshold)

            for i in range(0, len(detect_scores)):
                # print("detected: %f\t%s\n" % (detect_scores[i], detect_labels[i]))
                top_left = tuple(detect_locations[i][[0, 1]])
                bottom_right = tuple(detect_locations[i][[2, 3]])
                color = (255, 0, 0)
                thickness = 3
                frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                cv2.putText(frame, detect_labels[i], top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness - 1, cv2.LINE_AA)

        # object tracking
        if status is Status.TRACKING:
            flag_target_tracked = True

            if tracker_frame_ctr == 0:
                tracker.init(tracker_init_frame, tracker_init_bbox)

            _, bbox = tracker.update(frame)
            bbox = tuple(int(elem) for elem in bbox)  # elements to int

            # draw the rectangle
            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            color = (0, 0, 255)
            thickness = 3
            info = "[TARGET LOCKED]"
            frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
            cv2.putText(frame, info, top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness + 1, cv2.LINE_AA)
            tracker_frame_ctr += 1

            # motor control
            if motor_serial.is_open:
                xmean = bbox[0] + (bbox[2] >> 1)
                ymean = bbox[1] + (bbox[3] >> 1)
                bbox_center = numpy.array([xmean, ymean])
                screen_center = numpy.array([FRAME_WIDTH >> 1, FRAME_HEIGHT >> 1])

                # pid control
                pid_signal = pid.update(bbox_center, screen_center)
                motor_controller.set_angles(motor_serial,
                                            pid_signal[1],
                                            0,
                                            -pid_signal[0])

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
