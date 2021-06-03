import enum
import keyboard
import serial
import cv2
import numpy
from gimbal import motor_controller as motor
from tensorflow_lite import lite_detector as detector


class Status(enum.Enum):
    DETECTING = enum.auto(),
    TRACKING = enum.auto()


if __name__ == '__main__':

    motor_port = 'COM12'
    motor_baudrate = 115200
    motor_serial = serial.Serial(motor_port, baudrate=motor_baudrate)
    motor.set_angles(motor_serial, 0, 0, 0)
    err = numpy.array([0, 0])
    err_integral = numpy.array([0, 0])
    motor_pid_k = 0.1
    motor_pid_i = 0.01
    motor_pid_d = 0.01

    detector_target_class = "person"
    detector_threshold = 0.7
    detector_model_file = "./tensorflow_lite/data/mobnet_v3_coco_official.tflite"
    detector_label_file = "./tensorflow_lite/data/labelmap.txt"

    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker_frame_ctr = 0

    cap = cv2.VideoCapture(1)
    _, frame = cap.read()
    FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape
    status = Status.DETECTING
    while True:
        _, frame = cap.read()

        # object detection
        if status is Status.DETECTING:
            detect_scores, detect_labels, detect_locations = \
                detector.detect(frame,
                                detector_model_file,
                                detector_label_file,
                                threshold=detector_threshold)

            for i in range(0, len(detect_scores)):
                if detect_labels[i] == detector_target_class:
                    print("detected: %f\t%s\n" % (detect_scores[i], detect_labels[i]))

                    top_left = tuple(detect_locations[i][[0, 1]])
                    bottom_right = tuple(detect_locations[i][[2, 3]])
                    color = (255, 0, 0)
                    thickness = 3
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                    cv2.imshow("cap", frame)
                    cv2.waitKey(2000)

                    # store vars for tracker
                    tracker_init_bbox = (detect_locations[i][0],
                                         detect_locations[i][1],
                                         detect_locations[i][2] - detect_locations[i][0],
                                         detect_locations[i][3] - detect_locations[i][1])
                    tracker_init_frame = frame
                    status = Status.TRACKING
                    break

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
            color = (255, 0, 0)
            thickness = 3
            frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
            tracker_frame_ctr += 1

            # motor control
            if motor_serial.is_open:
                xmean = bbox[0] + (bbox[2] >> 1)
                ymean = bbox[1] + (bbox[3] >> 1)
                ideal_vector = numpy.array([xmean, ymean])
                center_vector = numpy.array([FRAME_WIDTH >> 1, FRAME_HEIGHT >> 1])

                # pid control
                err_pre = err
                err = ideal_vector - center_vector
                err_integral = err + err_integral
                err_diff = err - err_pre
                pid_vector = motor_pid_k * err \
                    + motor_pid_i * err_integral \
                    + motor_pid_d * err_diff

                motor.set_angles(motor_serial,
                                 pid_vector[1],
                                 0,
                                 -pid_vector[0])

        cv2.imshow("cap", frame)
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()
