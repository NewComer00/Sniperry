import keyboard
import serial
import cv2
import numpy
from gimbal import motor_controller as motor
from tensorflow_lite import lite_detector as detector
from mosse import mosse_tracker as tracker

if __name__ == '__main__':

    motor_port = 'COM5'
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
    flag_target_detected = False

    tracker_frame_ctr = 0
    tracker_init_frame = None
    tracker_last_region_block = None
    tracker_init_region_block = None
    tracker_Ai = None
    tracker_Bi = None
    tracker_G = None
    flag_target_tracked = False

    cap = cv2.VideoCapture(1)
    _, frame = cap.read()
    FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape

    while True:
        _, frame = cap.read()

        # if target has not been detected, do the detection
        if not flag_target_detected:
            detect_scores, detect_labels, detect_locations = \
                detector.detect(frame,
                                detector_model_file,
                                detector_label_file,
                                threshold=detector_threshold)

            for i in range(0, len(detect_scores)):
                if detect_labels[i] == detector_target_class:
                    flag_target_detected = True
                    print("%f\t%s\n" % (detect_scores[i], detect_labels[i]))

                    top_left = tuple(detect_locations[i][[0, 1]])
                    bottom_right = tuple(detect_locations[i][[2, 3]])
                    color = (255, 0, 0)
                    thickness = 3
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                    cv2.imshow("cap", frame)
                    cv2.waitKey(2000)
                    # store vars for tracker
                    tracker_init_region_block = detect_locations[i]
                    tracker_init_frame = frame
                    break

        # if target has been detected, do the tracking
        else:
            flag_target_tracked = True
            tracker_is_first_frame = False

            if tracker_frame_ctr == 0:
                frame = tracker_init_frame
                tracker_is_first_frame = True
                tracker_init_region_block[[0, 2]] = numpy.clip(tracker_init_region_block[[0, 2]], 0, frame.shape[1])
                tracker_init_region_block[[1, 3]] = numpy.clip(tracker_init_region_block[[1, 3]], 0, frame.shape[0])
                tracker_last_region_block = numpy.copy(tracker_init_region_block)

            tracker_last_region_block, tracker_Ai, tracker_Bi, tracker_G = \
                tracker.track(frame,
                              tracker_last_region_block,
                              tracker_Ai, tracker_Bi, tracker_G,
                              tracker_init_region_block,
                              tracker_is_first_frame)
            # draw the rectangle
            top_left = tuple(tracker_last_region_block[[0, 1]])
            bottom_right = tuple(tracker_last_region_block[[2, 3]])
            color = (255, 0, 0)
            thickness = 3
            frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
            tracker_frame_ctr += 1

        if motor_serial.is_open and flag_target_tracked:
            xmean = numpy.mean(tracker_last_region_block[[0, 2]])
            ymean = numpy.mean(tracker_last_region_block[[1, 3]])
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
