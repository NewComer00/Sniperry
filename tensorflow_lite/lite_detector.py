# Some codes are from label_image.py, Copyright 2018 The TensorFlow Authors.
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def detect(frame, model_file, label_file, input_mean=127.5, input_std=127.5, threshold=0.4):
    height_frame = np.size(frame, 0)
    width_frame = np.size(frame, 1)

    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    input_is_float = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     (width, height),
                     interpolation=cv2.INTER_AREA)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if input_is_float:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # invoke the tflite interpreter, main process here
    interpreter.invoke()

    # check the type of the output tensor
    output_is_float = output_details[0]['dtype'] == np.float32

    # model outputs
    output_locations_ = interpreter.get_tensor(output_details[0]['index'])
    output_classes_ = interpreter.get_tensor(output_details[1]['index'])
    output_scores_ = interpreter.get_tensor(output_details[2]['index'])
    num_detections_ = interpreter.get_tensor(output_details[3]['index'])

    output_locations = np.squeeze(output_locations_)
    output_classes = np.squeeze(output_classes_).astype(int)
    output_scores = np.squeeze(output_scores_)
    # num_detections = np.squeeze(num_detections_).astype(int)

    # the indices of detected objects
    result_indices = np.where(output_scores >= threshold)[0]

    # the scores of detected objects
    result_scores = output_scores[result_indices]
    # to deal with quantized integer output
    result_scores = result_scores if output_is_float else result_scores / 255.0

    # the labels of detected objects
    # load labels array, pick up the labels of results
    labels = np.array(load_labels(label_file))
    # increase the idx by one to ignore the initial background class
    result_labels = labels[output_classes[result_indices] + 1]

    # the locations of detected objects
    locs = output_locations[result_indices]
    ymins = np.array([height_frame * locs[:, 0]], dtype=np.int32).T
    xmins = np.array([width_frame * locs[:, 1]], dtype=np.int32).T
    ymaxs = np.array([height_frame * locs[:, 2]], dtype=np.int32).T
    xmaxs = np.array([width_frame * locs[:, 3]], dtype=np.int32).T
    result_locations = np.concatenate((xmins, ymins, xmaxs, ymaxs), axis=1)

    return result_scores, result_labels, result_locations
