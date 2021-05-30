# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='./data/orange_banana_apple.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='./data/mobnet_v3_coco_official.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='./data/labelmap.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--threshold',
        default=0.4, type=float,
        help='detection possibility threshold')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    input_is_float = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if input_is_float:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

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
    num_detections = np.squeeze(num_detections_).astype(int)

    top_k = np.where(output_scores >= args.threshold)[0]
    labels = load_labels(args.label_file)
    for i in top_k:
        # increase the idx by one to ignore the initial background class
        label_index = output_classes[i] + 1

        if output_is_float:
            print('{:08.6f}: {}'.format(float(output_scores[i]), labels[label_index]))
        else:
            print('{:08.6f}: {}'.format(float(output_scores[i] / 255.0), labels[label_index]))

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
