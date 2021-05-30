# Some codes are from mosse.py, Copyright 2018 TianhongDai
import numpy as np
import cv2
from mosse.utils import linear_mapping, pre_process, random_warp


# get the ground-truth gaussian response...
def get_gauss_response(img, region, sigma):
    # get the shape of the image..
    height, width = img.shape
    # get the mesh grid...
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    # get the center of the object...
    center_x = 0.5 * (region[0] + region[2])
    center_y = 0.5 * (region[1] + region[3])
    # cal the distance...
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * sigma)
    # get the response map...
    response = np.exp(-dist)
    # normalize...
    response = linear_mapping(response)
    return response


# pre train the filter on the first frame...
def pre_training(init_frame, G, num_pretrain, rotate):
    height, width = G.shape
    fi = cv2.resize(init_frame, (width, height))
    # pre-process img..
    fi = pre_process(fi)
    Ai = G * np.conjugate(np.fft.fft2(fi))
    Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
    for _ in range(num_pretrain):
        if rotate:
            fi = pre_process(random_warp(init_frame))
        else:
            fi = pre_process(init_frame)
        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

    return Ai, Bi


# start to do the object tracking...
def track(frame, last_region_block, Ai, Bi, G, init_region_block, is_first_frame, lr=0.125, sigma=100, num_pretrain=128, rotate=True):
    """

    :param Bi:
    :param Ai:
    :param last_region_block:
    :param frame:
    :param is_first_frame:
    :param init_region_block: [xmin, ymin, xmax, ymax]
    :param lr:
    :param sigma:
    :param num_pretrain:
    :param rotate:
    :return:
    """
    if is_first_frame:
        init_frame = frame
        # get the image of the first frame... (read as gray scale image...)
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        # start to draw the gaussian response...
        response_map = get_gauss_response(init_frame, init_region_block, sigma)
        # start to create the training set ...
        # get the goal..
        g = response_map[init_region_block[1]:init_region_block[3], init_region_block[0]:init_region_block[2]]
        fi = init_frame[init_region_block[1]:init_region_block[3], init_region_block[0]:init_region_block[2]]
        G = np.fft.fft2(g)
        # start to do the pre-training...
        Ai, Bi = pre_training(fi, G, num_pretrain, rotate)

    # start the tracking...
    current_frame = frame
    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float32)
    if is_first_frame:
        pos = init_region_block.copy()
        clip_pos = init_region_block.copy()
        clip_pos[[0, 2]] = np.clip(pos[[0, 2]], 0, current_frame.shape[1])
        clip_pos[[1, 3]] = np.clip(pos[[1, 3]], 0, current_frame.shape[0])

        Ai = lr * Ai
        Bi = lr * Bi
    else:
        init_region_width = init_region_block[2] - init_region_block[0]
        init_region_height = init_region_block[3] - init_region_block[1]
        pos = last_region_block.copy()
        clip_pos = last_region_block.copy()
        clip_pos[[0, 2]] = np.clip(pos[[0, 2]], 0, current_frame.shape[1])
        clip_pos[[1, 3]] = np.clip(pos[[1, 3]], 0, current_frame.shape[0])

        Hi = Ai / Bi
        fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (init_region_width, init_region_height)))
        Gi = Hi * np.fft.fft2(fi)
        gi = linear_mapping(np.fft.ifft2(Gi))
        # find the max pos...
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

        # update the position...
        pos[[0, 2]] = pos[[0, 2]] + dx
        pos[[1, 3]] = pos[[1, 3]] + dy

        # trying to get the clipped position [xmin, ymin, xmax, ymax]
        clip_pos[[0, 2]] = np.clip(pos[[0, 2]], 0, current_frame.shape[1])
        clip_pos[[1, 3]] = np.clip(pos[[1, 3]], 0, current_frame.shape[0])

        # get the current fi..
        fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
        fi = pre_process(cv2.resize(fi, (init_region_width, init_region_height)))
        # online update...
        Ai = lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Ai
        Bi = lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Bi

    region_block = pos
    return [region_block, Ai, Bi, G]
