# coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import pickle
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
STEP = LENGTH / 4
WINS = [LENGTH * 1]
# FRAME_DIR = '/media/F/THUMOS14'
# META_DIR = os.path.join(FRAME_DIR, 'annotation_')
FRAME_DIR = '/home/vltava/disk2/THUMOS14_fewshot/frames'
META_DIR = '/home/vltava/disk2/THUMOS14_fewshot/annotations_'

USE_FLIPPED = False
###
train_segment = dataset_label_parser(META_DIR + 'test', 'test', use_ambiguous=False)


###

def generate_roi(video, start, end, stride, split):
    tmp = {}
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
    tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
    #  print (os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg'))
    if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end - 1).zfill(5) + '.jpg')):
        print(os.path.join(FRAME_DIR, split, video, 'image_' + str(end - 1).zfill(5) + '.jpg'))
        raise
    return tmp


def generate_roidb(split, segment):
    VIDEO_PATH = os.path.join(FRAME_DIR, split)
    video_list = os.listdir(VIDEO_PATH)
    roidb = []

    ###
    for vid in segment:
        if vid in video_list:
            ###

            # for i,vid in enumerate(video_list):
            # print i
            length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))

            db = np.array(segment[vid])
            if len(db) == 0:
                continue
            db[:, :2] = db[:, :2] * FPS

            for win in WINS:
                stride = int(win / LENGTH)
                step = int(stride * STEP)
                # Forward Direction
                for start in range(0, max(1, length - win + 1), step):
                    end = min(start + win, length)
                    assert end <= length

                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Add data
                    if len(rois) > 0:
                        tmp = generate_roi(vid, start, end, stride, split)
                        roidb.append(tmp)

                        if USE_FLIPPED:
                            flipped_tmp = copy.deepcopy(tmp)
                            flipped_tmp['flipped'] = True
                            roidb.append(flipped_tmp)

                # Backward Direction
                # for end in xrange(length, win, - step):
                for end in range(length, win - 1, - step):
                    start = end - win
                    assert start >= 0

                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Add data
                    if len(rois) > 0:
                        tmp = generate_roi(vid, start, end, stride, split)
                        roidb.append(tmp)

                        if USE_FLIPPED:
                            flipped_tmp = copy.deepcopy(tmp)
                            flipped_tmp['flipped'] = True
                            roidb.append(flipped_tmp)

    return roidb


val_roidb = generate_roidb('test', train_segment)
print(len(val_roidb))

print("Save dictionary")
pickle.dump(val_roidb, open('val_data_25fps.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)