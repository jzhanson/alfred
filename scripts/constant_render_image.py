import os
import numpy as np
import cv2
from time import sleep

"""
Script to render images in a tight loop to get around the limitation when a
headless docker container running its own x server must be used (i.e. X
forwarding from the docker container is not possible)
"""

while True:
    frame_path = os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_frame.png')
    segs_path = os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_segs.png')
    mask_path = os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_mask.png')
    if os.path.isfile(frame_path):
        cv2.imshow('frame', cv2.imread(frame_path))
    if os.path.isfile(segs_path):
        cv2.imshow('segs', cv2.imread(segs_path))
    if os.path.isfile(mask_path):
        cv2.imshow('mask', cv2.imread(mask_path))
    cv2.waitKey(1000)

