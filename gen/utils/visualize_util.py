import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from time import sleep

def save_test_images(frame, segmentation_frame=None, masks=None,
        selected_mask=None):
    cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
        'test_frame.png'), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if segmentation_frame is not None:
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_segs.png'),
            cv2.cvtColor(ie.env.last_event.instance_segmentation_frame,
                cv2.COLOR_BGR2RGB))
    if masks is not None:
        mask_segmentations = np.zeros_like(masks[0], dtype='uint8')
        for i, mask in enumerate(masks):
            mask_segmentations += mask.astype('uint8') * (i + 1)
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_superpixels.png'),
            cv2.cvtColor((mark_boundaries(frame, mask_segmentations) *
                255).astype('uint8'), cv2.COLOR_BGR2RGB))
    if selected_mask is not None:
        mask_image = np.zeros((300, 300, 3))
        mask_image[:, :, :] = selected_mask[:, :, np.newaxis] == 1
        mask_image *= 255
        cv2.imwrite(os.path.join(os.environ['ALFRED_ROOT'], 'saved',
            'test_mask.png'), mask_image)
    sleep(2)

