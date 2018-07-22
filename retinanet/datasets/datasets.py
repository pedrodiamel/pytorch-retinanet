
import numpy as np
import random

import torch
from .encoder import DataEncoder

import warnings


def filter_boxes(image_group, boxs_group ):
    """ Filter boxes by removing those that are outside of the image bounds or whose width/height < 0.
    """
    # test all annotations
    boxs_group_filter = [] 
    for index, (image, boxes) in enumerate(zip(image_group, boxs_group)):
        assert(isinstance(boxes, torch.Tensor)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(boxes))

        # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
        invalid_indices = (
            (boxes[:, 2] <= boxes[:, 0]) |
            (boxes[:, 3] <= boxes[:, 1]) |
            (boxes[:, 0] < 0) |
            (boxes[:, 1] < 0) |
            (boxes[:, 2] > image.shape[1]) |
            (boxes[:, 3] > image.shape[0])
        )

        boxes = boxes[invalid_indices]
        boxs_group_filter.append(boxes)

    return boxs_group_filter

def collate(batch):
    '''Pad images and encode targets.
    As for images are of different sizes, we need to pad them to the same size.
    Args:
        batch: (list) of images, cls_targets, loc_targets.
    Returns:
        padded images, stacked cls_targets, stacked loc_targets.
    '''

    encoder = DataEncoder()

    imgs   = [x['image'] for x in batch]
    boxes  = [x['annotations'] for x in batch]
    labels = [x['labels'] for x in batch]

    # imgs -> N,C,H,W 
    input_size = np.stack([ img.shape for img in imgs  ], 0).max(0)

    c,h,w, = input_size
    num_imgs = len(imgs)
    inputs = torch.zeros(num_imgs, c, h, w)

    boxes = filter_boxes(imgs, boxes)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        loc_target, cls_target = encoder.encode(boxes[i], labels[i], input_size=(w,h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    

    return { 
        'image': inputs, 
        'annotations': torch.stack(loc_targets), 
        'labels': torch.stack(cls_targets)        
        }
