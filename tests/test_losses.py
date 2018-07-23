

import sys
sys.path.append('../')
from pytretina import losses

import torch
import numpy as np
import pytest


def test_smooth_l1():

    regression = np.array([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    ], dtype=float)
    regression = torch.from_numpy(regression)

    regression_target = np.array([
        [
            [0, 0, 0   , 1, 1],
            [0, 0, 1   , 0, 1],
            [0, 0, 0.05, 0, 1],
            [0, 0, 1   , 0, 0],
        ]
    ], dtype=float)
    regression_target = torch.from_numpy(regression_target)

    loss = losses.Smooth_l1()(regression_target, regression).numpy()
        
    print(loss)
    print((((1 - 0.5 / 9) * 2 + (0.5 * 9 * 0.05 ** 2)) / 3))

    assert loss == pytest.approx((((1 - 0.5 / 9) * 2 + (0.5 * 9 * 0.05 ** 2)) / 3))


test_smooth_l1()