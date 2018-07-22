
import numpy as np
import torch
import torch.nn as nn

from . import utils

#https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet
class Anchors(nn.Module):
   
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            @size: The base size of the anchors to generate.
            @stride: The stride of the anchors to generate.
            @ratios: The ratios of the anchors to generate (defaults to [0.5, 1, 2]).
            @scales: The scales of the anchors to generate (defaults to [2^0, 2^(1/3), 2^(2/3)]).
        """        

        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        
        if ratios is None:
            self.ratios  = torch.Tensor([0.5, 1, 2]) 
        elif isinstance(ratios, list):
            self.ratios  = torch.Tensor(ratios)
        if scales is None:
            self.scales  = torch.Tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
        elif isinstance(scales, list):
            self.scales  = torch.Tensor(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = torch.Tensor(
            utils.generate_anchors(
                base_size=size,
                ratios=ratios,
                scales=scales,
            ))            
             
        super(Anchors, self).__init__(*args, **kwargs)
            

    def forward(self, inputs):

        features = inputs
        features_shape = features.shape

        # generate proposals from bbox deltas and shifted anchors
        anchors = utils.shift(features_shape[2:], self.stride, self.anchors)
        anchors = anchors.unsqueeze(dim=0).repeat( features_shape[0], 1, 1 )

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[2:]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = torch.Tensor([0.5, 1, 2]).float() ,
    scales  = torch.Tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]).float() ,
)
