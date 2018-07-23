
import torch
import torch.nn as nn
import numpy as np
import pytest

from pytretina.models.utils import meshgrid, shift

#check

def test_meshgrid():
    
    stride = 3
    a = torch.arange(0,3).float() + 0.5 * stride
    b = torch.arange(0,2).float() + 0.5 * stride
    print(a,b)
    x,y = meshgrid(a,b,row_major=False)
    print(x.shape)
    print(y.shape)
    print( torch.stack( (x,y) ).squeeze().shape )
    

def test_shift():
    
    shape = torch.Tensor( [ 2, 2 ] )
    stride = 3 
    #anchors = torch.randint( 3,6,  (3,4) )
    anchors = torch.Tensor( [[ 0, 0, 3, 3 ]] )
    print(shape)
    print(stride)
    print(anchors)
    anchors = shift(shape, stride, anchors)
    anchors = anchors.unsqueeze(dim=0).repeat( 3, 1, 1 )

    print( anchors )

