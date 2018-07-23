

import pytest
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import sys
sys.path.append('../')
from pytretina import losses
from pytretina.models import retina_resnet as retina

from pytvision.datasets.colorchecker_dataset import SyntheticColorCheckerExDataset
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view


import warnings

device = torch.device("cpu")

def test_colorcheck():
    
    num_classes=2
    net = retina.retinenet18( num_classes=num_classes, pretrained=True )
    net.to(device)
    
    data = SyntheticColorCheckerExDataset(
        pathname='~/.datasets/real/',
        generate=SyntheticColorCheckerExDataset.generate_image_and_annotations,
        transform=transforms.Compose([
            ## resize and crop                           
            mtrans.ToResize( (224,224), resize_mode='crop' ) ,
            #mtrans.CenterCrop( (200,200) ),
            #mtrans.RandomCrop( (255,255), limit=50, padding_mode=cv2.BORDER_REFLECT_101  ),
            #mtrans.ToResizeUNetFoV(388, cv2.BORDER_REFLECT_101),     
            ## color 
            #mtrans.RandomSaturation(),
            #mtrans.RandomHueSaturationShift(),
            #mtrans.RandomHueSaturation(),
            #mtrans.RandomRGBShift(),
            #mtrans.ToNegative(),
            #mtrans.RandomRGBPermutation(),
            #mtrans.ToGrayscale(),
            ## blur
            #mtrans.ToRandomTransform( mtrans.ToLinealMotionBlur( lmax=1 ), prob=0.5 ),
            #mtrans.ToRandomTransform( mtrans.ToMotionBlur( ), prob=0.5 ),
            #mtrans.ToRandomTransform( mtrans.ToGaussianBlur(), prob=0.5 ),
            ## geometrical 
            #mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 )
            #mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 )
            #mtrans.RandomScale(factor=0.2, padding_mode=cv2.BORDER_REFLECT101 ),
            #mtrans.RandomGeometricalTranform( angle=360, translation=0.2, warp=0.02, padding_mode=cv2.BORDER_REFLECT101),
            #mtrans.RandomElasticDistort( size_grid=50, padding_mode=cv2.BORDER_REFLECT101 ),
            ## tensor               
            mtrans.ToTensor(),  
            mtrans.ToNormalization(),
            ])
        )

    dataloader = DataLoader(data, batch_size=2, shuffle=False, num_workers=1 )
    
    loss_reg = losses.Smooth_l1()
    loss_cls = losses.FocalLoss()
    opt = torch.optim.Adam( net.parameters(), lr=1e-5) 

    net.train()
    label_batched = []
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['annotations'].size(),
            sample_batched['labels'].size()
            )
        
        image = sample_batched['image'] 
        annotations_target = sample_batched['annotations'] 
        labels_target = sample_batched['labels'] 

        print( image.shape )
        print( annotations_target.shape )
        print( labels_target.shape)
    
        boxes, scores, labels = net( image )


        # zero pad the outputs
        pad_size = max(0, labels_target.shape[1] - labels.shape[1] )

        #torch.nn.functional.pad(input, pad, mode='constant', value=0)
        boxes    = F.pad(boxes,  ( 0, 0, 0, pad_size ), mode='constant', value=-1)
        scores   = F.pad(scores, ( 0, pad_size), mode='constant', value=-1)
        labels   = F.pad(labels, ( 0, pad_size), mode='constant', value=-1)
        labels   = labels.int()  

        print('b', boxes.shape)
        print('s', scores.shape)
        print('l', labels.shape)


        lr = loss_reg( annotations_target  , boxes  ) 
        lc = loss_cls( labels_target       , labels )    
        loss = lr + lc

        print( lr )
        print( lc )
        print( loss )

        print(i_batch)

        # observe 4th batch and stop.
        if i_batch == 0: 
            break        




def test_backbone():

    num_classes = 20

    inputs = torch.randn( (2, 3 , 224, 224) , device=device )
    targets = [ torch.zeros(2, 9441, 5), torch.zeros(2, 9441, num_classes + 1) ]
    
    print( inputs.shape )
    print( targets[0].shape )

    net = retina.retinenet18( num_classes=num_classes, pretrained=False )

    #boxes, scores, labels = net( inputs )
    boxes, labels = net( inputs )
    
    print(boxes.shape)
    print(labels.shape)

    loc_grads = torch.randn( boxes.shape )  
    #cls_grads = torch.randn( labels.shape ) 
    labels.backward(loc_grads)
    #labels.backward(cls_grads)

    # if (boxes + 1).sum() == 0:
    #     print('not box detection!!!')
    #     assert(False)
 




if __name__ == '__main__':
    test_backbone()
    #test_colorcheck()
