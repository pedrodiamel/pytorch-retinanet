
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from . import utils
from .retinalayers import PyramidFeatures50, ClassificationModel, RegressionModel
from .boxfilter import FilterDetections
from .anchors import Anchors, AnchorParameters


class BBoxTransform(nn.Module):
    """ Pytorch layer for applying regression values to boxes.
    """

    def __init__(self, mean=[0, 0, 0, 0], std=[0.1, 0.1, 0.2, 0.2]):
        super(BBoxTransform, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, anchors, regression):
        return utils.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        shape = img.shape
        x1 = torch.clamp(boxes[:, :, 0], min=0, max=shape[2])
        y1 = torch.clamp(boxes[:, :, 1], min=0, max=shape[1])
        x2 = torch.clamp(boxes[:, :, 2], min=0, max=shape[2])
        y2 = torch.clamp(boxes[:, :, 3], min=0, max=shape[1])  
        return torch.stack([x1, y1, x2, y2], dim=2)

class RetinaNet(nn.Module):
    
    def __init__(self, 
        num_classes=20,
        anchor_parameters = AnchorParameters.default 
        ):
        super(RetinaNet, self).__init__()

        self.num_anchors = AnchorParameters.default.num_anchors()
        self.fpn = PyramidFeatures50()
        self.num_classes = num_classes
        self.loc_layer   = RegressionModel( self.num_anchors )
        self.cls_layer   = ClassificationModel(self.num_anchors, num_classes=num_classes )

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = torch.cat([ self.loc_layer(fm) for fm in fms], dim=1)
        cls_preds = torch.cat([self.cls_layer(fm) for fm in fms], dim=1)
        return loc_preds, cls_preds

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def retinanet50( pretrained=False, **kwargs):
    """Constructs for Retinanet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model







def test():
    net = RetinaNet()
    
    loc_preds, cls_preds = net( Variable( torch.randn( 2,3,224,224 )) )
    
    print(loc_preds.size())
    print(cls_preds.size())

    loc_grads = Variable(torch.randn(loc_preds.shape))
    cls_grads = Variable(torch.randn(cls_preds.shape))
    loc_preds.backward(loc_grads, retain_graph=True)
    cls_preds.backward(cls_grads)
    


# test()