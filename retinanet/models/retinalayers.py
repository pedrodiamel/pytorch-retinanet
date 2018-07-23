
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CONVRLU(nn.Module):
    def __init__(self, num_features_in, feature_size=256, kernel_size=3, padding=1):
        super(CONVRLU, self).__init__()        
        self.conv = nn.Conv2d(num_features_in, feature_size, kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        return F.rlu( self.conv( x ) )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class PyramidFeatures(nn.Module):
    def __init__(self, block, num_blocks):
        super(PyramidFeatures, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):

        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7

class HeadModel(nn.Module):
    def __init__(self ):
        super(HeadModel, self).__init__()
    
    def _make_head(self, out_planes, feature_size=256):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)   

class RegressionModel(HeadModel):
    def __init__(self, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        out_planes = num_anchors*4
        self.reg_head = self._make_head( out_planes, feature_size )
        self.reg = nn.Conv2d(feature_size, out_planes, kernel_size=3, stride=1, padding=1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
        self.reg.weight.data.normal_(0,0.01)
        self.reg.bias.data.fill_( 0 )


    def forward(self, x):
        x = self.reg( self.reg_head(x) )
        x = x.permute(0,2,3,1).contiguous().view(x.size(0),-1,4) # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
        return x

class ClassificationModel(HeadModel):
    def __init__(self, num_anchors=9, num_classes=20, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        out_planes = num_anchors*num_classes        
        self.cls_head = self._make_head( out_planes, feature_size )
        self.clss = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
        self.clss.weight.data.fill_( 0 )
        self.clss.bias.data.fill_( -math.log((1.0-prior)/prior) )

    def forward(self, x):
        x = self.clss( self.cls_head(x) )
        x = x.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20
        return x    


def PyramidFeatures50():
    return PyramidFeatures(Bottleneck, [3,4,6,3])

def PyramidFeatures101():
    return PyramidFeatures(Bottleneck, [2,4,23,3])

def PyramidFeatures152():
    return PyramidFeatures(Bottleneck, [3, 8, 36, 3])


def test_fms():
    net = PyramidFeatures50()
    fms = net( torch.randn(1,3,600,300) )
    for fm in fms:
        print(fm.shape)

def test_loc():
    net = PyramidFeatures50()
    reg = RegressionModel()    
    fms = net( torch.randn(1,3,600,300) )
    for i, fm in enumerate(fms):
        # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
        out = reg(fm)
        print(fm.shape, out.shape )

def test_cls():
    net = PyramidFeatures50()
    clss = ClassificationModel()
    
    fms = net( torch.randn(1,3,600,300) )
    for i, fm in enumerate(fms):
        # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20
        out = clss(fm)
        print(fm.shape, out.shape )

# test_fms()
# test_loc
# test_cls()