from __future__ import absolute_import

import torch
import torch.nn as nn
import math
from models.extension.normailzation.iterative_normalization import IterNorm
from models.extension.normailzation.dbn import DBN2
from models.shiftnet_cuda_v2.nn import GenericShift_cuda


__all__ = ['shiftneta']


class CSCBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_block, norm_cfg, unit_type, bias, stride=1, downsample=None, shift_exp = 6, kernel = 3):
        super(CSCBlock, self).__init__()
        self.shift_exp = shift_exp
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.kernel = kernel
        self.unit_type = unit_type

        if  downsample is not None:
            planes = planes - inplanes
        self.mid_planes = int(planes * self.shift_exp)
        self.shift = GenericShift_cuda(kernel_size=self.kernel, dilate_factor=1)
        self.conv1 = nn.Conv2d(inplanes, self.mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(self.mid_planes, planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.bn1 = norm_block(inplanes, **norm_cfg)
        self.bn2 = norm_block(self.mid_planes, **norm_cfg)
            
    def forward(self, x):
        if 'org' == self.unit_type:
            residual = x

            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.shift(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
                out = torch.cat((out, residual), 1)
            else:
                out = out + residual
            return out
        elif 'ours' == self.unit_type:
            residual = x
            
            out = self.relu(x)
            out = self.bn1(out)
            out = self.conv1(out)

            out = self.shift(out)
            out = self.relu(out)
            out = self.bn2(out)
            out = self.conv2(out)
            
            if self.downsample is not None:
                residual = self.downsample(x)
                out = torch.cat((out, residual), 1)
            else:
                out = out + residual
            return out


class ShiftNetA(nn.Module):

    def __init__(self, args, num_classes=1000):
        super(ShiftNetA, self).__init__()
        self.unit_type = args.unit_type
        self.norm_cfg = args.norm_cfg
        self.bias = args.bias
        self.df = args.df
        mult = args.mult

        if args.norm_type == 'iternorm':
            self.norm_block = IterNorm
        elif args.norm_type == 'dbn':
            self.norm_block = DBN2
        else:
            self.norm_block = nn.BatchNorm2d

        blocks = CSCBlock
        depths = [5,6,7,3] 
        depths = [int(i*mult) for i in depths] 

        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=self.bias)
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self._make_layer(blocks, int(64*mult), depths[0], stride=2, kernel=5, expansion=[4,4])
        self.layer2 = self._make_layer(blocks, int(128*mult), depths[1], stride=2, kernel=5, expansion=[4,3])
        self.layer3 = self._make_layer(blocks, int(256*mult), depths[2], stride=2, expansion=[3,2])
        self.layer4 = self._make_layer(blocks, int(512*mult), depths[3], stride=2, expansion=[2,1])

        if 'org' == self.unit_type:
            self.bn = self.norm_block(self.inplanes, **self.norm_cfg)
        if self.df or 'ours' == self.unit_type:
            self.dfbn = self.norm_block(self.inplanes, **self.norm_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(self.inplanes , num_classes)

            
        
        print('initializing shiftnetA')    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,expansion = [6,6], kernel = 3):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3,stride =2,padding=1)
            )
        
        layers = nn.Sequential()
        layers.add_module('0',block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias, stride, downsample,shift_exp = expansion[0], kernel = kernel))
        self.inplanes = planes
        for i in range(1, blocks):                    
            layers.add_module('%d'%(i),block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias, shift_exp = expansion[1], kernel = kernel))
        return layers
        

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if 'org' == self.unit_type: 
            x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        if self.df or 'ours' == self.unit_type:
            x = self.dfbn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  
        return x


def shiftneta(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ShiftNetA(**kwargs)
