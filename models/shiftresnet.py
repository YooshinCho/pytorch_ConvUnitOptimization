from __future__ import absolute_import
import torch
import torch.nn as nn
import math
from models.extension.normailzation.iterative_normalization import IterNorm
from models.extension.normailzation.dbn import DBN2
from models.shiftnet_cuda_v2.nn import GenericShift_cuda
from utils.tools import *


__all__ = ['shiftresnet']

class CSCBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_block, norm_cfg, unit_type, bias, stride=1, downsample=None, shift_exp = 6, kernel = 3):
        super(CSCBlock, self).__init__()
        self.unit_type = unit_type
        self.shift_exp = shift_exp
        self.kernel = kernel
        self.mid_planes = int(planes * self.shift_exp)

        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.shift = GenericShift_cuda(kernel_size=self.kernel, dilate_factor=1)
        self.conv1 = nn.Conv2d(inplanes, self.mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(self.mid_planes, planes, kernel_size=1, stride=stride, padding=0, bias=bias)

        self.bn1 = norm_block(inplanes, **norm_cfg)
        self.bn2= norm_block(self.mid_planes, **norm_cfg)
            
    def forward(self, x):
        if 'org' == self.unit_type:
            residual = x

            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu(out)
            out = self.shift(out)
            out = self.conv2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual

        elif 'ours' == self.unit_type:
            residual = x
            
            out = self.relu(x)
            out = self.bn1(out)
            out = self.conv1(out)

            out = self.relu(out)
            out = self.shift(out)
            out = self.bn2(out)
            out = self.conv2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual
        return out


class ShiftResNet(nn.Module):

    def __init__(self,args, depth, num_classes=1000):
        super(ShiftResNet, self).__init__()
        if args.norm_type == 'iternorm':
            self.norm_block = IterNorm
        elif args.norm_type == 'dbn':
            self.norm_block = DBN2
        else:
            self.norm_block = nn.BatchNorm2d
        self.unit_type = args.unit_type
        self.bias = args.bias
        self.norm_cfg = args.norm_cfg
        self.expansion = args.expansion

        block = CSCBlock 
        n = (depth - 2) // 6
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                             bias=self.bias)
        self.relu = nn.ReLU(inplace = True)
        
        self.layer1 = self._make_layer(block, 16, n, expansion=self.expansion)
        self.layer2 = self._make_layer(block, 32, n, stride = 2, expansion=self.expansion)
        self.layer3 = self._make_layer(block, 64, n, stride = 2, expansion=self.expansion)

        self.bn = self.norm_block(64,**self.norm_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
        
        print('initializing shiftresnet')    
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

    def _make_layer(self, block, planes, blocks, stride=1, expansion = 6, kernel = 3):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if 'org' == self.unit_type:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes,
                    kernel_size=1, stride=stride, bias=self.bias),
                    self.norm_block(planes , **self.norm_cfg),
                )
            if 'ours' == self.unit_type:
                downsample = nn.Sequential(
                    self.norm_block(self.inplanes , **self.norm_cfg),
                    nn.Conv2d(self.inplanes, planes,
                    kernel_size=1, stride=stride, bias=self.bias),
                )
                    
        layers = nn.Sequential()
        layers.add_module('0',block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias, stride, downsample,shift_exp = expansion, kernel = kernel))
        self.inplanes = planes
        for i in range(1, blocks):                    
            layers.add_module('%d'%(i),block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias, shift_exp = expansion, kernel = kernel))
        return layers
        

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        if 'org' == self.unit_type:
            x = self.bn(x)
        x = self.relu(x)
        if 'ours' == self.unit_type:
            x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def shiftresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ShiftResNet(**kwargs)
