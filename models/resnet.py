from __future__ import absolute_import

import torch
import torch.nn as nn
import math
from models.extension.normailzation.iterative_normalization import IterNorm
from models.extension.normailzation.dbn import DBN2
from utils.tools import *

__all__ = ['resnet']


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, norm_block, norm_cfg, unit_type, bias, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.unit_type = unit_type

        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample
        if 'org' == self.unit_type:
            self.bn1 = norm_block(planes, **norm_cfg)
            self.bn2= norm_block(planes, **norm_cfg)
        elif 'ours' == self.unit_type:
            self.bn1 = norm_block(inplanes, **norm_cfg)
            self.bn2= norm_block(planes, **norm_cfg)
        
            
    def forward(self, x):
        if 'org' == self.unit_type:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
        elif 'ours' == self.unit_type:
            residual = x
            
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.relu(out)

            out = self.bn2(out)
            out = self.conv2(out)
            
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, norm_block, norm_cfg, unit_type, bias, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.unit_type = unit_type

        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=bias)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=bias)
        if 'org' == self.unit_type:
            self.bn1 =  norm_block(planes, **norm_cfg)
            self.bn2 =  norm_block(planes, **norm_cfg)      
            self.bn3 =  norm_block(planes*4, **norm_cfg)
        elif 'ours' == self.unit_type:
            self.bn1 =  norm_block(inplanes, **norm_cfg)
            self.bn2 =  norm_block(planes, **norm_cfg)      
            self.bn3 =  norm_block(planes, **norm_cfg)
                        

    def forward(self, x):
        if 'org' == self.unit_type:
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

        elif 'ours' == self.unit_type:
            residual = x

            out = self.bn1(x)
            out = self.conv1(out)
            out = self.relu(out)

            out = self.bn2(out)
            out = self.conv2(out)
            out = self.relu(out)

            out = self.bn3(out)
            out = self.conv3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args, depth, num_classes=1000):
        super(ResNet, self).__init__()
        if args.norm_type == 'iternorm':
            self.norm_block = IterNorm
        elif args.norm_type == 'dbn':
            self.norm_block = DBN2
        else:
            self.norm_block = nn.BatchNorm2d
        self.bias = args.bias
        self.norm_cfg = args.norm_cfg
        self.unit_type = args.unit_type
        self.df = args.df
        self.dataset = args.dataset

        
        if self.dataset.startswith('cifar'):
            block = Bottleneck if depth >=44 else BasicBlock
            if block == BasicBlock:
                n = int(depth - 2) // 6
            elif block == Bottleneck:
                n = int(depth -2) // 9

            
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                 bias=self.bias)
        
            if 'org' == self.unit_type:
                self.bn = self.norm_block(16, **self.norm_cfg)
            elif 'ours' == self.unit_type:
                self.bn = self.norm_block(64 * block.expansion, **self.norm_cfg)
            
            self.relu = nn.ReLU(inplace = True)
            
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride = 2)
            self.layer3 = self._make_layer(block, 64, n, stride = 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
    
                
        elif self.dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=self.bias)
            if 'org' == self.unit_type:
                self.bn = self.norm_block(64,**self.norm_cfg)

            if 'ours' == self.unit_type or self.df:
                self.dfbn = self.norm_block(512 * blocks[depth].expansion,**self.norm_cfg)
                            

            self.relu = nn.ReLU(inplace = True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1) 
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

            
        
        print('initializing resnet')    
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if 'org' == self.unit_type:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=self.bias),
                    self.norm_block(planes * block.expansion, **self.norm_cfg),
                )
        elif 'ours' == self.unit_type:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    self.norm_block(self.inplanes, **self.norm_cfg),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=self.bias),
                )
        
        layers = nn.Sequential()
        layers.add_module('0',block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):                    
            layers.add_module('%d'%(i),block(self.inplanes, planes, self.norm_block, self.norm_cfg, self.unit_type, self.bias))

        return layers
    def forward(self, x):
        if self.dataset.startswith('cifar'):
            x = self.conv1(x)
            if 'org' == self.unit_type:
                x = self.bn(x)
            x = self.relu(x)    # 32x32
            
            x = self.layer1(x)  # 32x32
            x = self.layer2(x)  # 16x16
            x = self.layer3(x)  # 8x8
            
            if 'ours' == self.unit_type:
                x = self.bn(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
                
        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            if 'org' == self.unit_type:
                x = self.bn(x)
            x = self.relu(x)
            
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            if self.df or 'ours' ==  self.unit_type:
                x = self.dfbn(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)  
     
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
