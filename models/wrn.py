from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.extension.normailzation.iterative_normalization import IterNorm



__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_block, norm_cfg, unit_type, bias, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.unit_type = unit_type
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                 padding=1, bias=bias)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                 padding=1, bias=bias)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                 padding=0, bias=bias) or None

        self.bn1 = norm_block(in_planes, **norm_cfg)
        self.bn2= norm_block(out_planes, **norm_cfg)
            
            
                
    def forward(self, x):
        if 'org' == self.unit_type:
            if not self.equalInOut:
                x = self.bn1(x)
                x = self.relu(x)
                
            else:
                out = self.bn1(x)
                out = self.relu(out)
            out = self.conv1(out if self.equalInOut else x)

            out = self.bn2(out)
            out = self.relu(out)
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
            
        elif 'ours' == self.unit_type:
            if not self.equalInOut:
                x = self.relu(x)
                x = self.bn1(x)
            else:
                out = self.relu(x)
                out = self.bn1(out)

            out = self.conv1(out if self.equalInOut else x)
                    

            out = self.relu(out)
            out = self.bn2(out)
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out

class NetworkBlock(nn.Module):
    def __init__(self,  nb_layers, in_planes, out_planes, block, norm_block, norm_cfg, unit_type, bias, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer( block, in_planes, out_planes, nb_layers, norm_block, norm_cfg, unit_type, bias, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers,norm_block, norm_cfg, unit_type, bias, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, norm_block, norm_cfg, unit_type, bias, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, args, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.args = args
        if args.norm_type == 'iternorm':
            norm_block = IterNorm
        else:
            norm_block = nn.BatchNorm2d
        bias = args.bias
        self.unit_type = args.unit_type
        norm_cfg = args.norm_cfg
        
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                 padding=1, bias=bias)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, norm_block, norm_cfg, self.unit_type, bias, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block,  norm_block, norm_cfg, self.unit_type, bias,2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block,  norm_block, norm_cfg, self.unit_type, bias,2, dropRate)
        # global average pooling and classifier

        self.bn1 = norm_block(nChannels[3],  **norm_cfg)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.weight is not None:
                    m.weight.data.fill_(1)
                
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if 'org' == self.unit_type:
            out = self.bn1(out)
            out = self.relu(out)
        elif 'ours' == self.unit_type:
            out = self.relu(out)
            out = self.bn1(out)
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        out =  self.fc(out)
        return  out

        
def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model
