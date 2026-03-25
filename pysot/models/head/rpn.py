# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_fast, xcorr_depthwise
from pysot.models.init_weight import init_weights

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        else:
            self.cls_weight = None
            self.loc_weight = None

    def forward(self, z_fs:List[torch.Tensor], x_fs:List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls: List[torch.Tensor] = []
        loc: List[torch.Tensor] = []
        
        if False:
            for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
                rpn = getattr(self, 'rpn'+str(idx))
                c, l = rpn(z_f, x_f)
                cls.append(c)
                loc.append(l)
        else:
            c, l = self.rpn2(z_fs[0], x_fs[0])
            cls.append(c)
            loc.append(l)
            c, l = self.rpn3(z_fs[1], x_fs[1])
            cls.append(c)
            loc.append(l)
            c, l = self.rpn4(z_fs[2], x_fs[2])
            cls.append(c)
            loc.append(l)

        if self.cls_weight is not None and self.loc_weight is not None:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)

def avg(lst: List[torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.stack(lst), dim=0)

def weighted_avg(lst: List[torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    # 堆叠成 (N, *shape) 的张量
    stacked = torch.stack(lst)  # shape: (N, ...)
    # 归一化权重并调整维度用于广播
    #weights = weights / weights.sum()
    # weights shape: (N,) -> (N, 1, 1, ...) 用于广播
    #view_shape = [-1] + [1] * (stacked.dim() - 1)
    weight = weight.view(-1, 1, 1, 1, 1)
    # 加权求和并在第0维求和
    return (stacked * weight).sum(dim=0)

