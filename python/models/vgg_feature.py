#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: vgg_feature.py
@time: 2019/12/12 12:31
@desc: VGG Model. Based on torch VISION
"""
import torch
import torch.nn as nn


class ConvReLU(nn.Sequential):

    def __init__(self, in_ch, out_ch, dilation, layer_idx, seq_idx):
        super(ConvReLU, self).__init__()

        self.add_module(f"conv{layer_idx}_{seq_idx}",
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilation,
                                  dilation=dilation))
        self.add_module(f"relu{layer_idx}_{seq_idx}",
                        nn.ReLU(inplace=True))


class VGGLayer(nn.Sequential):

    def __init__(self, in_ch, out_ch, conv_num, dilation, pool_size, pool_stride, layer_idx):
        super(VGGLayer, self).__init__()

        for seq_idx in range(1, conv_num+1):
            self.add_module(f"conv_relu_{seq_idx}",
                            ConvReLU(in_ch=in_ch if seq_idx == 1 else out_ch, out_ch=out_ch,
                                     dilation=dilation, layer_idx=layer_idx, seq_idx=seq_idx))

        # Padding size of pooling will be 1 when kernel size is 3 and 0 when kernel size is 2
        self.add_module(f"pool{layer_idx}",
                        nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=pool_size%2, ceil_mode=True))


class VGGFeature(nn.Sequential):

    def __init__(self, in_ch, out_chs, conv_nums, dilations, pool_strides, pool_size):
        super(VGGFeature, self).__init__()

        for i, layer_idx in enumerate(range(1, len(out_chs) + 1)):
            self.add_module(f"layer{layer_idx}",
                            VGGLayer(in_ch=in_ch if layer_idx == 1 else out_chs[i - 1],
                                     out_ch=out_chs[i], conv_num=conv_nums[i], dilation=dilations[i],
                                     pool_size=pool_size, pool_stride=pool_strides[i], layer_idx=layer_idx))


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(log_dir="temp")

    writer.add_graph(VGGFeature(
        in_ch=3,
        out_chs=[64, 128, 256, 512, 512],
        conv_nums=[2, 2, 3, 3, 3],
        dilations=[1, 1, 1, 1, 2],
        pool_strides=[2, 2, 2, 1, 1],
        pool_size=3
    ), input_to_model=torch.randn((10, 3, 321, 321), dtype=torch.float32))

    print("Graph has been writen to the temp dir")







