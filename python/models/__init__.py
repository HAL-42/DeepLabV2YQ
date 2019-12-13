#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: __init__.py
@time: 2019/12/12 18:10
@desc:
"""
import torch
from torch import nn

from .vgg_feature import VGGFeature
from .vgg_ASPP import VGGASPP


class VOC_VGG16_DeepLabV2(nn.Module):

    def __init__(self):
        super(VOC_VGG16_DeepLabV2, self).__init__()

        self.VGGFeature = VGGFeature(
            in_ch=3,
            out_chs=[64, 128, 256, 512, 512],
            conv_nums=[2, 2, 3, 3, 3],
            dilations=[1, 1, 1, 1, 2],
            pool_strides=[2, 2, 2, 1, 1],
            pool_size=3
        )

        self.VGGASPP = VGGASPP(
            in_ch=512,
            num_classes=21,
            rates=[6, 12, 18, 24],
            start_layer_idx=6,
            net_id='pascal'
        )

    def forward(self, x):
        x = self.VGGFeature(x)
        return self.VGGASPP(x)


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    voc_vgg16_deeplabV2 = VOC_VGG16_DeepLabV2()

    for name, module in voc_vgg16_deeplabV2.named_modules():
        print("-----------------------------------------")
        print(f"Module name is {name}")
        print(f"Module type is {type(module)}")

    with SummaryWriter(log_dir="temp") as writer:
        writer.add_graph(voc_vgg16_deeplabV2, input_to_model=torch.randn((10, 3, 321, 321), dtype=torch.float32))

    print("Graph has been writen to the temp dir")
