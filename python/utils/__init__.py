#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: __init__.py
@time: 2019/12/13 8:13
@desc:
"""
import torch
import os
from .lr_scheduler import PolynomialLR
from .crf import DenseCRF

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device