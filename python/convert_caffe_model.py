#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: convert_caffe_model.py
@time: 2019/12/12 20:59
@desc:
"""
from collections import Counter, OrderedDict

import numpy as np
import torch
from addict import Dict
import yaml

from python.caffe import caffe_pb2
from python.models import VOC_VGG16_DeepLabV2

with open("configs/convert.yaml", "r") as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
CONFIG = Dict(yaml_config)

NAME_TYPE = {
    "NONE": 0,
    "ABSVAL": 35,
    "ACCURACY": 1,
    "ARGMAX": 30,
    "BNLL": 2,
    "CONCAT": 3,
    "CONTRASTIVE_LOSS": 37,
    "CONVOLUTION": 4,
    "DATA": 5,
    "DECONVOLUTION": 39,
    "DROPOUT": 6,
    "DUMMY_DATA": 32,
    "EUCLIDEAN_LOSS": 7,
    "ELTWISE": 25,
    "EXP": 38,
    "FLATTEN": 8,
    "HDF5_DATA": 9,
    "HDF5_OUTPUT": 10,
    "HINGE_LOSS": 28,
    "IM2COL": 11,
    "IMAGE_DATA": 12,
    "INFOGAIN_LOSS": 13,
    "INNER_PRODUCT": 14,
    "LRN": 15,
    "MEMORY_DATA": 29,
    "MULTINOMIAL_LOGISTIC_LOSS": 16,
    "MVN": 34,
    "POOLING": 17,
    "POWER": 26,
    "RELU": 18,
    "SIGMOID": 19,
    "SIGMOID_CROSS_ENTROPY_LOSS": 27,
    "SILENCE": 36,
    "SOFTMAX": 20,
    "SOFTMAX_LOSS": 21,
    "SPLIT": 22,
    "SLICE": 33,
    "TANH": 23,
    "WINDOW_DATA": 24,
    "THRESHOLD": 31,
}

TYPE_NAME = {v: k for k, v in NAME_TYPE.items()}


def caffe_type2name(type):
    if isinstance(type, str):
        return type.upper()
    elif isinstance(type, int):
        return TYPE_NAME[type]


def PrintCaffeModel(model, verbose=True):
    layers = model.layer if model.layer else model.layers
    print('name: "%s"' % model.name)

    print("-------------------Layers-----------------------")
    if verbose:
        layer_id = -1
        for layer in layers:
            layer_id = layer_id + 1
            print('layer {')
            print('  name: "%s"' % layer.name)
            print('  type: "%s"' % caffe_type2name(layer.type))

            tops = layer.top
            for top in tops:
                print('  top: "%s"' % top)

            bottoms = layer.bottom
            for bottom in bottoms:
                print('  bottom: "%s"' % bottom)

            if len(layer.include) > 0:
                print('  include {')
                includes = layer.include
                phase_mapper = {
                    '0': 'TRAIN',
                    '1': 'TEST'
                }

                for include in includes:
                    if include.phase is not None:
                        print('    phase: ', phase_mapper[str(include.phase)])
                print('  }')

            if layer.transform_param is not None and layer.transform_param.scale is not None and layer.transform_param.scale != 1:
                print('  transform_param {')
                print('    scale: %s' % layer.transform_param.scale)
                print('  }')

            if layer.data_param is not None and (
                    layer.data_param.source != "" or layer.data_param.batch_size != 0 or layer.data_param.backend != 0):
                print('  data_param: {')
                if layer.data_param.source is not None:
                    print('    source: "%s"' % layer.data_param.source)
                if layer.data_param.batch_size is not None:
                    print('    batch_size: %d' % layer.data_param.batch_size)
                if layer.data_param.backend is not None:
                    print('    backend: %s' % layer.data_param.backend)
                print('  }')

            if layer.param is not None:
                params = layer.param
                for param in params:
                    print('  param {')
                    if param.lr_mult is not None:
                        print('    lr_mult: %s' % param.lr_mult)
                    print('  }')

            if layer.convolution_param is not None:
                print('  convolution_param {')
                conv_param = layer.convolution_param
                if conv_param.num_output is not None:
                    print('    num_output: %d' % conv_param.num_output)
                if len(conv_param.kernel_size) > 0:
                    for kernel_size in conv_param.kernel_size:
                        print('    kernel_size: ', kernel_size)
                if len(conv_param.stride) > 0:
                    for stride in conv_param.stride:
                        print('    stride: ', stride)
                if conv_param.weight_filler is not None:
                    print('    weight_filler {')
                    print('      type: "%s"' % conv_param.weight_filler.type)
                    print('    }')
                if conv_param.bias_filler is not None:
                    print('    bias_filler {')
                    print('      type: "%s"' % conv_param.bias_filler.type)
                    print('    }')
                print('  }')

            print('}')
    else:
        for layer in layers:
            print(f"Layer's type = {caffe_type2name(layer.type)}")
            print(f"Layer's name = {layer.name}")
            print(f"Layer's blobs len = {len(layer.blobs)}")

    print("------------------Static-----------------------")
    print(
        *Counter(
            [(caffe_type2name(layer.type), len(layer.blobs)) for layer in layers]
        ).most_common(),
        sep="\n",
    )


def parse_caffemodel(model_path):
    # *Read Model
    caffemodel = caffe_pb2.NetParameter()
    with open(model_path, 'rb') as f:
        caffemodel.MergeFromString(f.read())
    # *PrintCaffeModel
    PrintCaffeModel(caffemodel)
    # * Get model's conv parameters
    layers = caffemodel.layer if caffemodel.layer else caffemodel.layers
    params = OrderedDict()
    for layer in layers:
        if "CONVOLUTION" == caffe_type2name(layer.type):
            params[layer.name] = {}
            params[layer.name]["weight"] = np.array(layer.blobs[0].data, dtype=np.float32)
            params[layer.name]["bias"] = np.array(layer.blobs[1].data, dtype=np.float32)

            print("\033[34m[Read Param]\033[00m", layer.name)

    return params


def find_named_module(model, layer_name):
    for name, module in model.named_modules():
        if name.split('.')[-1] == layer_name:
            return name, module
    raise LookupError(f"No such model named {layer_name}")


if __name__ == "__main__":
    caffe_params = parse_caffemodel(CONFIG.INIT_MODEL_FILE)
    model = VOC_VGG16_DeepLabV2()

    converted_state_dict = OrderedDict()
    for layer_name, layer_params in caffe_params.items():
        module_name, module = find_named_module(model, layer_name)
        for layer_param_name, layer_param_value in layer_params.items():
            module_param_name = module_name + layer_param_name
            module_param = eval(f"model.{module_name}.{layer_param_name}")

            converted_state_dict[module_param_name] = torch.from_numpy(layer_param_value).view_as(module_param)
            print("\033[33m[Param]\033[00m", f"{layer_param_name} is converted")
        print("\033[32m[Layer]\033[00m", f"{layer_name} is converted")

    print("\033[33mVerify the converted model\033[00m")
    model.load_state_dict(converted_state_dict, strict=True)

    print(f"Saving to {CONFIG.SAVE_MODEL_FILE}")
    torch.save(converted_state_dict, CONFIG.SAVE_MODEL_FILE)
















