#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: train.py.py
@time: 2019/12/13 8:04
@desc:
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
import torch.nn as nn
import yaml
from collections import defaultdict
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from python.data import get_dataset
from python.models import VOC_VGG16_DeepLabV2
from python.utils import PolynomialLR, makedirs, get_device
from python.metric import scores


# Configuration
with open("configs/train.yaml", 'r') as f:
    yaml_config = yaml.load(f)
CONFIG = Dict(yaml_config)

device = get_device()
torch.backends.cudnn.benchmark = True


def get_train_params(model):
    params = defaultdict(list)

    for name, param in model.named_parameters():
        if CONFIG.MODEL.NET_ID in name:
            if "weight" in name:
                params["10xweight"].append(param)
            elif "bias" in name:
                param["20xbias"].append(param)
            else:
                print(f"\033[31m Warning: Unrecognized parameter {name}\033[00m")
        else:
            if "weight" in name:
                params["1xweight"].append(param)
            elif "bias" in name:
                param["2xbias"].append(param)
            else:
                print(f"\033[31m Warning: Unrecognized parameter {name}\033[00m")


def train():
    """
    Training DeepLab by v2 protocol
    """
    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
    )
    loader_iter = iter(loader)

    model = VOC_VGG16_DeepLabV2()
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL_FILE)
    print("    Init:", CONFIG.MODEL.INIT_MODEL_FILE)
    #for m in model.base.state_dict().keys():
    for m in model.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    # model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model.load_state_dict(state_dict, strict=True)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    # Optimizer
    train_params = get_train_params(model)
    optimizer = torch.optim.SGD(
        params=[
            {
                "params": train_params["1xweight"],
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": train_params["10xweight"],
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": train_params["2xbias"],
                "lr": 2 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
            {
                "params": train_params["20xbias"],
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Setup loss logger
    writer = SummaryWriter(os.path.join("experiment", CONFIG.EXP_ID, "summary"))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    # Path to save models
    checkpoint_dir = os.path.join(
        "experiment",
        CONFIG.EXP_ID,
        "checkpoints"
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # Random Dropout
    model.train()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                _, images, labels = next(loader_iter)

            # Propagate forward
            logits = model(images.to(device))

            labels = F.interpolate(labels, logits.size()[-2:], mode="bilinear", align_corners=True)
            # Loss
            iter_loss = criterion(logits, labels.to(device))

            # Propagate backward (just compute gradients wrt the loss)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()

            loss += float(iter_loss)

        average_loss.add(loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=iteration)

        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )

    torch.save(
        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )


if __name__ == "__main__":
    train()