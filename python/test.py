#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: test.py.py
@time: 2019/12/14 0:51
@desc:
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print(f"Current working dir is {os.getcwd()}")
import sys
print(f"Current python environment path is\n{sys.path}")

import torch
import torch.nn as nn
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import pprint
import json

from data import get_dataset
from models import VOC_VGG16_DeepLabV2
from utils import PolynomialLR, makedirs, get_device
from metric import scores


# Configuration
with open("configs/test.yaml", 'r') as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
CONFIG = Dict(yaml_config)

device = get_device()
torch.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')


def inference(img, model, msc_factors):
    img_h, img_w = img.size(2), img.size(3)
    logits = []
    for factor in msc_factors:
        scaled_img = F.interpolate(img, scale_factor=factor, mode='bilinear', align_corners=True)
        fliped_scaled_img = torch.flip(scaled_img, [3])

        batch_X = torch.cat([scaled_img, fliped_scaled_img], dim=0)
        scaled_logits = model(batch_X)

        scaled_logits[1] = torch.flip(scaled_logits[1], [2])
        scaled_logit = torch.max(scaled_logits, dim=0, keepdim=True)[0]
        logit = F.interpolate(scaled_logit, (img_h, img_w), mode="bilinear", align_corners=True)
        logits.append(logit)

    logits = torch.cat(logits, dim=0)
    fused_logit = torch.max(logits, dim=0)[0]
    prob = F.softmax(fused_logit, dim=0)
    label = torch.argmax(prob, dim=0)
    return label.cpu().numpy()


if __name__ == "__main__":
    """
    Test DeepLab by v2 protocol
    """
    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = VOC_VGG16_DeepLabV2()
    model_path = os.path.join("experiment", CONFIG.EXP_ID, "checkpoints", f"checkpoint_{CONFIG.MODEL.TEST_AT}.pth")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=True)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    save_dir = os.path.join(
        "experiment",
        CONFIG.EXP_ID,
        "metric",
        CONFIG.TEST_ID
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Metric dst:", save_path)

    preds, gts = [], []
    for image_id, image, gt_label in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        image = image.to(device)

        # # Forward propagation
        # logits = model(image)
        #
        # # Pixel-wise labeling
        # probs = F.softmax(logits, dim=1)
        # labels = torch.argmax(probs, dim=1)

        pred = inference(image, model, CONFIG.MODEL.MSC_FACTORS)

        preds.append(pred)
        gts += list(gt_label.numpy())
        print("\n")
        pprint.pprint(scores(gts[-1:], preds[-1:], CONFIG.DATASET.N_CLASSES))

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

