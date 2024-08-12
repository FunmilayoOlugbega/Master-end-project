# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:33:57 2024

@author: Gast
"""

import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import u_net
import utils

def FoldPatches(patches_list):
    tensor_low = torch.stack(patches_list,1).squeeze()
    unfold_shape = (8,8,16,16)
    patches_orig= tensor_low.view(unfold_shape)  
    output_h = unfold_shape[0] * unfold_shape[2]
    output_w = unfold_shape[1] * unfold_shape[3]  
    patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()  
    patches_orig = patches_orig.view(output_h, output_w)
    
    return patches_orig

# Get cpu or gpu device for training.
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nUsing {} device".format(device))

# clear GPU cache
torch.cuda.empty_cache()

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"

# best epoch
BEST_EPOCH = 3
CHECKPOINTS_DIR = Path.cwd() / "model_weights" / f"u_net_{BEST_EPOCH}.pth"

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [16,128,128]
KERNEL_SIZE = 16 
STRIDE = 16
# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [path for path in DATA_DIR.glob("*") if not any(part.startswith(".") for part in path.parts)]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {"train": patients[:-NO_VALIDATION_PATIENTS],"validation": patients[-NO_VALIDATION_PATIENTS:]}


# load validation data
valid_dataset = utils.ValidDataset(partition["validation"], IMAGE_SIZE, KERNEL_SIZE, STRIDE, 1, 5)


unet_model = u_net.UNet(num_classes=1).to(device)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR), strict=False)
unet_model.eval()

low_res = []
high_res = []
predictions = []


with torch.no_grad():
    for input, target in valid_dataset:
        input, target = input.to(device), target.to(device)
        low_res.append(input.cpu())
        high_res.append(target.cpu())
        output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
        prediction = torch.round(output)
        predictions.append(output.cpu())

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(FoldPatches(low_res), cmap="gray")
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(FoldPatches(high_res), cmap="gray")
    ax[1].set_title("Ground-truth")
    ax[1].axis("off")

    ax[2].imshow(FoldPatches(predictions), cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")
