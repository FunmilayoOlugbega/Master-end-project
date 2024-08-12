# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:29:06 2024

@author: Gast
"""

import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import torch
import u_net
import utils
import os
import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset

    
# Get cpu or gpu device for training.
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nUsing {} device".format(device))


# clear GPU cache
torch.cuda.empty_cache()

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path("D:\Funmilayo_data\Anonimised data Julian")

# best epoch
BEST_EPOCH = 35
CHECKPOINTS_DIR = Path(rf"D:\Funmilayo_data\SR_all\u_net_{BEST_EPOCH}.pth")

# hyperparameters
NO_VALIDATION_PATIENTS = 1
IMAGE_SIZE = [288, 176, 272]
KERNEL_SIZE = 16
STRIDE = 8
FILTERS = 64

# load model
unet_model = u_net.AttentionNet(num_classes=1, nr_filters=FILTERS).to(device)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR), strict=False)
unet_model.eval()
print("Model loaded.")

# find patient folders in training directory
patients = [path for path in DATA_DIR.glob("*") if path.is_dir() and 126 <= int(path.name.replace('PAT', '')) <= 250]
random.shuffle(patients)

for path in patients:
    print(f"Processing path: {path}")
    utils.SRApply(path, IMAGE_SIZE, unet_model)
    

