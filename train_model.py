# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:26:14 2024

@author: Gast
"""
import torch
import random
from torch.utils.data import Dataset, DataLoader
import utils
import u_net
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

#%% Test

# path = [r"\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\TrainingData\10.000000-tfldynfasttra1.5x1.5t3.5sec-58929"]
# dataset = utils.ProstateDataset(path, [16,128,128], 16, 16)

# result1 = [t[0] for t in dataset]
# dataset1 = torch.cat(tuple(result1), dim=1)
# result2 = [t[1] for t in dataset]
# dataset2 = torch.cat(tuple(result2), dim=1)
# new = [dataset1, dataset2]

# plt.imshow(dataset.squeeze()[500], cmap='gray')
# plt.show()
 
# dataloader = DataLoader(dataset)
# for img, image in dataloader:
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.show()
#     plt.imshow(image.squeeze(), cmap='gray')
#     plt.show()
    
    
#%%    
# Get cpu or gpu device for training.
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nUsing {} device".format(device))

# clear GPU cache
torch.cuda.empty_cache()

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 1
IMAGE_SIZE = [50, 128, 128]
KERNEL_SIZE = 16 
STRIDE = 16
BATCH_SIZE = 16
N_EPOCHS = 5
LEARNING_RATE = 1e-4
TOLERANCE = 0.01  # for early stopping


# find patient folders in training directory excluding hidden folders (start with .)
patients = [path for path in DATA_DIR.glob("*") if not any(part.startswith(".") for part in path.parts)]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {"train": patients[:-NO_VALIDATION_PATIENTS],"validation": patients[-NO_VALIDATION_PATIENTS:]}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ValidDataset(partition["train"], IMAGE_SIZE, 16, 16,0,6)
dataloader = DataLoader(dataset, shuffle=False, drop_last=True, pin_memory=True, batch_size=BATCH_SIZE)

# load validation data
valid_dataset = utils.ValidDataset(partition["validation"], IMAGE_SIZE, 16, 16,0,6)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, drop_last=True, pin_memory=True, batch_size=BATCH_SIZE)

# initialise model, optimiser, and loss function
unet_model = u_net.UNet(num_classes=1).to(device)
loss_function = utils.MAELoss().to(device)
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    # training iterations
    # tqdm is for timing iteratiions
    for inputs, labels in tqdm(dataloader, position=0):
        inputs, labels = inputs.to(device), labels.to(device)
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs)  # forward pass
        loss = loss_function(outputs, labels.float())
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()  # turn off training option for evaluation
        for inputs, labels in tqdm(valid_dataloader, position=0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = unet_model(inputs)#.to(device) # forward pass
            loss = loss_function(outputs, labels.float())
            current_valid_loss += loss.item()

        unet_model.train()  # turn training back on

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        if epoch > 1:
            torch.save(
                unet_model.state_dict(),
                CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
            )