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
from matplotlib.pyplot import savefig
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image

#tensorboard --logdir="D:\Funmilayo_data\runs"
#%% DATA PREPARATION

# Get cpu or gpu device for training
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing {} device".format(device))

# clear GPU cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path("D:\Funmilayo_data\Anonimised data Julian")
CHECKPOINTS_DIR = Path(r"D:\Funmilayo_data\SR_all")  
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = CHECKPOINTS_DIR 
TEXT_DIR = CHECKPOINTS_DIR/"parameters.txt"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 13
IMAGE_SIZE = [42, 128, 128]
PATCH_SIZE = 16
STRIDE = 8
FILTERS = 64
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 2.5e-4
TOLERANCE = 10 # for early stopping

# write parameters to text file
text = open(TEXT_DIR, 'w') 
text.writelines(['Learning rate: '+str(LEARNING_RATE), ' Patch size: '+str(PATCH_SIZE), ' Filters: '+str(FILTERS), ' Batch size: '+str(BATCH_SIZE)])
text.close()

# find patient folders in training directory excluding hidden folders (start with .)
patients = [path for path in DATA_DIR.glob("*") if path.is_dir() and 1 <= int(path.name.replace('PAT', '')) <= 113]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {"train": patients[:-NO_VALIDATION_PATIENTS],"validation": patients[-NO_VALIDATION_PATIENTS:]}

# load training data and create DataLoader 
dataset = utils.ProstateDataset(partition["train"], IMAGE_SIZE, PATCH_SIZE , STRIDE)
dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=BATCH_SIZE)
 
# load validation data
valid_dataset = utils.ProstateDataset(partition["validation"], IMAGE_SIZE, PATCH_SIZE , STRIDE)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, pin_memory=True, batch_size=BATCH_SIZE)

# initialise model, optimizer, and loss function
unet_model = u_net.AttentionNet(num_classes=1, nr_filters=FILTERS).to(device)
mae = utils.MAELoss().to(device)
VGG19 = utils.VGGLoss().to(device)
ssim = utils.SSIMLoss().to(device)
edge = utils.EdgeLoss().to(device)
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# print number of training pixels and number of parameters
print('Number of training pixels: '+str(len(dataset)*PATCH_SIZE**2))
print('Total number of parameters: '+str(sum(p.numel() for p in unet_model.parameters())))
print('Number of training parameters: '+str(sum(p.numel() for p in unet_model.parameters() if p.requires_grad)))

#%% TRAINING

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
        pixel_loss = 1 * mae(outputs, labels)
        perc_loss = 0.2 * VGG19(outputs, labels)
        #ssim_loss = 0.2 * ssim(outputs, labels)
        #edge_loss = 0.1 * edge(outputs, labels)
        loss = pixel_loss  + perc_loss
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()  # turn off training option for evaluation
        for inputs, labels in tqdm(valid_dataloader, position=0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = unet_model(inputs) # forward pass
            pixel_loss = 1 * mae(outputs, labels)
            perc_loss = 0.2 * VGG19(outputs, labels)
            #ssim_loss = 0.2 * ssim(outputs, labels)
            #edge_loss = 0.1 * edge(outputs, labels)
            loss = pixel_loss  + perc_loss
            current_valid_loss += loss.item()

        unet_model.train()  # turn training back on

    # write to tensorboard log
    print("Loss/train: "+str(current_train_loss / len(dataloader))+ ' Epoch: '+ str(epoch))
    print("Loss/validation: "+str(current_valid_loss / len(valid_dataloader))+ ' Epoch: '+ str(epoch))
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar("Loss/validation", current_valid_loss / len(valid_dataloader), epoch)

    # save model weights of lowest validation loss
    # only start saving after 10 epochs
    # stop training after 10 epochs of no improvement
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss or epoch<= 9:
        epochs_no_improve = 0
        best_epoch = epoch
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        best_weights = unet_model.state_dict()
        
    else:
        epochs_no_improve += 1
        
    if epoch > 9 and epochs_no_improve >= TOLERANCE:
        break
    else:
        continue  
weight_path =  CHECKPOINTS_DIR / f"u_net_{best_epoch}.pth"    
torch.save(best_weights, CHECKPOINTS_DIR / f"u_net_{best_epoch}.pth")     
  
#%% APPLICATION

# Get 10 images and save to folder
CHECKPOINTS_DIR2 = weight_path 
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR2), strict=False)
unet_model.eval()
for i in range(10):
    valid_dataset = utils.ValidDataset(partition["validation"], IMAGE_SIZE, PATCH_SIZE, STRIDE,i, 'Basisplan', 25) 
    
    # load model
    unet_model = u_net.AttentionNet(num_classes=1, nr_filters=FILTERS).to(device)
    unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR2), strict=False)
    unet_model.eval()
    
    # lists to store results
    predictions = []

    # get results and plot images    
    with torch.no_grad():
        for input, target, norms in valid_dataset:
            input, target = input.to(device), target.to(device)
            output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
            prediction = torch.round(output)
            predictions.append(output.cpu())

        pred_img = (utils.FoldPatches(predictions, PATCH_SIZE, STRIDE).squeeze()).numpy()
        # save image
        fig, ax = plt.subplots()
        ax.imshow(pred_img, cmap = 'gray')
        ax.axis('off')  
        plt.savefig(CHECKPOINTS_DIR / f"img{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
