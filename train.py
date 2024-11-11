# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:09:30 2024

@author: Gast
"""
import torch
torch.autograd.set_detect_anomaly(True)
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import multi_utils
import sr_net
import torchvision
import seg_net
import multi_loader
import metrics
import plotter
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import savefig
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import cv2 as cv


#%% DATA PREPARATION

# Get cpu or gpu device for training
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing {} device".format(device))
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path("D:\Funmilayo_data\Anonimised data Julian")

# SR training settings and hyperparameters
SR_DIR = Path(r"D:\Funmilayo_data\multi_model\SR")  
SR_DIR.mkdir(parents=True, exist_ok=True)

#TENSORBOARD_LOGDIR = CHECKPOINTS_DIR 
NO_VALIDATION_PATIENTS = 20
DATA_SIZE = [50,128,128]
IMAGE_SIZE = [48, 128, 128]
PATCH_SIZE = 64
STRIDE = 32
FILTERS = 64
BATCH_SIZE = 432
N_EPOCHS = 100
SR_LEARNING_RATE = 2.5e-4

#TEXT_DIR = CHECKPOINTS_DIR/"parameters.txt"
# # write parameters to text file
# text = open(TEXT_DIR, 'w') 
# text.writelines(['Learning rate: '+str(LEARNING_RATE), ' Patch size: '+str(PATCH_SIZE), ' Filters: '+str(FILTERS), ' Batch size: '+str(BATCH_SIZE)])
# text.close()

# SegNet training settings and hyperparameters
SEG_DIR = Path(r"D:\Funmilayo_data\multi_model\SEG")  
SEG_DIR.mkdir(parents=True, exist_ok=True)
image_path_seg = SEG_DIR / "Images"
image_path_seg.mkdir(parents=True, exist_ok=True)

STRUCTURES = ['CTV']
EQUALIZATION_MODE = None
THREED = True
LOAD_DOSE = False
SEG_LEARNING_RATE = 2e-3
METRICS = [metrics.HausdorffDistance(), metrics.RelativeVolumeDifference(), metrics.SurfaceDice()]


# find patient folders in training directory excluding hidden folders (start with .)
patients = [path for path in DATA_DIR.glob("*") if path.is_dir() and 1 <= int(path.name.replace('PAT', '')) <= 200]
partition = {"train": patients[:-NO_VALIDATION_PATIENTS],"validation": patients[-NO_VALIDATION_PATIENTS:]}

# load training data and create DataLoader 
dataset = multi_utils.ProstateDataset(partition["train"], DATA_SIZE, PATCH_SIZE , STRIDE)
dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=1)
# label_set = multi_utils.DataSet(partition["train"], masks=STRUCTURES, size=IMAGE_SIZE, folder='ProSeg')
# train_labelloader = DataLoader(label_set, shuffle=False, pin_memory=True, batch_size=1)

# load validation data
valid_dataset = multi_utils.ProstateDataset(partition["validation"], DATA_SIZE, PATCH_SIZE , STRIDE)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, pin_memory=True, batch_size=1)
# label_valid = multi_utils.DataSet(partition["train"], masks=STRUCTURES, size=IMAGE_SIZE, folder='ProSeg')
# valid_labelloader = DataLoader(label_valid, shuffle=False, pin_memory=True, batch_size=1)

#%% Initialisations 

# initialise SR model, optimizer, and loss function
sr_model = sr_net.UNet(num_classes=1, nr_filters=FILTERS).to(device)
mae = multi_utils.MAELoss().to(device)
VGG19 = multi_utils.VGGLoss().to(device)
sr_optimizer = torch.optim.Adam(sr_model.parameters(), lr=SR_LEARNING_RATE)

# initialise Seg model, optimizer, and loss function
seg_model = seg_net.SegNet3D(in_chs=1, out_chs=1).to(device)
seg_optimizer = torch.optim.AdamW(seg_model.parameters(), lr=SEG_LEARNING_RATE, betas=(0.95,0.99), weight_decay=0.01)
dice_focal_loss = metrics.DiceFocalLoss(gamma=1, alpha=0.75)
    
#minimum_valid_loss = 10  # initial validation loss
#writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

#%% TRAINING
seg_train_list = []
seg_valid_list = []
total_train_list = []
total_valid_list = []
minimum_valid_loss = 10
tolerance = 10
scaler = GradScaler()
            
# training loop
for epoch in range(N_EPOCHS):
    seg_train_loss = 0.0
    seg_valid_loss = 0.0
    total_train_loss = 0.0
    total_valid_loss = 0.0
    count = 0
    images = [] 

    # training iterations
    # tqdm is for timing iteratiions
    for inputs, labels in tqdm(dataloader, position=0):
        inputs, labels = inputs.squeeze(0).squeeze(0).unsqueeze(1).to(device), labels.squeeze(0).squeeze(0).unsqueeze(1).to(device)
        
        # ---- Super-resolution pass ----
        with autocast():  # Mixed precision
            outputs = sr_model(inputs)  # forward pass
            pixel_loss = 1 * mae(outputs, labels)
            perc_loss = 0.2 * VGG19(outputs, labels)
            sr_loss = pixel_loss  + perc_loss
        
        # Clean up SR-specific memory
        del inputs, labels, pixel_loss, perc_loss
        #torch.cuda.empty_cache()


        # fold patches to 3D image
        with torch.no_grad():
            image = multi_utils.FoldPatches(outputs.cpu(), PATCH_SIZE, STRIDE)
            # plt.figure()
            # plt.imshow(image.squeeze().numpy()[25], cmap= 'gray')
            # plt.show()
        
        # load contours
        image_path = os.path.join(partition["train"][count], 'Basisplan', 'ProSeg')
        imageee, _, labels, _, _, _, _, info = multi_loader.extract_basisplan_and_fractions(folder_path=image_path,masks = STRUCTURES, size = IMAGE_SIZE, equalization_mode = None, folder='ProSeg')
        labels = labels.flip(1)
        imageee = imageee.flip(1)
        
        del image_path
        
        # ---- Segmentation pass ----
        outputs = (seg_model(image.unsqueeze(0).to(device).float()))
        seg_loss = dice_focal_loss(outputs.to(device), labels.to(device))
        seg_train_loss += seg_loss.item()
        
        # true_segmentation = plotter.plot_slice_with_contours(imageee.unsqueeze(0)[:, :, 25].cpu(), outputs.long().cpu()[0,:, 25], info['label_info'])
        # plt.figure()
        # plt.imshow(true_segmentation .squeeze())
        # plt.show()
        
        # ---- update models ----
        # update seg model with seg loss
        seg_optimizer.zero_grad()
        seg_loss.backward(retain_graph=True)  
        seg_optimizer.step()
    
        # update sr model with combined loss
        total_loss = 2*sr_loss + 0*seg_loss.detach()  
        total_train_loss += total_loss.item()
        sr_optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(sr_optimizer)
        scaler.update()
        
        del total_loss, seg_loss, sr_loss,  outputs, labels
        torch.cuda.empty_cache()
        count += 1
        
    seg_train_list.append(seg_train_loss/ len(dataloader))
    total_train_list.append(total_train_loss/ len(dataloader))
   
    
    # ---- validation losses ----
    with torch.no_grad():
        sr_model.eval()  # turn off training option for evaluation
        seg_model.eval()
        index = 0
        for inputs, labels in tqdm(valid_dataloader, position=0):
            inputs, labels = inputs.squeeze(0).squeeze(0).unsqueeze(1).to(device), labels.squeeze(0).squeeze(0).unsqueeze(1).to(device)
            with autocast():  # Mixed precision
                outputs = sr_model(inputs)  # forward pass
                pixel_loss = 1 * mae(outputs, labels)
                perc_loss = 0.2 * VGG19(outputs, labels)
                sr_loss = pixel_loss  + perc_loss
               
            # fold patches to 3D image
            image = multi_utils.FoldPatches(outputs.cpu(), PATCH_SIZE, STRIDE)
            
            # load contours
            image_path = os.path.join(partition["validation"][index], 'Basisplan', 'ProSeg')
            _, _, labels, _, _, _, _, info = multi_loader.extract_basisplan_and_fractions(folder_path=image_path, masks = STRUCTURES, size = IMAGE_SIZE, equalization_mode = None, folder='ProSeg')
            labels = labels.flip(1)
            
            # ---- Segmentation pass ----
            #with autocast():  # Mixed precision
            outputs = seg_model(image.unsqueeze(0).to(device).float())
            seg_loss = dice_focal_loss(outputs, labels.to(device))
            seg_valid_loss += seg_loss.item()
            total_loss = 2*sr_loss + seg_loss.detach()  
            total_valid_loss += total_loss.item()
            #del total_loss, seg_loss, sr_loss,  outputs, labels
            
            # save progress images 
            if index < 3:  # Limit to 3 validation samples
                # Original slice
                sr_image = image.unsqueeze(0)[:, :, 25].cpu()
                sr_image = sr_image[:, 0].numpy()*255
                sr_image = sr_image[0].astype(np.uint8)
                sr_image = cv.cvtColor(sr_image, cv.COLOR_GRAY2RGB)
                slice_image = torch.from_numpy(sr_image)  
                images.append(slice_image.unsqueeze(0))

                # Predicted segmentation
                pred_segmentation = plotter.plot_slice_with_contours(image.unsqueeze(0)[:, :, 25].cpu(), outputs.long().cpu()[0,:, 25], info['label_info'])
                images.append(pred_segmentation)
                plt.figure()
                plt.imshow(pred_segmentation.squeeze())
                plt.show()
     
    
                # True segmentation
                true_segmentation = plotter.plot_slice_with_contours(image.unsqueeze(0)[:, :, 25].cpu(), labels[:, 25].cpu(), info['label_info'])
                images.append(true_segmentation)

                
            index += 1

        # Stack images and create a 3x3 grid
        images_stacked = torch.cat(images).permute(0,3,1,2)
        img_grid = make_grid(images_stacked, nrow=3, padding=12, pad_value=-1)
        plt.imsave(image_path_seg / f"epoch_{epoch}.png", img_grid.permute(1, 2, 0).numpy())
        
        sr_model.train()  # turn training back on
        seg_model.train()  # turn training back on 
        del total_loss, seg_loss, sr_loss,  outputs, labels, inputs, pixel_loss, perc_loss, image_path         
        torch.cuda.empty_cache()
        
        
    seg_valid_list.append(seg_valid_loss/ len(valid_dataloader))
    total_valid_list.append(total_valid_loss/ len(valid_dataloader))
    print("Loss/train: "+str(seg_train_loss / len(dataloader))+ ' Epoch: '+ str(epoch))
    print("Loss/validation: "+str(seg_valid_loss / len(valid_dataloader))+ ' Epoch: '+ str(epoch))
    
    if (seg_valid_loss / len(valid_dataloader)) < minimum_valid_loss or epoch<= 9:
        epochs_no_improve = 0
        best_epoch = epoch
        minimum_valid_loss = seg_valid_loss / len(valid_dataloader)
        best_weights_seg = seg_model.state_dict()
        best_weights_sr = sr_model.state_dict()
        
    else:
        epochs_no_improve += 1
        
    if epoch > 9 and epochs_no_improve >= tolerance:
        break
    else:
        continue  

#%% save results

# make  loss curves and save
plt.clf()
plt.plot(range(1, len(seg_train_list)+1), seg_train_list, label="Training loss")
plt.plot(range(1, len(seg_train_list)+1), seg_valid_list, label="Validation loss")
plt.xlabel('Number of epochs')
plt.ylabel('Dice Focal Loss')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.ylim(0, max(seg_valid_list) + max(seg_valid_list)*0.1)
plt.legend()
plot_name = SEG_DIR / "loss_curves.png"
plt.savefig(plot_name, dpi=200)

plt.clf()
plt.plot(range(1, len(total_train_list)+1), total_train_list, label="Training loss")
plt.plot(range(1, len(total_train_list)+1), total_valid_list, label="Validation loss")
plt.xlabel('Number of epochs')
plt.ylabel('MAE and VGG19 Loss')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.ylim(0, max(total_valid_list) + max(total_valid_list)*0.1)
plt.legend()
plot_name = SR_DIR / "loss_curves.png"
plt.savefig(plot_name, dpi=200)

# save excel of losses
np.savetxt(SEG_DIR / "train_losses.csv", seg_train_list, delimiter=",")
np.savetxt(SR_DIR / "train_losses.csv", total_train_list, delimiter=",")
np.savetxt(SEG_DIR / "validation_losses.csv", seg_valid_list, delimiter=",")
np.savetxt(SR_DIR / "validation_losses.csv", total_valid_list, delimiter=",")  

# save weights
torch.save(best_weights_seg, SEG_DIR / f"seg_net_{best_epoch}.pth") 
torch.save(best_weights_sr, SR_DIR / f"sr_net_{best_epoch}.pth") 


#%% Test scriptjes 

# image = image.squeeze(0)
# plt.figure()
# plt.imshow(image.detach().numpy()[30], cmap = 'gray')  
# plt.show()  

#torch.cuda.memory_allocated()

# for name, param in seg_model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}2: {param.grad.norm()}")  # Check gradient norms
#         break