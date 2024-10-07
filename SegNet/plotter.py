import torch
import cv2 as cv
import numpy as np        
from PIL import Image

def plot_select_slice_with_contours(images, labels, rings, label_info, current_fraction, current_slice, current_view = "Transverse", labels_to_display=["Skin", "CTV", "BLADDER", "RECTUM"]):
    if current_view == "Transverse":
        img = images[current_fraction, current_slice, :, :].numpy()
        if np.max(img > 0):
            img = (img/np.max(img) * 255)
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        overlay = img.copy()
        for label_num, info in label_info.items():
            if info[0] in labels_to_display and info[0]!="RING":
                mask = torch.where(labels[current_fraction, current_slice, :, :]==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)
        
            elif info[0] in labels_to_display and info[0]=="RING":
                mask = torch.where(rings[current_fraction, current_slice, :, :]==1, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)

        img = cv.addWeighted(overlay, 0.3, img, 0.7, 0)

    elif current_view == "Sagittal":
        img = images[current_fraction, :, :, current_slice].numpy()
        if np.max(img > 0):
            img = (img/np.max(img) * 255)
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        overlay = img.copy()
        for label_num, info in label_info.items():
            if info[0] in labels_to_display and info[0]!="RING":
                mask = torch.where(labels[current_fraction, :, :, current_slice]==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)
        
            elif info[0] in labels_to_display and info[0]=="RING":
                mask = torch.where(rings[current_fraction, :, :, current_slice]==1, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)

        img = cv.addWeighted(overlay, 0.3, img, 0.7, 0)
        
    elif current_view == "Coronal":
        img = images[current_fraction, :, current_slice, :].numpy()
        if np.max(img > 0):
            img = (img/np.max(img) * 255)
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)            
        
        overlay = img.copy()
        for label_num, info in label_info.items():
            if info[0] in labels_to_display and info[0]!="RING":
                mask = torch.where(labels[current_fraction, :, current_slice, :]==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)
        
            elif info[0] in labels_to_display and info[0]=="RING":
                mask = torch.where(rings[current_fraction, :, current_slice, :]==1, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
                color = info[1]
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(img, contours, -1, color, 1, 1)
                cv.drawContours(overlay, contours, -1, color, cv.FILLED)

        img = cv.addWeighted(overlay, 0.3, img, 0.7, 0)                        
    
    img = Image.fromarray(img)
    
    return img

def plot_slice_with_contours(images, labels, label_info):
    images = images[:, 0].numpy()*255
    drawn_images = []
    for image in range(images.shape[0]):
        img = images[image].astype(np.uint8)
        label = labels[image]
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        overlay = img.copy()
        for label_num, info in label_info.items():
            mask = torch.where(label==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
            color = (0,255,0)#info[1]
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img, contours, -1, color, 1, 1)
            cv.drawContours(overlay, contours, -1, color, cv.FILLED)

        img = cv.addWeighted(overlay, 0.3, img, 0.7, 0)

        img = torch.from_numpy(img)
        drawn_images.append(img)
    drawn_images = torch.stack(drawn_images, dim=0)
    return drawn_images

def plot_slice_with_dose(labels, doses, label_info):
    percentages = {
        102: (156, 0, 0),       # Dark Red
        100: (255, 0, 0),       # Red
        98: (255, 165, 0),      # Light Orange
        95: (255, 255, 0),      # Yellow
        90: (50, 205, 50),      # Lime
        80: (0, 128, 0),        # Dark green
        70: (0, 255, 255),      # Cyan
        50: (0, 0, 255),        # Light Blue
        30: (0, 0, 128)         # Dark Blue
    }
    drawn_images = []
    for i in range(labels.shape[0]):
        img = torch.zeros_like(labels[i]).numpy().astype(np.uint8)
        label = labels[i]
        dose = doses[i]
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for label_num, info in label_info.items():
            mask = torch.where(label==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
            color = info[1]
            img[mask==1]=color
            
        for percentage, color in percentages.items():
            mask = torch.where(dose>int(percentage/100*3625), torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img, contours, -1, color, 1, 1)   
            
        img = torch.from_numpy(img)
        drawn_images.append(img)
    drawn_images = torch.stack(drawn_images, dim=0)
    return drawn_images
