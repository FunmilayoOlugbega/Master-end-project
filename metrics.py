# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:15:24 2024

@author: Gast
"""
import pydicom                       
import pydicom.data                 
import matplotlib.pyplot as plt   
import numpy as np  
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pydicom.pixel_data_handlers import apply_modality_lut
from pathlib import Path
from scipy import signal
from matplotlib.pyplot import savefig
import piq
from torch.utils.data import DataLoader
import pandas as pd

def normalize(image):
    """Normalize images in list by multiplying with 95th percentile
    img_list: list
        list of numpy.ndarrays of 3D images
    """

    flat_img = image.flatten()
    max_val = np.max(flat_img)
    if max_val > 0:
        norm_img = image / max_val
    else:
        norm_img = image * 0
    return norm_img, max_val

def downsampling(img, sampling_factor, a):
    """Downsamples the images in a list with the specified sampling factor
    ----------
    image_list: list
        list of prostate images with a high resolution
    sampling_factor: int
        factor with which the k-space is reduced
    """
    x, y, z = img.shape
    center_x, center_y, center_z = x // 2, y // 2, z // 2
    kx_crop, ky_crop, kz_crop = round(x / sampling_factor / 2), round(y / sampling_factor / 2), round(z / sampling_factor / 2)

    # Fourier transform
    dft_shift = np.fft.fftshift(np.fft.fftn(img))
    fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
    # Hanning filter
    A, B, C = np.ix_(signal.windows.tukey(kx_crop * 2, alpha=a), signal.windows.tukey(ky_crop * 2, alpha=a), signal.windows.tukey(kz_crop * 2, alpha=a))
    window = A * B * C
    fft_crop = fft_shift * window
    # Zero padding
    pad_width = ((center_x - kx_crop, center_x - kx_crop),(center_y - ky_crop, center_y - ky_crop),(center_z - kz_crop, center_z - kz_crop))
    result = np.pad(fft_crop , pad_width, mode='constant')
    # Inverse Fourier
    fft_ifft_shift = np.fft.ifftshift(result)
    image_then = np.fft.ifftn(fft_ifft_shift)
    image_then = np.abs(image_then)
    
    return image_then[1:-1,:,:]

        
# Path to images
f_name = r"D:\Funmilayo_data\Anonimised data Julian\PAT6\Basisplan\MR"

# Form image
dicom_set = []
for root, _, filenames in os.walk(f_name):
    for filename in filenames:
        dcm_path = Path(root, filename)
        dicom = pydicom.dcmread(dcm_path, force=True)
        dicom_set.append(dicom)
                        
dicom_set.sort(key=lambda x: float(x.ImagePositionPatient[2]))
images = []
for dicom in dicom_set:
    hu = apply_modality_lut(dicom.pixel_array, dicom)
    images.append(hu)
img = np.asarray(images)[:,86:214,103:231]

img2 = torch.from_numpy(normalize(downsampling(img, 2, 0.3))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img3 = torch.from_numpy(normalize(downsampling(img, 2, 0.6))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img4 = torch.from_numpy(normalize(downsampling(img, 2, 0.9))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img5 = torch.from_numpy(normalize(downsampling(img, 2, 1))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img = torch.from_numpy(normalize(img[1:-1,:,:])[0])[140,:,:].unsqueeze(0).unsqueeze(0)

# img2 = torch.from_numpy(normalize(low_img)[0]).unsqueeze(0).unsqueeze(0)
# img3 =  torch.from_numpy(normalize(pred_img)[0]).unsqueeze(0).unsqueeze(0)
# img = torch.from_numpy(normalize(high_img)[0]).unsqueeze(0).unsqueeze(0)
images = [img2, img3, img4, img5]
df = []
for i in images:
    psnr = piq.psnr(img, i, data_range=1., reduction='none').item()
    ssim = piq.ssim(img, i, data_range=1.).item()
    gmsd = piq.gmsd(img, i, data_range=1., reduction='none').item()
    vif = piq.vif_p(img, i, data_range=1.).item()
    dists = piq.DISTS(reduction='none', mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])(img, i).item()
    haar = piq.haarpsi(img, i, data_range=1., reduction='none').item()
    df.append({"psnr":psnr, "ssim":ssim,"gmsd":gmsd,
                "vif":vif,"dists":dists, "haar":haar})
    
df_x = pd.DataFrame(df)

patch = normalize(pred_img)[0][0:30,98:128]
plt.figure()
plt.imshow(patch, cmap = 'gray')  
plt.show() 

y = patch[12,:]
x = np.arange(0,len(y))
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.plot(x, y, color ="red") 
plt.show()

def extract_segment(arr):
    max_index = np.argmax(arr)

    for i in range(max_index+1, len(arr)):
        if arr[i] > arr[i - 1]:
            end_index = i
            break
    segment = arr[max_index:end_index]

    
    return segment

def find_percentile(arr, percentile):
    percentile_value = np.percentile(arr, percentile)
    print(f"{percentile}th percentile value: {percentile_value}")
    closest_index = np.argmin(np.abs(arr - percentile_value))
    return closest_index

small_y = extract_segment(y)
edge_response = find_percentile(small_y, 10)-find_percentile(small_y, 90)#-find_percentile(small_y, 10)#-np.percentile(y, 10)
print(edge_response)

