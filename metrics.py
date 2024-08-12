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
import matplotlib.image as mpimg
from PIL import Image
import scipy.signal
from scipy.stats import linregress

#%% Functions

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

#%% Load data      

# Path to images
f_name = r"D:\Funmilayo_data\Anonimised data Julian\PAT1\Basisplan\MR"

# Load DICOM sequence
dicom_set = []
for root, _, filenames in os.walk(f_name):
    for filename in filenames:
        dcm_path = Path(root, filename)
        dicom = pydicom.dcmread(dcm_path, force=True)
        dicom_set.append(dicom)
        
# Sort images on DICOM metadata                        
dicom_set.sort(key=lambda x: float(x.ImagePositionPatient[2]))
images = []
for dicom in dicom_set:
    hu = apply_modality_lut(dicom.pixel_array, dicom)
    images.append(hu)
img = np.asarray(images)[144,86:214,103:231]
image = Image.fromarray(img)
if image.mode != 'RGB':
    img = image.convert('RGB')
# Save the image
# image.save(r"D:\Funmilayo_data\runs\img.png")

#%% Calculate metrics for different ways of downsampling

# Normalize the image and downsample in different ways
img2 = torch.from_numpy(normalize(downsampling(img, 2, 0.3))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img3 = torch.from_numpy(normalize(downsampling(img, 2, 0.6))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img4 = torch.from_numpy(normalize(downsampling(img, 2, 0.9))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img5 = torch.from_numpy(normalize(downsampling(img, 2, 1))[0])[140,:,:].unsqueeze(0).unsqueeze(0)
img = torch.from_numpy(normalize(img[1:-1,:,:])[0])[140,:,:].unsqueeze(0).unsqueeze(0)

images = [img2, img3, img4, img5]
df = []

# Calculate metrics for every image
for i in images:
    psnr = piq.psnr(img, i, data_range=1., reduction='none').item()
    ssim = piq.ssim(img, i, data_range=1.).item()
    gmsd = piq.gmsd(img, i, data_range=1., reduction='none').item()
    vif = piq.vif_p(img, i, data_range=1.).item()
    dists = piq.DISTS(reduction='none', mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])(img, i).item()
    haar = piq.haarpsi(img, i, data_range=1., reduction='none').item()
    df.append({"psnr":psnr, "ssim":ssim,"gmsd":gmsd,
                "vif":vif,"dists":dists, "haar":haar})
# Form dataframe    
df_x = pd.DataFrame(df)

#%% Calculate average metrics for images in folders

empty = pd.DataFrame(columns=['psnr', 'ssim', 'gmsd', 'vif', 'dists', 'haar'])

psnr = 0
ssim = 0
gmsd = 0
vif = 0
dists = 0
haar = 0
df = []

# Calculate average metrics for 10 saved images
for i in range(10):
    img = np.array(Image.open(rf"D:\Funmilayo_data\New folder{i}_b.png").convert('L'))
    img1 = np.array(Image.open(rf"D:\Funmilayo_data\model_weights_attention\img{i}.png").convert('L'))
    
    img1 = torch.from_numpy(normalize(img1)[0]).unsqueeze(0).unsqueeze(0)
    img = torch.from_numpy(normalize(img)[0]).unsqueeze(0).unsqueeze(0)
    
    psnr += piq.psnr(img, img1, data_range=1., reduction='none').item()
    ssim += piq.ssim(img, img1, data_range=1.).item()
    gmsd == piq.gmsd(img, img1, data_range=1., reduction='none').item()
    vif += piq.vif_p(img, img1, data_range=1.).item()
    dists += piq.DISTS(reduction='none', mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])(img, img1).item()
    haar += piq.haarpsi(img, img1, data_range=1., reduction='none').item()

# Fill in average score in df
df.append({"psnr":psnr/10, "ssim":ssim/10,"gmsd":gmsd/10,
            "vif":vif/10,"dists":dists/10, "haar":haar/10})
empty = pd.concat([empty, pd.DataFrame(df)], ignore_index=True)
        
#%% Determine edge sharpness

# Open images (no DICOM)     
img = np.array(Image.open(r"D:\Funmilayo_data\New folder2_b.png").convert('L'))
img1 = np.array(Image.open(r"D:\Funmilayo_data\model_weights64\img2.png").convert('L'))
img2 = np.array(Image.open(r"D:\Funmilayo_data\model_weights_attention\img2.png").convert('L'))
img3 = np.array(Image.open(r"D:\Funmilayo_data\model_weights2\img2.png").convert('L'))

# Normalize and take a patch of the image with a clear edge    
img_n = normalize(img)[0]
patch = img_n[95:130,115:150]#[95:130,115:150]#[0:30,187:216]#[95:130,60:95]
plt.figure()
plt.imshow(patch, cmap = 'gray')  
plt.show() 

# Take a line across the edge and plot the intenisty values
y = patch[15,:]
x = np.arange(0,len(y))
plt.title("Line graph") 
plt.xlabel("X axis") 
plt.ylabel("Y axis") 
plt.ylim((0,1))
plt.plot(x, y, color ="red") 
plt.show()

# Find peaks in the intensity 
#y_smooth = scipy.signal.savgol_filter(y, window_length=5, polyorder=3) # Smooth line plot if needed
dy = np.gradient(y, x)
max_slope_idx = np.argmax(dy)
region_width = 3  # Adjust based on the expected width of the linear region
start_idx = max(max_slope_idx - region_width, 0)
end_idx = min(max_slope_idx + region_width, len(x))

# Extract the linear region
x_linear = x[start_idx:end_idx]
y_linear = y[start_idx:end_idx]

# Fit a line to the linear region
slope, intercept, r_value, p_value, std_err = linregress(x_linear, y_linear)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Data')
#plt.plot(x, y_smooth, label='Smoothed Data', linewidth=2)
plt.plot(x_linear, y_linear, label='Linear Region', linewidth=2)
plt.plot(x_linear, intercept + slope * x_linear, label=f'Fitted Line (Slope = {slope:.2f})', linestyle='--', color='red')
plt.legend()
plt.xlabel('Pixel distance')
plt.ylabel('Intensity')
plt.ylim((0,1))
plt.title('Edge response')
plt.show()
print(f'Slope of the middle linear region: {slope:.2f}')