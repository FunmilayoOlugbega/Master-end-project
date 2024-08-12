# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:36:21 2024

@author: 20192059
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
import matplotlib.image as mpimg

#%% Load Data

# Path to images
f_name = r"D:\Funmilayo_data\Anonimised data Julian\PAT1\Basisplan\MR"

dicom_set = []
for root, _, filenames in os.walk(f_name):
    for filename in filenames:
        dcm_path = Path(root, filename)
        dicom = pydicom.dcmread(dcm_path, force=True)
        dicom_set.append(dicom)

# Sort images on DICOM image position metadata                        
dicom_set.sort(key=lambda x: float(x.ImagePositionPatient[2]))
images = []
for dicom in dicom_set:
    hu = apply_modality_lut(dicom.pixel_array, dicom)
    images.append(hu)
img3D = np.asarray(images)

#%% 2D 

# Visualize image in 2D  
img = img3D[140,94:222,103:231]
plt.figure()
plt.imshow(img, cmap = 'gray')  
plt.show()  

# Visualize the k-space in 2D
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
k_space = np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])+1e-9)
plt.figure()
plt.imshow(k_space, cmap = 'gray')  
plt.show()  

# Sampling of k-space
sampling_factor = 4
row, col = img.shape
center_row, center_col = row // 2, col // 2
k_crop = round(row/sampling_factor/2)

# Inverse fourier of downsampled k-space
fft_shift = dft_shift[center_row - k_crop:center_row + k_crop, center_col - k_crop:center_col + k_crop]

# Hanning filter
f_complex = fft_shift[:,:,0] +fft_shift[:,:,1]*1j
x, y = f_complex.shape
window = np.outer(np.hanning(x), np.hanning(y))
f_complex*= window
f_filtered = np.stack([np.real(f_complex), np.imag(f_complex)], axis=-1)

# Zero padding
zero_k = np.zeros(dft.shape)
zero_k[center_row - k_crop:center_row + k_crop, center_col - k_crop:center_col + k_crop] = f_filtered
han_img = 20*np.log(cv2.magnitude(zero_k[:,:,0],zero_k[:,:,1]))
plt.figure()
plt.imshow(han_img,  cmap = 'gray')  
plt.show()  

# Inverse Fourier
fft_ifft_shift = np.fft.ifftshift(zero_k)
imageThen = cv2.idft(fft_ifft_shift)

# Magnitude of the inverse fourier
imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
plt.figure()
plt.imshow(imageThen)  
plt.show()  

#%% 3D

# Visualize sice of 3D k-space
dft = np.fft.fftn(img3D)
dft_shift = np.fft.fftshift(dft)
k_space =  np.log(np.abs(dft_shift)+1e-9)
plt.figure()
plt.imshow(k_space[144,:,:], cmap = 'gray') 
#plt.savefig(r'\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\filename.png', format='png', transparent=True) 
plt.show()   

# Sampling of k-space
x, y, z  = x.shape
center_x, center_y, center_z = x// 2, y // 2, z // 2
sampling_factor = 2
kx_crop = round(x/sampling_factor/2)
ky_crop = round(y/sampling_factor/2)
kz_crop = round(z/sampling_factor/2)

# Inverse fourier of downsampled k-space
fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
k_space1 =  np.log(np.abs(fft_shift  )+1e-9)
plt.figure()
plt.imshow(k_space1[72,:,:], cmap = 'gray')  
plt.show() 
 
# Hanning filter
if  sampling_factor!=2:   
    A, B, C = np.ix_(signal.windows.tukey(kx_crop * 2, alpha=0.3), signal.windows.tukey(ky_crop * 2, alpha=0.3), signal.windows.tukey(kz_crop * 2, alpha=0.3))
    window = A * B * C
    fft_crop = fft_shift * window
else:
    fft_crop = fft_shift
    
# Zero padding
pad_width = ((center_x - kx_crop, center_x - kx_crop),
              (center_y - ky_crop, center_y - ky_crop),
              (center_z - kz_crop, center_z - kz_crop))
result = np.pad(fft_crop , pad_width, mode='constant')

# Plot K-space after zero padding
k_space =  np.log(np.abs(result )+1e-9)
plt.figure()
plt.imshow(k_space[144,:,:], cmap = 'gray', vmin=0, vmax=k_space1.max())  
#plt.savefig(r'\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\filename.png', format='png', transparent=True)
plt.show()  

# Inverse Fourier
fft_ifft_shift = np.fft.ifftshift(result)
imageThen = np.fft.ifftn(fft_ifft_shift)
imageThen = np.abs(imageThen)

# Visualize downsampled image
plt.figure()
plt.imshow(imageThen[140,:,:], cmap = 'gray')  
plt.show()  

plt.figure()
plt.imshow(img[140,:,:], cmap = 'gray')  
plt.show()  

