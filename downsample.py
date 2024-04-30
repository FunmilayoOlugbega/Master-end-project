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


# Functions
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def hanning(window_shape):
    """
    Create a 3D Hanning window with the specified shape.
    """
    hanning_1d_x = np.hanning(window_shape[0])
    hanning_1d_y = np.hanning(window_shape[1])
    hanning_1d_z = np.hanning(window_shape[2])

    # Expand 1D Hanning windows to 3D
    hanning_3d_x = hanning_1d_x[:, np.newaxis, np.newaxis]
    hanning_3d_y = hanning_1d_y[np.newaxis, :, np.newaxis]
    hanning_3d_z = hanning_1d_z[np.newaxis, np.newaxis, :]

    # Create 3D Hanning window
    hanning_3d = hanning_3d_x * hanning_3d_y * hanning_3d_z

    return hanning_3d


# Path to images
f_name = r"C:\Users\20192059\Documents\MEP\Test data\samples-of-mr-images-1.0.0\samples-of-mr-images-1.0.0\E1154S7I.dcm" 

#%% 2D 

# Read image  
dataset = pydicom.dcmread(f_name)  
img = dataset.pixel_array[50,:,:]

# Visualize image in 2D  
plt.figure()
plt.imshow(img, cmap = plt.cm.bone)  
plt.show()  

# Visualize the k-space in 2D
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
k_space = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.figure()
plt.imshow(k_space, cmap = 'gray')  
plt.show()  

# # Inverse fourier to get original image back
# f_ishift = np.fft.ifftshift(dft_shift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# Sampling of k-space
sampling_factor = 2
row, col = img.shape
center_row, center_col = row // 2, col // 2
k_crop = round(row/sampling_factor/2)

# Inverse fourier of downsampled k-space
fft_shift = dft_shift[center_row - k_crop:center_row + k_crop, center_col - k_crop:center_col + k_crop]
# downsampled = 20*np.log(cv2.magnitude(fft_shift[:,:,0],fft_shift[:,:,1]))
# plt.figure()
# plt.imshow(downsampled, cmap = 'gray')  
# plt.show() 

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

## bicubic interpolation
#imageThen = cv2.resize(imageThen, (row, col), interpolation=cv2.INTER_CUBIC)

plt.figure()
plt.imshow(han_img,  cmap = 'gray')  
plt.show()  

# Inverse Fourier
fft_ifft_shift = np.fft.ifftshift(zero_k)
imageThen = cv2.idft(fft_ifft_shift)

# Magnitude of the inverse fourier
imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])
#print(dataset.PixelSpacing[0]*sampling_factor)

plt.figure()
plt.imshow(imageThen, cmap = plt.cm.bone)  
plt.show()  

#%% 3D

# Read image  
dataset = pydicom.dcmread(f_name)  
img = dataset.pixel_array

# k-space in 3D
dft = np.fft.fftn(img)
dft_shift = np.fft.fftshift(dft)

# Sampling of k-space
x, y, z  = img.shape
center_x, center_y, center_z = x// 2, y // 2, z // 2
sampling_factor = 2
kx_crop = round(x/sampling_factor/2)
ky_crop = round(y/sampling_factor/2)
kz_crop = round(z/sampling_factor/2)

# Inverse fourier of downsampled k-space
fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]

# Hanning filter
#x, y, z = fft_shift.shape
#A,B,C = np.ix_(np.hanning(x), np.hanning(y), np.hanning(z))
window = hanning(fft_shift.shape)#A*B*C
fft_crop = fft_shift*window

# Zero padding
pad_width = ((center_x - kx_crop, center_x - kx_crop),
              (center_y - ky_crop, center_y - ky_crop),
              (center_z - kz_crop, center_z - kz_crop))
result = np.pad(fft_crop , pad_width, mode='constant')

# Inverse Fourier
fft_ifft_shift = np.fft.ifftshift(result)
imageThen = np.fft.ifftn(fft_ifft_shift)
imageThen = np.abs(imageThen)

plt.figure()
plt.imshow(imageThen[50, :, :], cmap = plt.cm.bone)  
plt.show()  