# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:43:21 2024

@author: Gast
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pydicom                       
import pydicom.data                   
import numpy as np  


class ProstateDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of the ROI around the prostate
    """

    def __init__(self, paths, img_size):
        self.img_size = img_size
        self.low_res_list = []
        self.high_res_list = []
        
        
        # Load images and save ROI to list
        for path in paths:
            img = pydicom.dcmread(path).pixel_array
            x, y, z  = img.shape
            center_x, center_y, center_z = x// 2, y // 2, z // 2
            kx_crop, ky_crop, kz_crop = round( self.img_size[0]/2), round( self.img_size[1]/2), round( self.img_size[2]/2)
            self.high_res_list.append(img[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop] ) 

        # Number of patients and slices in the dataset
        self.no_patients = len(self.high_res_list)
        self.no_slices = self.high_res_list[0].shape[0]
        
        # Transformation to tensor
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        

        # # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.high_res_list)
        self.train_data_std = np.std(self.high_res_list)
        self.norm_high = transforms.Normalize(self.train_data_mean, self.train_data_std)
        


        
    def downsampling(self, image_list, sampling_factor):
        """Downsamples the images in a list with the specified sampling factor
        ----------
        image_list: list
            list of prostate images with a high resolution
        sampling_factor: int
            factor with which the k-space is reduced
        """
        x, y, z  = self.img_size
        center_x, center_y, center_z = x// 2, y // 2, z // 2
        kx_crop, ky_crop, kz_crop = round(x/sampling_factor/2), round(y/sampling_factor/2), round(z/sampling_factor/2)
        for img in self.high_res_list:
            # Fourier transform
            dft_shift = np.fft.fftshift(np.fft.fftn(img))
            fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
            
            # Hanning filter
            #A,B,C = np.ix_(np.hanning(kx_crop*2), np.hanning(ky_crop*2), np.hanning(kz_crop*2))
            #window = A*B*C
            fft_crop = fft_shift#*window
            
            # Zero padding
            pad_width = ((center_x - kx_crop, center_x - kx_crop),
                          (center_y - ky_crop, center_y - ky_crop),
                          (center_z - kz_crop, center_z - kz_crop))
            result = np.pad(fft_crop , pad_width, mode='constant')
            
            # Inverse Fourier
            fft_ifft_shift = np.fft.ifftshift(result)
            imageThen = np.fft.ifftn(fft_ifft_shift)
            imageThen = np.abs(imageThen)
            self.low_res_list.append(imageThen)
        
        # standardise intensities based on mean and std deviation
        mean = np.mean(self.low_res_list)
        std = np.std(self.low_res_list)
        self.norm_low = transforms.Normalize(mean, std)

            
    def patches(self, img):
        """Turns image into patches of 64x64 with overlap of 14x14
        ----------
        img: nump.ndarray
            prostate image
        """
        kc, kh = 64, 64 # kernel size
        dc, dh = 50, 50 # stride
        
        # Pad to multiples of 32
        pad_img = nn.functional.pad(img,(img.size(1)%kh // 2, img.size(1)%kh // 2, img.size(0)%kc // 2, img.size(0)%kc // 2))
        
        # Make patches
        patch_img = pad_img.unfold(1,  kc, dc).unfold(2, kh, dh)
        return patch_img
            
    
        
    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices
    

    def __getitem__(self, index):
        """Returns the high and low resolution patches for a given index.
        Parameters
        ----------
        index : int
            index of the image in dataset
        """
        # Create list of low resolution images
        self.downsampling( self.high_res_list, 4)

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return  (self.patches(self.norm_high(self.img_transform(self.high_res_list[patient][the_slice, ...].astype(np.float32)))), self.patches(self.norm_low(self.img_transform(self.low_res_list[patient][the_slice, ...]))))


