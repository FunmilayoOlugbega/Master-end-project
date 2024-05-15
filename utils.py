# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:43:21 2024

@author: Gast
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pydicom.pixel_data_handlers import apply_modality_lut
from pathlib import Path
import pydicom                       
import pydicom.data                   
import numpy as np  
import os



class ProstateDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of the ROI around the prostate
    """

    def __init__(self, paths, img_size, patch_size, patch_stride):
        self.img_size = img_size
        self.low_res_list = []
        self.high_res_list = []
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        
        # Load images and save ROI to list
        for path in paths:
            dicom_set = []
            # Make 3D volume of DICOM slices
            for root, _, filenames in os.walk(path): 
                for filename in filenames:
                    dcm_path = Path(root, filename)
                    if dcm_path.suffix == ".dcm":
                        try:
                            dicom = pydicom.dcmread(dcm_path, force=True)
                        except IOError as e:
                            print(f"Can't import {dcm_path.stem}")
                        else:
                            hu = apply_modality_lut(dicom.pixel_array, dicom)
                            dicom_set.append(hu)
                
            img = np.asarray(dicom_set)
            # Crop image to specified shape
            x, y, z  = img.shape
            center_x, center_y, center_z = x// 2, y // 2, z // 2
            kx_crop, ky_crop, kz_crop = round( self.img_size[0]/2), round( self.img_size[1]/2), round( self.img_size[2]/2)
            self.high_res_list.append(img[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop] ) 

        # Number of patients and slices in the dataset
        self.no_patients = len(self.high_res_list)
        self.no_slices = self.high_res_list[0].shape[0]
        self.no_patches = ((self.img_size[1]+(self.patch_size-1))//self.patch_size)*((self.img_size[2]+(self.patch_size-1))//self.patch_size)

        
        # Transformation to tensor
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        

        # # standardise intensities based on mean and std deviation
        # self.train_data_mean = np.mean(self.high_res_list)
        # self.train_data_std = np.std(self.high_res_list)
        # self.norm_high = transforms.Normalize(self.train_data_mean, self.train_data_std)
        


        
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
        
        # # standardise intensities based on mean and std deviation
        # mean = np.mean(self.low_res_list)
        # std = np.std(self.low_res_list)
        # self.norm_low = transforms.Normalize(mean, std)

            
    def patches(self, img_list):
        """Turns images in list into patches of 16x16 with overlap of 0x0
        ----------
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        kc, kh = self.patch_size, self.patch_size # kernel size
        dc, dh = self.patch_stride, self.patch_stride # stride

        store = []
        for i in img_list:
            for j in range(i.shape[0]):
                # Pad to multiples of 16
                image = self.img_transform(i[j,:,:].astype(np.float32))
                pad_img = nn.functional.pad(image,(image.size(2)%kh // 2, image.size(2)%kh // 2, image.size(1)%kc // 2, image.size(1)%kc // 2))
                patch_img = pad_img.unfold(1,  kc, dc).unfold(2, kh, dh)
                patch_img = patch_img.reshape(1,-1,kc,kh)
                store.append(patch_img)
        patches = torch.cat(tuple(store), dim=1)
        
        return patches
    
    

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices*64 # 64 needs to be changed to the number of patches per slice
    

    def __getitem__(self, index):
        """Returns the high and low resolution patches for a given index.
        Parameters
        ----------
        index : int
            index of the image in dataset
        """
        # Create list of low resolution images
        self.downsampling( self.high_res_list, 4)
        
        return (self.patches(self.high_res_list)[:,index,:,:],  self.patches(self.low_res_list)[:,index,:,:])
        
        



class MSELoss(nn.Module):
    """Loss function computed as the mean squared error
    """

    def __init__(self):
        super(MSELoss, self).__init__()


    def forward(self, outputs, targets):
        """Calculates the mean squared error (MSE) loss between predicted and true values.
        Parameters
        ---------
        outputs : torch.Tensor
            predictions of super resolution model
        outputs : torch.Tensor
            ground truth labels
            
        Returns
         -------
         float
             mean squared error loss
        """
        mse = torch.nn.MSELoss()
        mse_loss = mse(targets, outputs)
        return mse_loss


