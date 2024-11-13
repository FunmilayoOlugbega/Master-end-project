# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:43:21 2024

@author: Gast
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from pydicom.pixel_data_handlers import apply_modality_lut
from pathlib import Path
from scipy import signal
from torchvision.models import vgg19, VGG19_Weights
from torch.autograd import Variable
from math import exp
import pydicom                       
import pydicom.data                   
import numpy as np  
from matplotlib import pyplot as plt
import os

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProstateDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.
    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of the ROI around the prostate
    patch_size : int
        dimensions of the patches
    patch_stride : int
        stride that is used when forming patches
    """

    
    def __init__(self, paths, img_size, patch_size, patch_stride):
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        
        self.images = self.load_images(paths, 'Basisplan')
        self.high_res_list = self.normalize(self.images)[0]
        self.low_res_list = self.downsampling(self.high_res_list, 2)
        self.high_res_patches = self.create_patches(self.crop(self.high_res_list)) # augmentatie weggelaten 
        self.low_res_patches = self.create_patches(self.low_res_list)
        self.total_images = len(paths)
        
 
    def load_images(self, paths, folder):
        """Make 3D volumes of DICOM slices and save images to list
        Parameters
        ----------
        paths : list[Path]
            paths to the patient data
        """
        images = []
        # load dicom images
        for path in paths:
            path =  os.path.join(path, folder, 'MR')
            dicom_set = []
            for root, _, filenames in os.walk(path): 
                for filename in filenames:
                    dcm_path = Path(root, filename)
                    dicom = pydicom.dcmread(dcm_path, force=True)
                    dicom_set.append(dicom)
            # sort slices in the right order        
            dicom_set.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            slice_sort = []
            for dicom in dicom_set:
                hu = apply_modality_lut(dicom.pixel_array, dicom)
                slice_sort.append(hu)        
            img = np.asarray(slice_sort)
            # crop images to specified shape
            x, y, z = img.shape
            center_x, center_y, center_z = x // 2, y // 2, z // 2
            kx_crop, ky_crop, kz_crop = self.img_size[0] // 2, self.img_size[1] // 2, self.img_size[2] // 2
            cropped_img = img[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
            images.append(cropped_img)
            
        return images

    def downsampling(self, image_list, sampling_factor):
        """Downsamples the images in a list with the specified sampling factor
        ----------
        image_list: list
            list of prostate images with a high resolution
        sampling_factor: int
            factor with which the k-space is reduced
        """
        low_res_images = []
        x, y, z = self.img_size
        center_x, center_y, center_z = x // 2, y // 2, z // 2
        kx_crop, ky_crop, kz_crop = round(x / sampling_factor / 2), round(y / sampling_factor / 2), round(z / sampling_factor / 2)
        for img in image_list:
            # Fourier transform
            dft_shift = np.fft.fftshift(np.fft.fftn(img))
            fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
            # Hanning filter
            A, B, C = np.ix_(signal.windows.tukey(kx_crop * 2, alpha=0.3), signal.windows.tukey(ky_crop * 2, alpha=0.3), signal.windows.tukey(kz_crop * 2, alpha=0.3))
            window = A * B * C
            fft_crop = fft_shift * window
            # Zero padding
            pad_width = ((center_x - kx_crop, center_x - kx_crop),(center_y - ky_crop, center_y - ky_crop),(center_z - kz_crop, center_z - kz_crop))
            result = np.pad(fft_crop , pad_width, mode='constant')
            # Inverse Fourier
            fft_ifft_shift = np.fft.ifftshift(result)
            image_then = np.fft.ifftn(fft_ifft_shift)
            image_then = np.abs(image_then)
            low_res_images.append(image_then[1:-1])
            
        return low_res_images
    
    def crop(self, image_list): 
        high_res_images = []
        for img in image_list:
            high_res_images.append(img[1:-1])
            
        return high_res_images

    def normalize(self, img_list):
        """Normalize images in list by multiplying with 95th percentile
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        norm_images = []
        percentages = []
        for image in img_list:
            flat_img = image.flatten()
            max_val = np.percentile(flat_img, 100)
            if max_val > 0:
                norm_img = image / max_val
            else:
                norm_img = image * 0
            norm_images.append(norm_img)
            percentages.append(max_val)
        return norm_images, percentages
 
    def augment(self, img_list):
        """Augments the images by flipping and rotation. 
        The augmented images are added to the list of input images
        ----------
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        augmented = []
        for image in img_list:
            image = image[1:-1,:,:]
            # flip in x and y diretion
            x_flip = np.flip(image, 1)
            y_flip = np.flip(image, 2)
            
            # rotate 90, 180 and 270 degrees
            rotate_90 = np.rot90(image, k=1, axes=(1,2))
            rotate_180 = np.rot90(image, k=2, axes=(1,2))
            rotate_270 = np.rot90(image, k=3, axes=(1,2))
            augmented.extend([image,x_flip,y_flip,rotate_90,rotate_180,rotate_270])
        #img_list = img_list+augmented
        
        return augmented
        
    def create_patches(self, img_list):
        """Turns images in list into patches with overlap
        ----------
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        # patch size and stride
        kc, kh = self.patch_size, self.patch_size
        dc, dh = self.patch_stride, self.patch_stride
        dataset = []
        # make patches
        for img in img_list:
            store = []
            for slice_ in range(img.shape[0]):
                image = self.img_transform(img[slice_,:,:].astype(np.float32))
                patch_img = image.unfold(1, kc, dc).unfold(2, kh, dh)
                patch_img = patch_img.reshape(1, -1, kc, kh)
                store.append(patch_img)
            store = torch.cat(tuple(store), dim=1)
            dataset.append(store)
        patches = torch.cat(tuple(dataset), dim=0)
        
        return patches.unsqueeze(0)

    def __len__(self):
        """Returns length of dataset"""
        return self.total_images

    def __getitem__(self, index):
        """Returns the high and low resolution patches for a given index.
        Parameters
        ----------
        index : int
            index of the image in dataset
        """
        low_res_patch = self.low_res_patches[:,index,:,:,:]
        high_res_patch = self.high_res_patches[:,index,:,:,:]
        
        return low_res_patch, high_res_patch
    



class MAELoss(nn.Module):
    """Loss function computed as the mean absolute error
    """

    def __init__(self):
        super(MAELoss, self).__init__()


    def forward(self, outputs, targets):
        """Calculates the mean absolute error (MAE) loss between predicted and true values.
        Parameters
        ---------
        outputs : torch.Tensor
            predictions of super resolution model
        outputs : torch.Tensor
            ground truth labels
            
        Returns
         -------
         float
             mean absolute error loss
        """
        mae = torch.nn.L1Loss()
        mae_loss = mae(targets, outputs)
        return mae_loss

class VGGLoss(nn.Module):
    """Perceptual/VGG loss based on the ReLu activation layers of the pre-trained VGG-19 network
    """ 
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval().to(device)
        self.loss = nn.L1Loss()
        self.trans = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(1)==1 else x)

    def forward(self, outputs, targets):
        """Calculates the perceptual loss between predicted and true values.
        Parameters
        ---------
        outputs : torch.Tensor
            predictions of super resolution model
        outputs : torch.Tensor
            ground truth labels
            
        Returns
         -------
         float
             perceptual loss
        """
        vgg_first = self.vgg(self.trans(outputs).to(device))
        vgg_second = self.vgg(self.trans(targets).to(device))
        perceptual_loss = self.loss(vgg_second, vgg_first)
        return perceptual_loss


    
    
def FoldPatches(patches, KERNEL_SIZE, STRIDE):
    """Fold the patches in the list back to the original image
    by averaging the overlapping voxel values
    ----------
    patches_list: list
        list of image patches
    """
    B, C, W, H = 1, 48, 128, 128
    kernel_size = KERNEL_SIZE
    stride = STRIDE

    # Create a tensor to store the reconstructed image and a weight tensor
    reconstructed_image = torch.zeros((B, C, W, H))
    weight_tensor = torch.zeros((B, C, W, H))

    # Calculate the number of patches in each dimension
    num_patches_x = (W - kernel_size) // stride + 1
    num_patches_y = (H - kernel_size) // stride + 1

    patch_idx = 0
    for k in range(C):
        for i in range(0, num_patches_y * stride, stride):
            for j in range(0, num_patches_x * stride, stride):
                reconstructed_image[:, k, i:i + kernel_size, j:j + kernel_size] += patches[patch_idx]
                weight_tensor[:, k, i:i + kernel_size, j:j + kernel_size] += 1
                patch_idx += 1
    # Normalize the reconstructed image by the weight tensor
    reconstructed_image /= weight_tensor

    return reconstructed_image


