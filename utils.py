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
from torchvision.models import vgg19
from torch.autograd import Variable
from math import exp
import pydicom                       
import pydicom.data                   
import numpy as np  
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
        
        self.high_res_list = self.normalize(self.load_images(paths))[0]
        self.low_res_list = self.downsampling(self.high_res_list, 2)
        self.high_res_patches = self.create_patches(self.high_res_list)
        self.low_res_patches = self.create_patches(self.low_res_list)
        self.total_patches = self.high_res_patches.shape[1]
        
 
    def load_images(self, paths):
        """Make 3D volumes of DICOM slices and save images to list
        Parameters
        ----------
        paths : list[Path]
            paths to the patient data
        """
        images = []
        # load dicom images
        for path in paths:
            path =  os.path.join(path, 'Basisplan', 'MR')
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
            low_res_images.append(image_then)
            
        return low_res_images

    def normalize(self, img_list):
        """Normalize images in list by multiplying with 95th percentile
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        norm_images = []
        percentages = []
        for image in img_list:
            flat_img = image.flatten()
            max_val = np.percentile(flat_img, 95)
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
            # flip in x and y diretion
            x_flip = np.flip(image, 1)
            y_flip = np.flip(image, 2)
            
            # rotate 90, 180 and 270 degrees
            rotate_90 = np.rot90(image, k=1, axes=(1,2))
            rotate_180 = np.rot90(image, k=2, axes=(1,2))
            rotate_270 = np.rot90(image, k=3, axes=(1,2))
            augmented.extend([x_flip,y_flip,rotate_90,rotate_180,rotate_270])
        img_list = img_list+augmented
        
        return img_list
        
    def create_patches(self, img_list):
        """Turns images in list into patches with overlap
        ----------
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        # patch size and stride
        kc, kh = self.patch_size, self.patch_size
        dc, dh = self.patch_stride, self.patch_stride
        store = []
        # make patches
        for img in img_list:
            img = img[1:-1,:,:]
            for slice_ in range(img.shape[0]):
                image = self.img_transform(img[slice_,:,:].astype(np.float32))
                patch_img = image.unfold(1, kc, dc).unfold(2, kh, dh)
                patch_img = patch_img.reshape(1, -1, kc, kh)
                store.append(patch_img)
        patches = torch.cat(tuple(store), dim=1)
        
        return patches

    def __len__(self):
        """Returns length of dataset"""
        return self.total_patches

    def __getitem__(self, index):
        """Returns the high and low resolution patches for a given index.
        Parameters
        ----------
        index : int
            index of the image in dataset
        """
        low_res_patch = self.low_res_patches[:,index,:,:]
        high_res_patch = self.high_res_patches[:,index,:,:]
        
        return low_res_patch, high_res_patch
    

class ValidDataset(torch.utils.data.Dataset):
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

    
    def __init__(self, paths, img_size, patch_size, patch_stride, patient_nr, slice_nr):
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patient_nr = patient_nr
        self.slice_nr = int(slice_nr)
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        self.high_res_list = self.normalize(self.load_images(paths))[0]
        self.percentile = self.normalize(self.load_images(paths))[1]
        self.low_res_list = self.downsampling(self.high_res_list, 2)[1:-1,:,:]
        self.high_res_list = self.high_res_list[1:-1,:,:]
        self.high_res_patches = self.create_patches(self.high_res_list[self.slice_nr,:,:])
        self.low_res_patches = self.create_patches(self.low_res_list[self.slice_nr,:,:])
        self.total_patches = self.high_res_patches.shape[1]
        
 
    def load_images(self, paths):
        """Make 3D volumes of DICOM slices and save images to list
        Parameters
        ----------
        paths : list[Path]
            paths to the patient data
        """
        path =  os.path.join(paths[self.patient_nr], 'Basisplan', 'MR')
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

        return cropped_img
    
    def augment(self, img_list):
        """Augments the images by flipping and rotation. 
        The augmented images are added to the list of input images
        ----------
        img_list: list
            list of numpy.ndarrays of 3D images
        """
        augmented = []
        for image in img_list:
            # flip in x and y diretion
            x_flip = np.flip(image, 1)
            y_flip = np.flip(image, 2)
            
            # rotate 90, 180 and 270 degrees
            rotate_90 = np.rot90(image, k=1, axes=(1,2))
            rotate_180 = np.rot90(image, k=2, axes=(1,2))
            rotate_270 = np.rot90(image, k=3, axes=(1,2))
            augmented.extend([x_flip,y_flip,rotate_90,rotate_180,rotate_270])
        img_list = img_list+augmented

    def downsampling(self, img, sampling_factor):
        """Downsamples the images in a list with the specified sampling factor
        ----------
        image_list: list
            list of prostate images with a high resolution
        sampling_factor: int
            factor with which the k-space is reduced
        """

        x, y, z = self.img_size
        center_x, center_y, center_z = x // 2, y // 2, z // 2
        kx_crop, ky_crop, kz_crop = round(x / sampling_factor / 2), round(y / sampling_factor / 2), round(z / sampling_factor / 2)
        # Fourier transform
        dft_shift = np.fft.fftshift(np.fft.fftn(img))
        fft_shift = dft_shift[center_x - kx_crop:center_x + kx_crop, center_y - ky_crop:center_y + ky_crop, center_z - kz_crop:center_z + kz_crop]
        # Hanning filter
        A, B, C = np.ix_(signal.windows.tukey(kx_crop * 2, alpha=0.3), signal.windows.tukey(ky_crop * 2, alpha=0.3), signal.windows.tukey(kz_crop * 2, alpha=0.3))
        window = A * B * C
        fft_crop = fft_shift * window
        # Zero padding
        pad_width = ((center_x - kx_crop, center_x - kx_crop),(center_y - ky_crop, center_y - ky_crop),(center_z - kz_crop, center_z - kz_crop))
        result = np.pad(fft_crop, pad_width, mode='constant')
        # Inverse Fourier
        fft_ifft_shift = np.fft.ifftshift(result)
        image_then = np.fft.ifftn(fft_ifft_shift)
        image_then = np.abs(image_then)
            
        return image_then
    
    
    def normalize(self, image):
        """Normalize images by multiplying with 95th percentile
        img_list: list
            list of numpy.ndarrays of 3D images
        """

        flat_img = image.flatten()
        max_val = np.percentile(flat_img, 95)
        if max_val > 0:
            norm_img = image / max_val
        else:
            norm_img = image * 0

        return norm_img, max_val 

    def create_patches(self, img):
        """Turns images in list into patches with overlap 
        ----------
        img: numpy.ndarray
            numpy.ndarray of 3D image
        """
        kc, kh = self.patch_size, self.patch_size 
        dc, dh = self.patch_stride, self.patch_stride #
        image = self.img_transform(img.astype(np.float32))
        patch_img = image.unfold(1, kc, dc).unfold(2, kh, dh)
        patch_img = patch_img.reshape(1,-1,kc,kh)
        
        return patch_img

    def __len__(self):
        """Returns length of dataset"""
        return self.total_patches

    def __getitem__(self, index):
        """Returns the high and low resolution patches for a given index.
        Parameters
        ----------
        index : int
            index of the image in dataset
        """
        low_res_patch = self.low_res_patches[:,index,:,:]
        high_res_patch = self.high_res_patches[:,index,:,:]
        
        return low_res_patch, high_res_patch, self.percentile
    


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
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
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



class SSIMLoss(nn.Module):
    """
    Loss function computed as the Structural Similarity Index Measure (SSIM) between images.
    """
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def gaussian(self, window_size, sigma):
        """
        Creates a 1D Gaussian kernel.

        Parameters
        ----------
        window_size : int
            Size of the Gaussian window.
        sigma : float
            Standard deviation of the Gaussian distribution.
        
        Returns
        -------
        torch.Tensor
            1D Gaussian kernel.
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size):
        """
        Creates a 2D Gaussian window.
     
        Parameters
        ----------
        window_size : int
            Size of the Gaussian window.
        
        Returns
        -------
        torch.Tensor
            2D Gaussian window.
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
        return window
    
    def loss(self, img1, img2, window_size=11):
        """
        Computes the SSIM loss between two images.
        
        Parameters
        ----------
        img1 : torch.Tensor
        Predicted image
        img2 : torch.Tensor
        Ground truth image
        window_size : int, optional
        Size of the Gaussian window. Default is 11.
        
        Returns
        -------
        torch.Tensor
        SSIM loss.
        """
        window = self.create_window(window_size).to(device)
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = 1)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = 1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 1) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = 1) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = 1-((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


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
        ssim_loss = self.loss(targets, outputs)
        return ssim_loss    


class EdgeLoss(nn.Module):
    """
    Class to calculate the edge loss using Sobel edge detection.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x, self.sobel_y = self.create_sobel_filters()

    def create_sobel_filters(self):
        """
        Create Sobel filters for edge detection in the x and y directions.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Sobel filters for x and y directions.
        """
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        sobel_x = nn.Parameter(data=sobel_x, requires_grad=False)
        sobel_y = nn.Parameter(data=sobel_y, requires_grad=False)

        return sobel_x, sobel_y

    def apply_sobel_filter(self, img, sobel_x, sobel_y):
        """
        Apply Sobel filters to an image to get edge maps.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.
        sobel_x : torch.Tensor
            Sobel filter for the x direction.
        sobel_y : torch.Tensor
            Sobel filter for the y direction.

        Returns
        -------
        torch.Tensor
            Edge map of the input image.
        """
        edge_x = F.conv2d(img, sobel_x, padding=1)
        edge_y = F.conv2d(img, sobel_y, padding=1)
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge_map

    def forward(self, pred, target):
        """
        Calculate the edge loss between predicted and ground truth images.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted image tensor.
        target : torch.Tensor
            Ground truth image tensor.

        Returns
        -------
        torch.Tensor
            Edge loss.
        """

        pred_edges = self.apply_sobel_filter(pred, self.sobel_x, self.sobel_y)
        target_edges = self.apply_sobel_filter(target, self.sobel_x, self.sobel_y)
        mae = torch.nn.L1Loss()
        loss = mae(pred_edges, target_edges)
        return loss
    
    
def FoldPatches(patches_list, KERNEL_SIZE, STRIDE):
    """Fold the patches in the list back to the original image
    by averaging the overlapping voxel values
    ----------
    patches_list: list
        list of image patches
    """
    B, C, W, H = 1, 1, 128, 128
    kernel_size = KERNEL_SIZE
    stride = STRIDE

    # Create a tensor to store the reconstructed image and a weight tensor
    reconstructed_image = torch.zeros((B, C, W, H))
    weight_tensor = torch.zeros((B, C, W, H))

    # Calculate the number of patches in each dimension
    num_patches_x = (W - kernel_size) // stride + 1
    num_patches_y = (H - kernel_size) // stride + 1

    patch_idx = 0
    for i in range(0, num_patches_y * stride, stride):
        for j in range(0, num_patches_x * stride, stride):
            reconstructed_image[:, :, i:i + kernel_size, j:j + kernel_size] += patches_list[patch_idx]
            weight_tensor[:, :, i:i + kernel_size, j:j + kernel_size] += 1
            patch_idx += 1
    # Normalize the reconstructed image by the weight tensor
    reconstructed_image /= weight_tensor

    return reconstructed_image

