import torch
from torch import nn, Tensor
from typing import Optional, List
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import DiceHelper, SurfaceDiceMetric
from monai.losses import DiceLoss as diceloss
from monai.losses import TverskyLoss as tverskyLoss
import segmentation_models_pytorch as smp

class RMSELoss():
    def __init__(self, reduction='mean'):
        self.criterion = nn.MSELoss(reduction=reduction)
    def __call__(self, y_pred, y_true, rings=None):
        if rings != None:
            y_pred = y_pred * rings
            y_true = y_true * rings
        loss = torch.sqrt(self.criterion(y_pred.float(), y_true.float()))
        return loss.unsqueeze(0).unsqueeze(0)

class AELoss():
    def __init__(self, reduction='mean'):
        self.criterion = nn.L1Loss(reduction=reduction)
    def __call__(self, y_pred, y_true, rings=None):
        if rings != None:
            y_pred = y_pred * rings
            y_true = y_true * rings
        loss = self.criterion(y_pred.float(), y_true.float())
        return loss.unsqueeze(0).unsqueeze(0)

class MSELoss():
    def __init__(self, reduction='mean'):
        self.criterion = nn.MSELoss(reduction=reduction)
    def __call__(self, y_pred, y_true):
        loss = self.criterion(y_pred.float(), y_true.float())
        return loss    
    

class HausdorffDistance():
    def __init__(self, directed=True, percentile=95, from_logits=True, mode="binary", num_classes=1):
        self.criterion = HausdorffDistanceMetric(include_background=False, directed=directed, percentile=percentile)
        self.from_logits = from_logits
     
    def __call__(self, y_pred, y_true, ring=None):
        threed = len(y_true.size())>3
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
    
        if not threed:
            y_pred = y_pred.squeeze(1)
            y_pred =  F.one_hot(y_pred.long(), 2).permute(0,3,1,2)
            y_true = F.one_hot(y_true.long(), 2).permute(0,3,1,2)

        else:
            y_pred = y_pred.squeeze(1)
            y_pred = F.one_hot(y_pred.long(), 2).permute(0,4,1,2,3)
            y_true = F.one_hot(y_true.long(), 2).permute(0,4,1,2,3)

        hausdorff = self.criterion(y_pred, y_true).squeeze(0)
        return hausdorff.unsqueeze(1)    

def calc_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 1, dims=None) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice_score

class DiceScore():
    def __init__(self, mode="binary", from_logits=True, num_classes=1):
        self.mode = mode
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.criterion =  DiceHelper(include_background=False, softmax = False, sigmoid=True, activate=True)

    def __call__(self, y_pred, y_true, ring=None):                
        bs = y_true.size(0)        
        y_true = y_true.unsqueeze(1)
        
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred) 
            y_pred = (y_pred > 0.5).float() 

        y_pred = y_pred.view(bs, -1)
        y_true = y_true.view(bs, -1)
        dims = (1)
        scores = calc_dice_score(y_pred, y_true.type_as(y_pred), dims=dims)
        return scores.unsqueeze(1)
    
class SurfaceDice():
    def __init__(self, mode="binary", from_logits=True, num_classes=1):
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.criterion =  SurfaceDiceMetric(class_thresholds=[2], include_background=False, distance_metric="euclidean")
    def __call__(self, y_pred, y_true, ring=None):                
        threed = len(y_true.size())>3
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
        
        if not threed:
            y_pred = y_pred.squeeze(1)
            y_pred =  F.one_hot(y_pred.long(), 2).permute(0,3,1,2)
            y_true = F.one_hot(y_true.long(), 2).permute(0,3,1,2)
        
        else:
            y_pred = y_pred.squeeze(1)
            y_pred = F.one_hot(y_pred.long(), 2).permute(0,4,1,2,3)
            y_true = F.one_hot(y_true.long(), 2).permute(0,4,1,2,3)
    
        surface_dice = self.criterion(y_pred, y_true).squeeze(0)
        return surface_dice.unsqueeze(1)
    
class RelativeVolumeDifference():
    def __init__(self, from_logits=True):
        self.from_logits = from_logits

    def __call__(self, y_pred, y_true, ring=None):  
        bs = y_true.size(0)                   
        y_true = y_true.unsqueeze(1)
        
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred) 
            y_pred = (y_pred > 0.5).float() 

        y_pred = y_pred.view(bs, -1)
        y_true = y_true.view(bs, -1)

        pred_volume = torch.sum(y_pred, dim=1)
        true_volume = torch.sum(y_true, dim=1)
        
        rvd = torch.where(
            true_volume == 0,
            torch.tensor(float("nan"), device=true_volume.device), 
            (pred_volume - true_volume) / true_volume
        )
        
        # Ensure rvd is at least 1-dimensional before unsqueezing
        if rvd.dim() == 0:
            rvd = rvd.unsqueeze(0)
            
        return rvd.unsqueeze(1)
    
class TverskyLoss(_Loss):
    def __init__(self ):
        super(TverskyLoss, self).__init__()
        self.criterion = tverskyLoss(include_background=False, sigmoid=True)
    
    def __call__(self, y_pred, y_true):
        y_true = y_true.unsqueeze(1)
        loss = self.criterion(y_pred.float(), y_true.float())
        return loss  
    
    
class DiceLoss(_Loss):
    def __init__(self ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.criterion = diceloss(include_background=False, sigmoid=True)
    

    def __call__(self, y_pred, y_true):
        y_true = y_true.unsqueeze(1)
        loss = self.criterion(y_pred.float(), y_true.float())
        return loss   


    
class BCELoss(nn.Module):
    """
    Args:
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels
    Examples:
        >>> criteria = WeightedCELoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, reduction: str='mean'):
        super(BCELoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean'), \
            f'reduction {reduction} does not exists'
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pred, y_true):
        y_true = y_true.unsqueeze(1)
        loss = self.criterion(y_pred.float(), y_true.float())
        return loss


class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are outputs (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.DL_criterion = DiceLoss()
        self.BCE_criterion = BCELoss()

    def forward(self, outputs, targets):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            logit predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels

        Returns
        -------
        float
            the sum of the dice loss and cross-entropy-loss
        """        
        DL = self.DL_criterion(outputs, targets)
        
        BCE = self.BCE_criterion(outputs, targets)

        return DL + BCE


class FocalLoss(nn.Module):
    """
    Args:
        reduction: the output type that can be selected from none, sum and mean
    Forward:
        logits: a prediction with torch tensor type and has shape of (B, N, W, H)
        labels: a label with torch tensor type and has shape of (B, W, H)
            where B is batch size, N is number of classes, W and H is the width and height of outputs and labels
    Examples:
        >>> criteria = WeightedCELoss()
        >>> outputs = model(images)
        >>> loss = criteria(outputs, labels)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25, reduction: str = "mean", eps: float = 1e-6, reduced_threshold: Optional[float] = None):
        super(FocalLoss, self).__init__()
        assert reduction in ('none', 'sum', 'mean'), \
            f'reduction {reduction} does not exists'
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.eps = eps
        self.reduced_threshold = reduced_threshold

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        logpt = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float(), reduction='none')
        pt = torch.exp(-logpt)

        # compute the loss
        if self.reduced_threshold is None:
            focal_term = (1.0 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term[pt < self.reduced_threshold] = 1
        
        loss = focal_term * logpt
        
        if self.alpha is not None:
            loss *= self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        if self.reduction == "mean":
           loss = loss.mean()
        if self.reduction == "sum":
           loss = loss.sum()
        if self.reduction == "batchwise_mean":
           loss = loss.sum(0)
        
        return loss

class DiceFocalLoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are outputs (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self, gamma=2, alpha=0.25):
        super(DiceFocalLoss, self).__init__()
        self.DL_criterion = DiceLoss()
        self.FL_criterion = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, outputs, targets):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            logit predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels

        Returns
        -------
        float
            the sum of the dice loss and cross-entropy-loss
        """        
        DL = self.DL_criterion(outputs, targets)
        
        FL = self.FL_criterion(outputs, targets)

        return DL + FL