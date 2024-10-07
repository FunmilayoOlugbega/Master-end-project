import torch
import torch.nn as nn
from MachineLearning.layers import *

class SegNet2D(nn.Module):
    """A representation for a unet

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        in_chs=1,
        out_chs=1,
        enc_chs=(64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64),
    ):
        super().__init__()
        self.encoder = Encoder2D((in_chs,)+enc_chs)
        self.decoder = Decoder2D(dec_chs)
        self.activation = nn.Sequential(nn.Conv2d(dec_chs[-1], out_chs, 1))

    def forward(self, x):
        """Performs the forward pass of the unet.

        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        logits = self.activation(out)
        return logits
                    
class DoseNet2D(nn.Module):
    """A representation for a unet

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        in_chs=7,
        out_chs=1,
        enc_chs=(64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64)
    ):
        super().__init__()
        self.encoder = Encoder2D((in_chs,)+enc_chs)
        self.decoder = Decoder2D(dec_chs)
        self.activation = ActivationDose2D(dec_chs[-1], out_chs)

    def forward(self, x):
        """Performs the forward pass of the unet.

        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        logits = self.activation(out)
        return logits.squeeze(1)
    
class SegNet3D(nn.Module):
    """A representation for a unet

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        in_chs=1,
        out_chs=1,
        enc_chs=(64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64)
    ):
        super().__init__()
        self.encoder = Encoder3D((in_chs,)+enc_chs)
        self.decoder = Decoder3D(dec_chs)
        self.activation = nn.Sequential(nn.Conv3d(dec_chs[-1], out_chs, 1))

    def forward(self, x):
        """Performs the forward pass of the unet.

        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        logits = self.activation(out)
        return logits               
                    
class DoseNet3D(nn.Module):
    """A representation for a unet

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        in_chs=7,
        out_chs=1,
        enc_chs=(64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64)
    ):
        super().__init__()
        self.encoder = Encoder3D((in_chs,)+enc_chs)
        self.decoder = Decoder3D(dec_chs)
        self.activation = ActivationDose3D(dec_chs[-1], out_chs)

    def forward(self, x):
        """Performs the forward pass of the unet.

        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        logits = self.activation(out)
        return logits.squeeze(1)
