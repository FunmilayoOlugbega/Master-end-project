# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:48:41 2024

@author: Gast
"""

import torch
import torch.nn as nn


# class Block(nn.Module):
#     """Convolutional building block of the unet. A block consists of
#     two convolutional layers with ReLU activations
#     Parameters
#     ----------
#     in_ch : int
#         number of input channels to the block
#     out_ch : int
#         number of output channels of the block
#     """

#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same', bias='True'),
#                                   nn.ReLU(),
#                                   nn.Conv2d(out_ch, out_ch, 3, padding='same', bias='True'),
#                                   nn.ReLU())

#     def forward(self, x):
#         """Performs the forward pass of the block.
#         Parameters
#         ----------
#         x : torch.Tensor
#             the input to the block
#         Returns
#         -------
#         x : torch.Tensor
#             the output of the forward pass
#         """
#         return self.conv(x)


# class AttentionBlock(nn.Module):
#     """The attention gate of the unet.
#     Parameters
#     ----------
#     in_c : tuple
#         holds the number of input channels of each block in the decoder
#     out_c : tuple
#         holds the number of output channels of each block in the decoder
#     """
#     def __init__(self, in_c, out_c):
#         super().__init__()
 
#         self.Wg = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
#         self.Ws = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.output = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),nn.Sigmoid())
 
#     def forward(self, gate, skip):
#         """Performs the forward pass in the attention gate.
#           Parameters
#           ----------
#           gate : torch.Tensor
#               gating signal from previous layer
#           gate : torch.Tensor
#               activation from corresponding encoder layer
#           Returns
#           -------
#           out: torch.Tensor
#               output activations
#           """
#         Wg = self.Wg(gate)
#         Ws = self.Ws(skip)
#         out = self.relu(Wg + Ws)
#         out = self.output(out)
#         return out 
    
    
# class Encoder(nn.Module):
#     """A representation for the encoder part of the unet.
#     Parameters
#     ----------
#     chs : tuple
#         holds the number of input channels of each block in the encoder
#     """

#     def __init__(self, chs):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
#         self.pool = nn.MaxPool2d((2,2))

#     def forward(self, x):
#         """Performs the forward pass for all blocks in the encoder.
#         Parameters
#         ----------
#         x : torch.Tensor
#             input
#         Returns
#         -------
#         list[torch.Tensor]
#             contains the outputs of each block in the encoder
#         """
#         ftrs = []  # a list to store features
#         for block in self.enc_blocks:
#             x = block(x)
#             ftrs.append(x) # save features to concatenate to decoder blocks
#             x = self.pool(x)
#         ftrs.append(x)
#         return ftrs


# class Decoder(nn.Module):
#     """A representation for the decoder part of the unet.
#     Layers consist of transposed convolutions followed by convolutional blocks.
#     Parameters
#     ----------
#     enc_chs : tuple
#         holds the number of input channels of each block in the encoder
#     dec_chs : tuple
#         holds the number of input channels of each block in the decoder
#     """

#     def __init__(self, enc_chs, dec_chs):
#         super().__init__()
#         self.chs = dec_chs
#         self.upconvs = nn.ModuleList([nn.ConvTranspose2d(dec_chs[i], dec_chs[i], 2, 2) for i in range(len(dec_chs) - 1)])
#        # self.attention = nn.ModuleList([AttentionBlock(dec_chs[i], dec_chs[i+1]) for i in range(len(dec_chs) - 1)])
#         self.dec_blocks = nn.ModuleList([Block(2*dec_chs[i], dec_chs[i + 1]) for i in range(len(dec_chs) - 1)])  # the first argument of the Block is multipled by 2 since you concatenate the features (which creates twice as many).

#     def forward(self, x, encoder_features):
#         """Performs the forward pass for all blocks in the decoder.
#         Parameters
#         ----------
#         x : torch.Tensor
#             input to the decoder
#         encoder_features: list
#             contains the encoder features to be concatenated to the corresponding level of the decoder
#         Returns
#         -------
#         x : torch.Tensor
#             output of the decoder
#         """
#         for i in range(len(self.chs) - 1):
#             x = self.upconvs[i](x) # transposed convolution
#             enc_ftrs = encoder_features[i] # get the features from the corresponding level of the encoder
#            # s = self.attention[i](x, enc_ftrs)
#             x = torch.cat([x, enc_ftrs], dim=1) # concatenate these features to x
#             x = self.dec_blocks[i](x) # convolutional block

#         return x


# class UNet(nn.Module):
#     """A representation for a unet
#     Parameters
#     ----------
#     enc_chs : tuple
#         holds the number of input channels of each block in the encoder
#     dec_chs : tuple
#         holds the number of input channels of each block in the decoder
#     num_classes : int
#         number of output classes of the segmentation
#     """

#     def __init__(self,enc_chs=(1, 64, 128, 256, 512),dec_chs=(512, 256, 128, 64, 32),num_classes=1,):
#         super().__init__()
#         self.encoder = Encoder(enc_chs)
#         self.decoder = Decoder(enc_chs,dec_chs)
#         self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], num_classes, 1))

#     def forward(self, x):
#         """Performs the forward pass of the unet.
#         Parameters
#         ----------
#         x : torch.Tensor
#             the input to the unet (image)
#         Returns
#         -------
#         out : torch.Tensor
#             unet output
#         """

#         enc_ftrs = self.encoder(x) # the encoder features are the input to the decoder
#         reverse_enc_ftrs = enc_ftrs[::-1] #Reverse the features,because the last output of the encoder (0 index after reverse) is the input to the decoder
#         out = self.decoder(reverse_enc_ftrs[0], reverse_enc_ftrs[1:])
#         out = self.head(out) # last layer ensures output has appropriate number of channels (1)
#         out = x + out
        
#         return out

class Block(nn.Module):
    """Convolutional building block of the unet. A block consists of
    two convolutional layers with ReLU activations
    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same', bias='True'),
                                  nn.ReLU(),
                                  nn.Conv2d(out_ch, out_ch, 3, padding='same', bias='True'),
                                  nn.ReLU())

    def forward(self, x):
        """Performs the forward pass of the block.
        Parameters
        ----------
        x : torch.Tensor
            the input to the block
        Returns
        -------
        x : torch.Tensor
            the output of the forward pass
        """
        return self.conv(x)
    
class Encoder_block(nn.Module):
    """The encoder part of the unet.
    Parameters
    ----------
    in_c : tuple
        holds the number of input channels of each block in the encoder
    out_c : tuple
        holds the number of output channels of each block in the encoder
    """
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.conv = Block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
 
    def forward(self, x):
        """Performs the forward pass in the encoder.
          Parameters
          ----------
          x : torch.Tensor
              input
          Returns
          -------
          p: torch.Tensor
              contains the output of the encoder block
          s: torch.Tensor
              contains the features of the encoder block
          """
        s = self.conv(x)
        p = self.pool(s)
        return s, p
    
class Attention_block(nn.Module):
    """The attention gate of the unet.
    Parameters
    ----------
    in_c : tuple
        holds the number of input channels of each block in the encoder
    out_c : tuple
        holds the number of output channels of each block in the encoder
    """
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.Wg = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.Ws = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),nn.Sigmoid())
 
    def forward(self, gate, skip):
        """Performs the forward pass in the attention gate.
          Parameters
          ----------
          gate : torch.Tensor
              gating signal from previous layer
          gate : torch.Tensor
              activation from corresponding encoder layer
          Returns
          -------
          out: torch.Tensor
              output activations
          """
        Wg = self.Wg(gate)
        Ws = self.Ws(skip)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out 
    
    
class Decoder_block(nn.Module):
    """The decoder part of the unet.
    Parameters
    ----------
    in_c : tuple
        holds the number of input channels of each block in the decoder
    out_c : tuple
        holds the number of output channels of each block in the decoder
    """
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = Attention_block(in_c, out_c)
        self.c1 = Block(in_c+out_c, out_c)
 
    def forward(self, x, s):
        """Performs the forward pass in the decoder
        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        s: torch.Tensor
            contains the encoder features to be concatenated to the corresponding level of the decoder
        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x
    
class AttentionNet(nn.Module):
    """A UNet with an attention gate
    Parameters
    ----------
    num_classes : int
        number of output classes of the segmentation
    nr_filters : int
        number of filters in the first layer
    """
    def __init__(self, num_classes=1, nr_filters=32):
        super().__init__()
 
        self.e1 = Encoder_block(1, nr_filters)
        self.e2 = Encoder_block(nr_filters, 2*nr_filters)
        self.e3 = Encoder_block(2*nr_filters, 4*nr_filters)
        self.e4 = Encoder_block(4*nr_filters, 8*nr_filters)
    
        self.b1 = Block(8*nr_filters, 16*nr_filters)
 
        self.d1 = Decoder_block(16*nr_filters, 8*nr_filters)
        self.d2 = Decoder_block(8*nr_filters, 4*nr_filters)
        self.d3 = Decoder_block(4*nr_filters, 2*nr_filters)
        self.d4 = Decoder_block(2*nr_filters, nr_filters)
 
        self.output = nn.Conv2d(nr_filters, num_classes, kernel_size=1, padding=0)
 
    def forward(self, x):
        """Performs the forward pass of the unet.
        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)
        Returns
        -------
        out : torch.Tensor
            unet output
        """

        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
 
        b1 = self.b1(p4)
 
        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
 
        output = self.output(d4)+x
        
        return output
    