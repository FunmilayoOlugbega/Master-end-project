# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:48:41 2024

@author: Gast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    
class Attention_gate(nn.Module):
    """The attention gate uses additive attention and can be deployed in the decoder of the U-net.
    Parameters
    ----------
    in_c : int
        holds the number of input channels of each block in the encoder
    out_c : int
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


class ChannelAttention(nn.Module):
    """The channel attention of CBAM
    Parameters
    ----------
    channel : int
        holds the number of output channels of the block in the encoder or decoder
    reduction: int
        the reduction ratio of the attention module
    """       
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),nn.ReLU(inplace=True),nn.Conv2d(channel // reduction, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Performs the forward pass in channel attention
          Parameters
          ----------
          x : torch.Tensor
              the input to the block
          Returns
          -------
          torch.Tensor
              output activations
          """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """The spatial attention of CBAM
    Parameters
    ----------
    kernel_size : int
        kernel size of the spatial module
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Performs the forward pass in spatial attention
          Parameters
          ----------
          x : torch.Tensor
              the input to the block
          Returns
          -------
          torch.Tensor
              output activations
          """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
    
class CBAM(nn.Module):
    """Convolution Block Attention Module
    Parameters
    ----------
    channel : int
        holds the number of output channels of the block in the encoder or decoder
    reduction: int
        the reduction ratio of the attention module
    kernel_size : int
        kernel size of the spatial module
    """      
    def __init__(self, channel, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """Performs the forward pass in CBAM by applying channel and spatial attention
          Parameters
          ----------
          x : torch.Tensor
              the input to the block
          Returns
          -------
          x : torch.Tensor
              output activations
          """
        x = self.ca(x)
        x = self.sa(x)
        return x


class Flatten(nn.Module):
    """Flatten tensor
      Parameters
      ----------
      x : torch.Tensor
          input tensor
      Returns
      -------
      torch.Tensor
          tensor with flattened shape
      """
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class ChannelGate(nn.Module):
    """The channel attention of BAM
    Parameters
    ----------
    gate_channel : int
        holds the number of output channels of the block in the encoder or decoder
    reduction_ratio: int
        the reduction ratio of the attention module
    num_layers : int
        number of layers
    """      
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
        
    def forward(self, in_tensor):
        """Performs the forward pass in channel attention
          Parameters
          ----------
          in_tensor : torch.Tensor
              the input to the block
          Returns
          -------
          torch.Tensor
              output activations
          """
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    """The spatial attention of BAM
    Parameters
    ----------
    gate_channel : int
        holds the number of output channels of the block in the encoder or decoder
    reduction_ratio: int
        the reduction ratio of the attention module
    dilation_conv_num : int
        number of convolutions with dilation
    dilation_val : int
        value with which padding is done in the convolution
    """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, padding_mode='replicate', dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        """Performs the forward pass in spatial attention
          Parameters
          ----------
          in_tensor : torch.Tensor
              the input to the block
          Returns
          -------
          torch.Tensor
              output activations
          """
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class BAM(nn.Module):
    """Bottleneck Attention Module
    Parameters
    ----------
    gate_channel : int
        holds the number of output channels of the block in the encoder or decoder
    """   
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self, in_tensor):
        """Performs the forward pass in BAM by applying channel and spatial attention
          Parameters
          ----------
          in_tensor : torch.Tensor
              the input to the block
          Returns
          -------
          x : torch.Tensor
              output activations after sigmoid is applied
          """
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor


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
        #self.bam = BAM(out_c)
 
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
        #s = self.bam(s)
        p = self.pool(s)
        return s, p
    
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
        self.c1 = Block(in_c+out_c, out_c)
        #self.ag = Attention_gate(in_c, out_c)
        #self.cbam = CBAM(out_c)
        #self.bam = BAM(out_c)
 
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
        #s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        #x = self.bam(x)
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
    