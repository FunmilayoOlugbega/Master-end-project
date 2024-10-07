import torch
import torch.nn as nn

class Block2D(nn.Module):
    """A representation for the basic convolutional building block of the unet

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

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
        # a block consists of two convolutional layers
        # with ReLU activations

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x

class ActivationBlock2D(nn.Module):
    """A representation for the activation block for dose prediction

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same')
        self.relu = nn.ReLU(inplace=True)

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
        # a block consists of one convolutional layers
        # with ReLU activation

        x = self.conv1(x)
        x = self.relu(x)
        return x

class Encoder2D(nn.Module):
    """A representation for the encoder part of the unet.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the encoder

    """

    def __init__(self, chs=(1, 64, 128, 256, 512)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block2D(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_block = Block2D(chs[-1], 2*chs[-1])

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        list[torch.Tensor]
            contains the outputs of each block in the encoder
        """
        ftrs = []  # a list to store features
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        ftrs.append(self.last_block(x))
        return ftrs[::-1]
    
class Decoder2D(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(2*chs[i], chs[i], 2, 2) for i in range(len(chs))]
        )
        self.dec_blocks = nn.ModuleList(
            [Block2D(2*chs[i], chs[i]) for i in range(len(chs))]
        )  # the first argument of the Block is multipled by 2 since you concatenate the features (which creates twice as many).

    def forward(self, x, encoder_features):
        """Performs the forward pass for all blocks in the decoder.

        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        for i in range(len(self.upconvs)):
            # transposed convolution
            x = self.upconvs[i](x)
            # get the features from the corresponding level of the encoder
            enc_ftrs = encoder_features[i]
            # concatenate these features to x
            x = torch.cat([x, enc_ftrs], dim=1)
            # convolutional block
            x = self.dec_blocks[i](x)
        return x
    
class ActivationDose2D(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs, out_chs):
        super().__init__()
        chs = self.find_subdivisions(chs, out_chs)        
        self.activation_blocks = nn.ModuleList(
            [ActivationBlock2D(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )        

    def find_subdivisions(self, n, out_chs):
        subdivisions = []
        subdivisions.append(n)
        while n > out_chs:
            n //= 2
            subdivisions.append(n)
        return subdivisions
        
    def forward(self, x):
        """Performs the forward pass for all blocks in the decoder.

        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        for i in range(len(self.activation_blocks)):
             x = self.activation_blocks[i](x)
        return x

class Block3D(nn.Module):
    """A representation for the basic convolutional building block of the unet

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(out_ch)
        self.batchnorm2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

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
        # a block consists of two convolutional layers
        # with ReLU activations

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x

class ActivationBlock3D(nn.Module):
    """A representation for the activation block for dose prediction

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding='same')
        self.relu = nn.ReLU(inplace=True)

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
        # a block consists of one convolutional layers
        # with ReLU activation

        x = self.conv1(x)
        x = self.relu(x)
        return x    
    
class Encoder3D(nn.Module):
    """A representation for the encoder part of the unet.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the encoder

    """

    def __init__(self, chs=(1, 64, 128, 256, 512)):
        super().__init__()
        self.chs = chs
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block3D(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.last_block = Block3D(chs[-1], 2*chs[-1])

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        list[torch.Tensor]
            contains the outputs of each block in the encoder
        """
        ftrs = []  # a list to store features
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        ftrs.append(self.last_block(x))
        return ftrs[::-1]


class Decoder3D(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose3d(2*chs[i], chs[i], 2, 2) for i in range(len(chs))]
        )
        self.dec_blocks = nn.ModuleList(
            [Block3D(2*chs[i], chs[i]) for i in range(len(chs))]
        )  # the first argument of the Block is multipled by 2 since you concatenate the features (which creates twice as many).

    def forward(self, x, encoder_features):
        """Performs the forward pass for all blocks in the decoder.

        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        for i in range(len(self.upconvs)):
            # transposed convolution
            x = self.upconvs[i](x)
            # get the features from the corresponding level of the encoder
            enc_ftrs = encoder_features[i]
            # concatenate these features to x
            x = torch.cat([x, enc_ftrs], dim=1)
            # convolutional block
            x = self.dec_blocks[i](x)
        return x
    
class ActivationDose3D(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs, out_chs):
        super().__init__()
        chs = self.find_subdivisions(chs, out_chs)        
        self.activation_blocks = nn.ModuleList(
            [ActivationBlock3D(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )        

    def find_subdivisions(self, n, out_chs):
        subdivisions = []
        subdivisions.append(n)
        while n > out_chs:
            n //= 2
            subdivisions.append(n)
        return subdivisions
        
    def forward(self, x):
        """Performs the forward pass for all blocks in the decoder.

        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder
        """
        for i in range(len(self.activation_blocks)):
             x = self.activation_blocks[i](x)
        return x   