"""
UNet model.

A UNet encoder, decoder and complete model.
"""
from typing import Callable, List, Optional

import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    """
    Convolution block used in UNets.

    Convolution block consist of two convolution layers with optional batch norm,
    dropout and with a final activation function.

    The parameters are directly mapped to PyTorch Conv2D and Conv3d parameters, see
    PyTorch torch.nn.Conv2d and torch.nn.Conv3d for more information.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolutions, 2 or 3.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    intermediate_channel_multiplier : int, optional
        Multiplied for the number of output channels, by default 1.
    stride : int, optional
        Stride of the convolutions, by default 1.
    padding : int, optional
        Padding of the convolutions, by default 1.
    bias : bool, optional
        Bias of the convolutions, by default True.
    groups : int, optional
        Controls the connections between inputs and outputs, by default 1.
    activation : str, optional
        Activation function, by default "ReLU".
    dropout_perc : float, optional
        Dropout percentage, by default 0.
    use_batch_norm : bool, optional
        Use batch norm, by default False.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        out_channels: int,
        intermediate_channel_multiplier: int = 1,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        activation: str = "ReLU",
        dropout_perc: float = 0,
        use_batch_norm: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolutions, 2 or 3.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        intermediate_channel_multiplier : int, optional
            Multiplied for the number of output channels, by default 1.
        stride : int, optional
            Stride of the convolutions, by default 1.
        padding : int, optional
            Padding of the convolutions, by default 1.
        bias : bool, optional
            Bias of the convolutions, by default True.
        groups : int, optional
            Controls the connections between inputs and outputs, by default 1.
        activation : str, optional
            Activation function, by default "ReLU".
        dropout_perc : float, optional
            Dropout percentage, by default 0.
        use_batch_norm : bool, optional
            Use batch norm, by default False.
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = getattr(nn, f"Conv{conv_dim}d")(
            in_channels,
            out_channels * intermediate_channel_multiplier,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.conv2 = getattr(nn, f"Conv{conv_dim}d")(
            out_channels * intermediate_channel_multiplier,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        self.batch_norm1 = getattr(nn, f"BatchNorm{conv_dim}d")(out_channels * intermediate_channel_multiplier)
        self.batch_norm2 = getattr(nn, f"BatchNorm{conv_dim}d")(out_channels)

        self.dropout = (getattr(nn, f"Dropout{conv_dim}d")(dropout_perc) if dropout_perc > 0 else None)
        self.activation = (getattr(nn, f"{activation}")() if activation is not None else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.use_batch_norm:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UnetEncoder(nn.Module):
    """
    Unet encoder pathway.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
    in_channels : int, optional
        Number of input channels, by default 1.
    depth : int, optional
        Number of encoder blocks, by default 3.
    num_channels_init : int, optional
        Number of channels in the first encoder block, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    pool_kernel : int, optional
        Kernel size for the max pooling layers, by default 2.
    """

    def __init__(
        self,
        conv_dim: int,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool_kernel: int = 2,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
        in_channels : int, optional
            Number of input channels, by default 1.
        depth : int, optional
            Number of encoder blocks, by default 3.
        num_channels_init : int, optional
            Number of channels in the first encoder block, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        pool_kernel : int, optional
            Kernel size for the max pooling layers, by default 2.
        """
        super().__init__()

        self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        encoder_blocks = []

        for n in range(depth):
            out_channels = num_channels_init * (2**n)
            in_channels = in_channels if n == 0 else out_channels // 2
            encoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_perc=dropout,
                    use_batch_norm=use_batch_norm,
                ))
            encoder_blocks.append(self.pooling)

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        List[torch.Tensor]
            Output of each encoder block (skip connections) and final output of the
            encoder.
        """
        encoder_features = []
        for module in self.encoder_blocks:
            x = module(x)
            if isinstance(module, Conv_Block):
                encoder_features.append(x)
        features = [x, *encoder_features]
        return features


class UnetDecoder(nn.Module):
    """
    Unet decoder pathway.

    Parameters
    ----------
    conv_dim : int
        Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
    depth : int, optional
        Number of decoder blocks, by default 3.
    num_channels_init : int, optional
        Number of channels in the first encoder block, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    """

    def __init__(
        self,
        conv_dim: int,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimension of the convolution layers, 2 for 2D or 3 for 3D.
        depth : int, optional
            Number of decoder blocks, by default 3.
        num_channels_init : int, optional
            Number of channels in the first encoder block, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        """
        super().__init__()

        upsampling = nn.Upsample(scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear")
        in_channels = out_channels = num_channels_init * 2**(depth - 1)
        self.bottleneck = Conv_Block(
            conv_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channel_multiplier=2,
            use_batch_norm=use_batch_norm,
            dropout_perc=dropout,
        )

        decoder_blocks = []
        for n in range(depth):
            decoder_blocks.append(upsampling)
            in_channels = num_channels_init * 2**(depth - n)
            out_channels = num_channels_init
            decoder_blocks.append(
                Conv_Block(
                    conv_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    intermediate_channel_multiplier=2,
                    dropout_perc=dropout,
                    activation="ReLU",
                    use_batch_norm=use_batch_norm,
                ))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, *features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        *features :  List[torch.Tensor]
            List containing the output of each encoder block(skip connections) and final
            output of the encoder.

        Returns
        -------
        torch.Tensor
            Output of the decoder.
        """
        x = features[0]
        skip_connections = features[1:][::-1]
        x = self.bottleneck(x)
        for i, module in enumerate(self.decoder_blocks):
            x = module(x)
            if isinstance(module, nn.Upsample):
                x = torch.cat([x, skip_connections[i // 2]], axis=1)
        return x


class UNet(nn.Module):
    """
    UNet model.

    Adapted for PyTorch from
    https://github.com/juglab/n2v/blob/main/n2v/nets/unet_blocks.py.

    Parameters
    ----------
    conv_dim : int
        Number of dimensions of the convolution layers (2 or 3).
    num_classes : int, optional
        Number of classes to predict, by default 1.
    in_channels : int, optional
        Number of input channels, by default 1.
    depth : int, optional
        Number of downsamplings, by default 3.
    num_channels_init : int, optional
        Number of filters in the first convolution layer, by default 64.
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True.
    dropout : float, optional
        Dropout probability, by default 0.0.
    pool_kernel : int, optional
        Kernel size of the pooling layers, by default 2.
    last_activation : Optional[Callable], optional
        Activation function to use for the last layer, by default None.
    """

    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool_kernel: int = 2,
        last_activation: Optional[Callable] = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        conv_dim : int
            Number of dimensions of the convolution layers (2 or 3).
        num_classes : int, optional
            Number of classes to predict, by default 1.
        in_channels : int, optional
            Number of input channels, by default 1.
        depth : int, optional
            Number of downsamplings, by default 3.
        num_channels_init : int, optional
            Number of filters in the first convolution layer, by default 64.
        use_batch_norm : bool, optional
            Whether to use batch normalization, by default True.
        dropout : float, optional
            Dropout probability, by default 0.0.
        pool_kernel : int, optional
            Kernel size of the pooling layers, by default 2.
        last_activation : Optional[Callable], optional
            Activation function to use for the last layer, by default None.
        """
        super().__init__()

        self.encoder = UnetEncoder(
            conv_dim,
            in_channels=in_channels,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            pool_kernel=pool_kernel,
        )

        self.decoder = UnetDecoder(
            conv_dim,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_channels_init,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.last_activation = last_activation if last_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x :  torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the model.
        """
        encoder_features = self.encoder(x)
        x = self.decoder(*encoder_features)
        x = self.final_conv(x)
        x = self.last_activation(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from collections import OrderedDict
# from torch.nn import init
# import numpy as np

# def conv3x3(in_channels, out_channels, stride=1,
#             padding=1, bias=True, groups=1):
#     return nn.Conv2d(
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=stride,
#         padding=padding,
#         bias=bias,
#         groups=groups)

# def upconv2x2(in_channels, out_channels, mode='transpose'):
#     if mode == 'transpose':
#         return nn.ConvTranspose2d(
#             in_channels,
#             out_channels,
#             kernel_size=2,
#             stride=2)
#     else:
#         # out_channels is always going to be the same
#         # as in_channels
#         return nn.Sequential(
#             nn.Upsample(mode='bilinear', scale_factor=2),
#             conv1x1(in_channels, out_channels))

# def conv1x1(in_channels, out_channels, groups=1):
#     return nn.Conv2d(
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         groups=groups,
#         stride=1)

# class DownConv(nn.Module):
#     """
#     A helper Module that performs 2 convolutions and 1 MaxPool.
#     A ReLU activation follows each convolution.
#     """
#     def __init__(self, in_channels, out_channels, pooling=True):
#         super(DownConv, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.pooling = pooling

#         self.conv1 = conv3x3(self.in_channels, self.out_channels)
#         self.conv2 = conv3x3(self.out_channels, self.out_channels)

#         if self.pooling:
#             self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         before_pool = x
#         if self.pooling:
#             x = self.pool(x)
#         return x, before_pool

# class UpConv(nn.Module):
#     """
#     A helper Module that performs 2 convolutions and 1 UpConvolution.
#     A ReLU activation follows each convolution.
#     """
#     def __init__(self, in_channels, out_channels,
#                  merge_mode='concat', up_mode='transpose'):
#         super(UpConv, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.merge_mode = merge_mode
#         self.up_mode = up_mode

#         self.upconv = upconv2x2(self.in_channels, self.out_channels,
#             mode=self.up_mode)

#         if self.merge_mode == 'concat':
#             self.conv1 = conv3x3(
#                 2*self.out_channels, self.out_channels)
#         else:
#             # num of input channels to conv2 is same
#             self.conv1 = conv3x3(self.out_channels, self.out_channels)
#         self.conv2 = conv3x3(self.out_channels, self.out_channels)

#     def forward(self, from_down, from_up):
#         """ Forward pass
#         Arguments:
#             from_down: tensor from the encoder pathway
#             from_up: upconv'd tensor from the decoder pathway
#         """
#         from_up = self.upconv(from_up)
#         if self.merge_mode == 'concat':
#             x = torch.cat((from_up, from_down), 1)
#         else:
#             x = from_up + from_down
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         return x

# from torch.autograd import Variable

# class UNet(nn.Module):
#     """ `UNet` class is based on https://arxiv.org/abs/1505.04597
#     The U-Net is a convolutional encoder-decoder neural network.
#     Contextual spatial information (from the decoding,
#     expansive pathway) about an input tensor is merged with
#     information representing the localization of details
#     (from the encoding, compressive pathway).
#     Modifications to the original paper:
#     (1) padding is used in 3x3 convolutions to prevent loss
#         of border pixels
#     (2) merging outputs does not require cropping due to (1)
#     (3) residual connections can be used by specifying
#         UNet(merge_mode='add')
#     (4) if non-parametric upsampling is used in the decoder
#         pathway (specified by upmode='upsample'), then an
#         additional 1x1 2d convolution occurs after upsampling
#         to reduce channel dimensionality by a factor of 2.
#         This channel halving happens with the convolution in
#         the tranpose convolution (specified by upmode='transpose')
#     """

#     def __init__(self, num_classes, in_channels=1, depth=5,
#                  start_filts=64, up_mode='transpose',
#                  merge_mode='add'):
#         """
#         Arguments:
#             in_channels: int, number of channels in the input tensor.
#                 Default is 3 for RGB images.
#             depth: int, number of MaxPools in the U-Net.
#             start_filts: int, number of convolutional filters for the
#                 first conv.
#             up_mode: string, type of upconvolution. Choices: 'transpose'
#                 for transpose convolution or 'upsample' for nearest neighbour
#                 upsampling.
#         """
#         super(UNet, self).__init__()

#         if up_mode in ('transpose', 'upsample'):
#             self.up_mode = up_mode
#         else:
#             raise ValueError("\"{}\" is not a valid mode for "
#                              "upsampling. Only \"transpose\" and "
#                              "\"upsample\" are allowed.".format(up_mode))

#         if merge_mode in ('concat', 'add'):
#             self.merge_mode = merge_mode
#         else:
#             raise ValueError("\"{}\" is not a valid mode for"
#                              "merging up and down paths. "
#                              "Only \"concat\" and "
#                              "\"add\" are allowed.".format(up_mode))

#         # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
#         if self.up_mode == 'upsample' and self.merge_mode == 'add':
#             raise ValueError("up_mode \"upsample\" is incompatible "
#                              "with merge_mode \"add\" at the moment "
#                              "because it doesn't make sense to use "
#                              "nearest neighbour to reduce "
#                              "depth channels (by half).")

#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.start_filts = start_filts
#         self.depth = depth

#         self.down_convs = []
#         self.up_convs = []

#         self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))

#         # create the encoder pathway and add to a list
#         for i in range(depth):
#             ins = self.in_channels if i == 0 else outs
#             outs = self.start_filts*(2**i)
#             pooling = True if i < depth-1 else False

#             down_conv = DownConv(ins, outs, pooling=pooling)
#             self.down_convs.append(down_conv)

#         # create the decoder pathway and add to a list
#         # - careful! decoding only requires depth-1 blocks
#         for i in range(depth-1):
#             ins = outs
#             outs = ins // 2
#             up_conv = UpConv(ins, outs, up_mode=up_mode,
#                 merge_mode=merge_mode)
#             self.up_convs.append(up_conv)

#         self.conv_final = conv1x1(outs, self.num_classes)

#         # add the list of modules to current module
#         self.down_convs = nn.ModuleList(self.down_convs)
#         self.up_convs = nn.ModuleList(self.up_convs)

#         self.reset_params()

#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, nn.Conv2d):
#             init.xavier_normal(m.weight)
#             init.constant(m.bias, 0)

#     def reset_params(self):
#         for i, m in enumerate(self.modules()):
#             self.weight_init(m)

#     def forward(self, x):
#         encoder_outs = []

#         # encoder pathway, save outputs for merging
#         for i, module in enumerate(self.down_convs):
#             x, before_pool = module(x)
#             encoder_outs.append(before_pool)

#         for i, module in enumerate(self.up_convs):
#             before_pool = encoder_outs[-(i+2)]
#             x = module(before_pool, x)

#         # No softmax is used. This means you need to use
#         # nn.CrossEntropyLoss is your training script,
#         # as this module includes a softmax already.
#         x = self.conv_final(x)
#         return x
