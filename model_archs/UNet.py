import torch
import logging
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger("Logger")  # Get logger
c_handler = logging.StreamHandler()  # Create a handler
logger.addHandler(c_handler)  # link handler to logger
logger.setLevel(logging.INFO)  # Set logging level to the logger


def conv_block(channels: Tuple[int, int],
               size: Tuple[int, int],
               stride: Tuple[int, int] = (1, 1),
               N: int = 1):
    """
    Create a block with N convolutional layers with ReLU activation function.
    The first layer is IN x OUT, and all others - OUT x OUT.

    Args:
        channels: (IN, OUT) - no. of input and output channels
        size: kernel size (fixed for all convolution in a block)
        stride: stride (fixed for all convolution in a block)
        N: no. of convolutional layers

    Returns:
        A sequential container of N convolutional layers.
    """
    # a single convolution + batch normalization + ReLU block
    block = lambda in_channels: nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=channels[1],
                  kernel_size=size,
                  stride=stride,
                  bias=False,
                  padding=(size[0] // 2, size[1] // 2)),
        nn.BatchNorm2d(num_features=channels[1]),
        nn.ReLU()
    )
    # create and return a sequential container of convolutional layers
    # input size = channels[0] for first block and channels[1] for all others
    return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(self,
                 channels: Tuple[int, int],
                 size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 N: int = 1):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(channels, size, stride, N),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)


class UNet(nn.Module):

    def __init__(self, channels, filters, kernel_size):
        super(UNet, self).__init__()

        initial_filters = (channels, filters)
        down_filters = (filters, filters)
        up_filters = (2 * filters, filters)
        kernel = (kernel_size, kernel_size)

        # downsampling
        self.block1 = conv_block(channels=initial_filters, size=kernel, N=2)
        self.block2 = conv_block(channels=down_filters, size=kernel, N=2)
        self.block3 = conv_block(channels=down_filters, size=kernel, N=2)

        # upsampling
        self.block4 = ConvCat(channels=down_filters, size=kernel, N=2)
        self.block5 = ConvCat(channels=up_filters, size=kernel, N=2)
        self.block6 = ConvCat(channels=up_filters, size=kernel, N=2)

        # density prediction
        self.block7 = conv_block(channels=up_filters, size=kernel, N=2)
        self.density_pred = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=(1, 1), bias=False)

        self.MaxPool2d = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor):

        # downsampling
        x_b1 = self.block1(x)
        x_b2 = self.block2(self.MaxPool2d(x_b1))
        x_b3 = self.block3(self.MaxPool2d(x_b2))

        print("X_B3", x_b3.shape)

        # upsampling
        x_b4 = self.block4(self.MaxPool2d(x_b3), x_b3)
        x_b5 = self.block5(x_b4, x_b2)
        x_b6 = self.block6(x_b5, x_b1)

        print("X_B6", x_b6.shape)

        # density prediction
        x_b7 = self.block7(x_b6)
        print("X_B7", x_b7.shape)
        x = self.density_pred(x_b7)

        return x


# img = torch.rand((1, 3, 1440, 1440), dtype=torch.float)
#
# # img_path = os.path.join("..", "data", "asmm", "PEG100ImageScale", "set3", "1day_012_1_1sec.jpg")
#
# # img = torchvision.io.read_image(img_path).float()
#
# unet = UNet(channels=3, filters=64, kernel_size=3)
# out_dmap = unet(img)
# print("out", out_dmap.shape)
