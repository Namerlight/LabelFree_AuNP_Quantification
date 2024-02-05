import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("Logger")        # Get logger
c_handler = logging.StreamHandler()         # Create a handler
logger.addHandler(c_handler)                # link handler to logger
logger.setLevel(logging.INFO)              # Set logging level to the logger


class FCRNA(nn.Module):

    def __init__(self):
        super(FCRNA, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.conv4_bn = nn.BatchNorm2d(512)

        self.deConv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deConv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deConv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, bias=False, padding=(3//2, 3//2))
        self.deconv3_bn = nn.BatchNorm2d(1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ReLU = nn.ReLU()
        self.MaxPool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.ReLU(x)
        x = self.MaxPool2d(x)

        logger.debug(f"After Conv1 Relu, shape:{x.size()}")

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.ReLU(x)
        x = self.MaxPool2d(x)

        logger.debug(f"After Conv2 Relu MaxPool, shape:{x.size()}")

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.ReLU(x)
        x = self.MaxPool2d(x)

        logger.debug(f"After Conv3 Relu Maxpool, shape:{x.size()}")

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.ReLU(x)

        logger.debug(f"After Conv4 Relu Maxpool, shape:{x.size()}")

        x = self.upsample(x)
        x = self.deConv1(x)
        x = self.deconv1_bn(x)
        x = self.ReLU(x)

        logger.debug(f"After Upsamp DeConv1 Relu, shape:{x.size()}")

        x = self.upsample(x)
        x = self.deConv2(x)
        x = self.deconv2_bn(x)
        x = self.ReLU(x)

        logger.debug(f"After Upsamp DeConv2 Relu, shape:{x.size()}")

        x = self.upsample(x)
        x = self.deConv3(x)
        x = self.deconv3_bn(x)
        x = self.ReLU(x)

        logger.debug(f"After Upsamp DeConv3 Relu, shape:{x.size()}")

        return x



