import os
import torch
import torch.nn as nn
import torchvision.io


class MCNN_b1b2only(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_b1b2only, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=28, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)

        x2 = self.branch2(img_tensor)

        x = torch.cat((x1, x2), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x


class MCNN_b2b3only(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_b2b3only, self).__init__()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):

        x2 = self.branch2(img_tensor)

        x3 = self.branch3(img_tensor)

        x = torch.cat((x2, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x


class MCNN_b1b3only(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_b1b3only, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=3//2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9, padding=9//2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):

        x1 = self.branch1(img_tensor)

        x3 = self.branch3(img_tensor)

        x = torch.cat((x1, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x


class MCNN_4layer(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_4layer, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=72, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        # x1 = self.dropout(x1)
        # print("x1", x1.shape)

        x2 = self.branch2(img_tensor)
        # x2 = self.dropout(x2)
        # print("x2", x2.shape)

        x3 = self.branch3(img_tensor)
        # x3 = self.dropout(x3)
        # print("x2", x2.shape)

        x = torch.cat((x1, x2, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x

class MCNN_2layer(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_2layer, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=72, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        # x1 = self.dropout(x1)
        # print("x1", x1.shape)

        x2 = self.branch2(img_tensor)
        # x2 = self.dropout(x2)
        # print("x2", x2.shape)

        x3 = self.branch3(img_tensor)
        # x3 = self.dropout(x3)
        # print("x2", x2.shape)

        x = torch.cat((x1, x2, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x


class MCNN_1layer(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNN_1layer, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        # x1 = self.dropout(x1)
        # print("x1", x1.shape)

        x2 = self.branch2(img_tensor)
        # x2 = self.dropout(x2)
        # print("x2", x2.shape)

        x3 = self.branch3(img_tensor)
        # x3 = self.dropout(x3)
        # print("x2", x2.shape)

        x = torch.cat((x1, x2, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x

class MCNNog(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''

    def __init__(self):
        super(MCNNog, self).__init__()

        # self.dropout = nn.Dropout(0.5)

        kernel_sizes = [3, 3, 9]

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1, padding=0))

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        # x1 = self.dropout(x1)
        # print("x1", x1.shape)

        x2 = self.branch2(img_tensor)
        # x2 = self.dropout(x2)
        # print("x2", x2.shape)

        x3 = self.branch3(img_tensor)
        # x3 = self.dropout(x3)
        # print("x2", x2.shape)

        x = torch.cat((x1, x2, x3), 1)
        # print("x cat", x.shape)

        x = self.fuse(x)
        # print("x fused", x.shape)

        return x


if __name__ == "__main__":
    img = torch.rand((1, 3, 256, 256), dtype=torch.float)

    # img_path = os.path.join("..", "data", "asmm", "PEG100ImageScale", "set3", "1day_012_1_1sec.jpg")

    # img = torchvision.io.read_image(img_path).float()

    mcnn = MCNNog()
    out_dmap = mcnn(img)
    print("out", out_dmap.shape)