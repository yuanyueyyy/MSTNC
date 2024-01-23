import torch
import torch.nn as nn
import torch.nn.functional as F

class Rep_2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Rep_2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(0, 0),bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x1 = self.norm1(self.conv1(x))
        out = self.act1(x1)
        return out


class Rep_3D_block(nn.Module):
    def __init__(self, in_channels, out_channels, bands_size):
        super(Rep_3D_block, self).__init__()
        bbb=int(bands_size/2)
        self.conv1= nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, bands_size), stride=1, padding=(0, 0, 0),bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU()


    def forward(self, x):

        x1 = self.norm1(self.conv1(x))
        out = self.act1(x1)
        return out

class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        patch = args.patch
        self.conv1 = Rep_3D_block(
                in_channels=1,
                out_channels=8,
                bands_size=7,
        )

        self.conv2 = Rep_3D_block(
                in_channels=8,
                out_channels=16,
                bands_size=5,
        )

        self.conv3 = Rep_3D_block(
                in_channels=16,
                out_channels=32,
                bands_size=3,
        )


        conv4_in_channels = int((args.band-12)*32)

        self.conv4 = Rep_2D_block(in_channels=conv4_in_channels, out_channels=64)
        self.pool=nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 1, 4, 2, 3)
        x = x.contiguous().view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        x = self.conv4(x)
        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])

        return x


class NET(nn.Module):
    def __init__(self, args):
        super(NET, self).__init__()
        self.name = 'NET'
        self.fe = encoder(args)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.classes))
        self.fprog = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256))


    def forward(self, X,f):
        if f==0:
            x = self.fe(X)
            z = self.fc(x)
            return z,x
        if f==1:
            x = self.fe(X)
            h = self.fprog(x)
            return h

class NET_mix(nn.Module):
    def __init__(self, args):
        super(NET_mix, self).__init__()
        self.name = 'NET_mix'
        self.fe = encoder(args)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.classes))
        self.fprog = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256))


    def forward(self, z):

        z = self.fc(z)
        return z
