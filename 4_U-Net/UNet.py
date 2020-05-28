import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def DoubleConv(in_channels, out_channels):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.down1 = DoubleConv(1,64)
        self.down2 = DoubleConv(64,128)
        self.down3 = DoubleConv(128,256)
        self.down4 = DoubleConv(256,512)
        self.down5 = DoubleConv(512,1024)
        self.up1 = DoubleConv(1024,512)
        self.up2 = DoubleConv(512,256)
        self.up3 = DoubleConv(256,128)
        self.up4 = DoubleConv(128,64)
        self.up5 = nn.Conv2d(64, 1, kernel_size=1) #last feature

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.UpConv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.UpConv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.UpConv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.UpConv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def forward(self, x):
        ## contracting path(left side)
        down1 = self.down1(x)
        MaxPool1 = self.MaxPool(down1)

        down2 = self.down2(MaxPool1)
        MaxPool2 = self.MaxPool(down2)

        down3 = self.down3(MaxPool2)
        MaxPool3 = self.MaxPool(down3)

        down4 = self.down4(MaxPool3)
        MaxPool4 = self.MaxPool(down4)

        down5 = self.down5(MaxPool4)
        UpConv1 = self.UpConv1(down5)

        ## expansive path(right side)
        cat1 = torch.cat((UpConv1, down4), dim=1)
        up1 = self.up1(cat1)
        UpConv2 = self.UpConv2(up1)

        cat2 = torch.cat((UpConv2, down3), dim=1)
        up2 = self.up2(cat2)
        UpConv3 = self.UpConv3(up2)

        cat3 = torch.cat((UpConv3, down2), dim=1)
        up3 = self.up3(cat3)
        UpConv4 = self.UpConv4(up3)

        cat4 = torch.cat((UpConv4, down1), dim=1)
        up4 = self.up4(cat4)
        up5 = self.up5(up4)

        return up5
