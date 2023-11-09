import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self,inn,out):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(inn, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)
        self.upcon1   = DoubleConv(1024,512)
        self.upcon2   = DoubleConv(512,256)
        self.upcon3   = DoubleConv(256,128)
        self.upcon4   = DoubleConv(128,64)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.up5 = nn.Conv2d(64, out, kernel_size=1)
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x3 = self.down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down4(x6)
        x8 = self.maxpool(x7)
        x9 = self.down5(x8)
        # Decoder
        x = self.up1(x9)
        x = torch.cat([x, x7], dim=1)
        x=self.upcon1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x=self.upcon2(x)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upcon3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x=self.upcon4(x)
        x = self.up5(x)
        return x