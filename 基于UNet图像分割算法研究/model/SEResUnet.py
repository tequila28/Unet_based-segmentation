import torch
import torch.nn as nn
from model.senet import seresnet34

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
class seresunet(nn.Module):
    def __init__(self,out):
        super(seresunet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resnet = seresnet34()
        self.layer0 = self.resnet.pre
        self.layer1 = self.resnet.stage1
        self.layer2 = self.resnet.stage2
        self.layer3 = self.resnet.stage3
        self.layer4 = self.resnet.stage4
        self.upcon1 = DoubleConv(512,256)
        self.upcon2   = DoubleConv(256,128)
        self.upcon3   = DoubleConv(128,64)
        self.upcon4   = DoubleConv(128,64)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.up5 = nn.Conv2d(64, out, kernel_size=1)
    def forward(self, x):
        # Encoder
        x1=self.layer0(x)
        x2 = self.maxpool(x1)
        x2=self.layer1(x2)
        x3=self.layer2 (x2)
        x4=self.layer3 (x3)
        x5=self.layer4 (x4)
        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x=self.upcon1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x=self.upcon2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upcon3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x=self.upcon4(x)
        x = self.up5(x)
        return x