import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, r=16):
        super(ChannelAttention, self).__init__()
        # 定义全局平均池化层（GAP）
        self.inc=in_channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义两个全连接层，用于计算通道注意力权重
        self.fc1 = nn.Linear(in_channels , in_channels // r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // r, in_channels)
        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 计算全局平均池化
        avg_pool = self.global_avg_pool(x).view(-1,self.inc)

        # 通过全连接层计算通道注意力权重

        channel_weights = self.fc2(self.relu(self.fc1(avg_pool)))
        # 应用sigmoid激活函数得到注意力权重
        channel_weights = self.sigmoid(channel_weights).view(-1,self.inc,1,1)
        # 将注意力权重应用到输入特征上

        x = x * channel_weights
        return x

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


class seunet(nn.Module):
    def __init__(self,out):
        super(seunet, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.excitation1=ChannelAttention(64)
        self.excitation2 = ChannelAttention(128)
        self.excitation3 = ChannelAttention(256)
        self.excitation4 = ChannelAttention(512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer0 = nn.Sequential(nn.Conv2d(3,64,7,stride=1,padding=3),resnet.bn1,DoubleConv(64,64))
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.downn = DoubleConv(512,512)
        self.upcon1 = DoubleConv(512,256)
        self.upcon2   = DoubleConv(256,128)
        self.upcon3   = DoubleConv(128,64)
        self.upcon4   = DoubleConv(128,64)
        self.up1 =nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.up5 = nn.Conv2d(64, out, kernel_size=1)
    def forward(self, x):
        # Encoder
        x1=self.layer0(x)
        x2=self.maxpool(x1)
        x2=self.layer1(x2)
        x3=self.layer2 (x2)
        x4=self.layer3 (x3)
        x5=self.layer4 (x4)
        x5=self.downn(x5)
        # Decoder
        x = self.up1(self.excitation4(x5))
        x = torch.cat([x, self.excitation3(x4)], dim=1)
        x=self.upcon1(x)
        x = self.up2(x)
        x = torch.cat([x, self.excitation2(x3)], dim=1)
        x=self.upcon2(x)
        x = self.up3(x)
        x = torch.cat([x, self.excitation1(x2)], dim=1)
        x = self.upcon3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x=self.upcon4(x)
        x = self.up5(x)

        return x