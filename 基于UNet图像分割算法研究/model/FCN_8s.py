# fcn_res101.py
import torch
import torch.nn as nn
import torchvision.models as models


class FCN8s(nn.Module):
    def __init__(self, out_channel=21):
        super(FCN8s, self).__init__()
        # self.backbone = models.resnet101(pretrained=True) #旧版本写法
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # 4倍下采样 256
        self.stage1 = nn.Sequential(*list(self.backbone.children())[:-5])
        # 8倍下采样 512
        self.stage2 = nn.Sequential(list(self.backbone.children())[-5])
        # 16倍下采样 1024
        self.stage3 = nn.Sequential(list(self.backbone.children())[-4])
        # 32倍下采样 2048
        self.stage4 = nn.Sequential(list(self.backbone.children())[-3])

        self.conv2048_256 = nn.Conv2d(2048, 256, 1)
        self.conv1024_256 = nn.Conv2d(1024, 256, 1)
        self.conv512_256 = nn.Conv2d(512, 256, 1)

        self.upsample2x = nn.Upsample(scale_factor=2)
        self.upsample8x = nn.Upsample(scale_factor=8)

        self.outconv = nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.stage1(input)
        output_s8 = self.stage2(output)
        output_s16 = self.stage3(output_s8)
        output_s32 = self.stage4(output_s16)

        output_s8 = self.conv512_256(output_s8)
        output_s16 = self.conv1024_256(output_s16)
        output_s32 = self.conv2048_256(output_s32)

        output_s32 = self.upsample2x(output_s32)
        output_s16 = output_s16 + output_s32

        output_s16 = self.upsample2x(output_s16)
        output_s8 = output_s8 + output_s16

        output_s8 = self.upsample8x(output_s8)
        final_output = self.outconv(output_s8)

        return final_output