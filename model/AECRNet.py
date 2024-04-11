import random

from torch import nn

from model.DeformableConv2d import DeformableConv2d
from model.FeatureAttentionBlock import FeatureAttentionBlock
from model.Mixup import Mixup


# Copied from https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class AECRNet(nn.Module):
    # Arguments copied from https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py
    def __init__(self, input_channels, output_channels, dim=64):
        super().__init__()

        # Need some padding for this model to work
        self.pad = nn.ReflectionPad2d(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Down-sampling layers
        self.down1 = nn.Conv2d(input_channels, dim, kernel_size=7, stride=1, padding=0)
        self.down2 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        # Feature Attention block
        self.fa_block = FeatureAttentionBlock(default_conv, dim * 4, 3)

        # Dynamic Feature Enhancement Block
        self.dfe_block = DeformableConv2d(dim * 4, dim * 4)

        # Mixup operations
        self.mix1 = Mixup(learnable_factor=-random.random())
        self.mix2 = Mixup(learnable_factor=-random.random())

        # Up-sampling layers
        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.Conv2d(dim, output_channels, kernel_size=7, padding=0)

    def forward(self, x):
        x = self.pad(x)

        x_down1 = self.down1(x)
        x_down1 = self.relu(x_down1)
        x_down2 = self.down2(x_down1)
        x_down2 = self.relu(x_down2)
        x_down3 = self.down3(x_down2)
        x_down3 = self.relu(x_down3)

        x1 = self.fa_block(x_down3)
        x2 = self.fa_block(x1)
        x3 = self.fa_block(x2)
        x4 = self.fa_block(x3)
        x5 = self.fa_block(x4)
        x6 = self.fa_block(x5)

        x_dcn1 = self.dfe_block(x6)
        x_dcn2 = self.dfe_block(x_dcn1)

        x_mix1 = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_mix1)
        x_up1 = self.relu(x_up1)
        x_mix2 = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_mix2)
        x_up2 = self.relu(x_up2)
        x_up3 = self.pad(x_up2)
        x_up3 = self.up3(x_up3)

        out = self.sigmoid(x_up3)
        return out
