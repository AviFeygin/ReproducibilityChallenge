# FA Block/Dehaze Block
# Copied from https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py

from torch import nn


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        #         collapse to channel dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            # convolutions on the channels
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            # non linearity
            nn.ReLU(),
            # another convolution
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            # sigmoid not sure why channel attention needs this.
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# Pixel Attention layer same as channel attention but for pixels. convolution on the pixels followed by nonlinearity
# and then convolution and sigmoid
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class FeatureAttentionBlock(nn.Module):
    def __init__(self, conv, dim, ker_size):
        super(FeatureAttentionBlock, self).__init__()
        self.convolution_1 = conv(dim, dim, ker_size, bias=True)
        self.relu_1 = nn.ReLU()
        self.convolution_2 = conv(dim, dim, ker_size, bias=True)
        self.CALayer = CALayer(dim)
        self.PALayer = PALayer(dim)

    def forward(self, x):
        output = self.convolution_1(x)
        output = self.relu_1(output)
        output = output + x
        output = self.convolution_2(output)
        output = self.CALayer(output)
        output = self.PALayer(output)
        output = output + x
        return output
