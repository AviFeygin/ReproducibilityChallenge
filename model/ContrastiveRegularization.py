import torch
from torch import nn
from torchvision import models


class Vgg19_sliced(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_sliced, self).__init__()
        #     get the pretrained vgg19 features
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # define the slices
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        self.block5 = torch.nn.Sequential()

        # add the features to the slices
        #         first conv block
        self.block1.add_module("0", vgg_pretrained_features[0])
        self.block1.add_module("1", vgg_pretrained_features[1])
        #     second conv block
        self.block2.add_module("2", vgg_pretrained_features[2])
        self.block2.add_module("3", vgg_pretrained_features[3])
        self.block2.add_module("4", vgg_pretrained_features[4])
        self.block2.add_module("5", vgg_pretrained_features[5])
        self.block2.add_module("6", vgg_pretrained_features[6])
        #     third conv block
        self.block3.add_module("7", vgg_pretrained_features[7])
        self.block3.add_module("8", vgg_pretrained_features[8])
        self.block3.add_module("9", vgg_pretrained_features[9])
        self.block3.add_module("10", vgg_pretrained_features[10])
        self.block3.add_module("11", vgg_pretrained_features[11])
        #     fourth conv block
        self.block4.add_module("12", vgg_pretrained_features[12])
        self.block4.add_module("13", vgg_pretrained_features[13])
        self.block4.add_module("14", vgg_pretrained_features[14])
        self.block4.add_module("15", vgg_pretrained_features[15])
        self.block4.add_module("16", vgg_pretrained_features[16])
        self.block4.add_module("17", vgg_pretrained_features[17])
        self.block4.add_module("18", vgg_pretrained_features[18])
        self.block4.add_module("19", vgg_pretrained_features[19])
        self.block4.add_module("20", vgg_pretrained_features[20])
        #     fifth conv block
        self.block5.add_module("21", vgg_pretrained_features[21])
        self.block5.add_module("22", vgg_pretrained_features[22])
        self.block5.add_module("23", vgg_pretrained_features[23])
        self.block5.add_module("24", vgg_pretrained_features[24])
        self.block5.add_module("25", vgg_pretrained_features[25])
        self.block5.add_module("26", vgg_pretrained_features[26])
        self.block5.add_module("27", vgg_pretrained_features[27])
        self.block5.add_module("28", vgg_pretrained_features[28])
        self.block5.add_module("29", vgg_pretrained_features[29])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        features_1 = self.block1(x)
        features_2 = self.block2(features_1)
        features_3 = self.block3(features_2)
        features_4 = self.block4(features_3)
        features_5 = self.block5(features_4)
        return [features_1, features_2, features_3, features_4, features_5]


class ContrastiveRegularization(nn.Module):
    def __init__(self, device):
        super(ContrastiveRegularization, self).__init__()
        self.vgg = Vgg19_sliced().to(device)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.l1distance = nn.L1Loss()
        # beta term for learnable penalty to reduce hallucination
        self.b = 0.1
        self.eps = 1e-10

    def forward(self, anchor, positive, negative):
        anchor_features, positive_features, negative_features = self.vgg(anchor), self.vgg(positive), self.vgg(negative)
        loss = 0

        length = len(self.weights)

        for i in range(length):
            distance_anchor_positive = self.l1distance(anchor_features[i], positive_features[i])
            distance_anchor_negative = self.l1distance(anchor_features[i], negative_features[i])
            contrastive = distance_anchor_positive / (distance_anchor_negative + self.eps)

            loss += self.weights[i] * contrastive
        return self.b * loss
