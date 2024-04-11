import torch
from torch import nn


class Mixup(nn.Module):
    def __init__(self, m):
        super(Mixup, self).__init__()
        # Initialize the learnable parameter 'w' with the initial value 'm'
        self.weight = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        # Define a sigmoid activation block
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2):
        # Apply sigmoid to 'w' to get the mixing factor between 0 and 1
        mix_factor = self.sigmoid(self.weight)

        # Expand mix_factor to match the dimensions of fea1 and fea2 if necessary
        # and compute the mixed output
        out = feature1 * mix_factor.expand_as(feature1) + feature2 * (1 - mix_factor.expand_as(feature2))
        return out
