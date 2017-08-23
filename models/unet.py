from torch import nn
import torch.nn.functional as F

class conv_bn_block(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding=1, stride=1):
        self.conv_1 = nn.Con2d(in_size, out_size, kernel_size,
                               padding=padding, stride=stride)
        self.bn_1 = nn.BatchNorm2d(out_size)
        self.relu = F.relu()

class UNET_256(nn.Module):

    def __init__(self):
        1+1

    def forward(self, *input):