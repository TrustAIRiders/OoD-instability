#https://github.com/y0ast/pytorch-snippets/tree/main/minimal_cifar
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet_model2(torch.nn.Module):
    def __init__(self):
        super(ResNet_model2, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        return self.resnet(x)