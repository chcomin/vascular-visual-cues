"""Patch classification network: two residual blocks with 64 channels followed by
three fully connected layers. Adapted from the torchvision ResNet."""

import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class ResNetClassifier(nn.Module):
    """ResNet for classifying patches of size `patch_size` x `patch_size`.

    The paper uses `ResNetClassifier([2], [64])`, i.e., a single stage with two
    residual blocks of 64 channels.
    """

    def __init__(self, layers, inplanes, num_classes=2, patch_size=9, zero_init_residual=False):
        super().__init__()

        self.norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, inplanes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = self.norm_layer(inplanes[0], momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        stages = [self._make_layer(inplanes[0], inplanes[0], layers[0])]
        for idx in range(len(layers)-1):
            stages.append(self._make_layer(inplanes[idx], inplanes[idx+1], layers[idx+1]))
        self.stages = nn.Sequential(*stages)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(patch_size*patch_size*inplanes[-1], 128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(64, num_classes)

        self._init_parameters(zero_init_residual)

    def _make_layer(self, inplanes, planes, blocks):

        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes),
                self.norm_layer(planes, momentum=0.1),
            )

        layers = [BasicBlock(inplanes, planes, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def _init_parameters(self, zero_init_residual):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stages(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
