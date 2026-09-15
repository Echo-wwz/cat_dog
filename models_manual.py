# -*- coding: utf-8 -*-
"""
models_manual.py —— 手动搭建的模型（供第三步及以后复用）

指导书第三步写了 `from step1_scratch import ResNet18Manual`，
但 step1_scratch.py 里**并没有** ResNet18Manual —— 这是个悬空引用。
这里补上：按 torchvision 的命名手写 ResNet18，保证 state_dict 键名一一对应，
预训练权重才能灌进去。
"""
import torch
import torch.nn as nn
from torchvision import models


class BasicBlock(nn.Module):
    """ResNet 的基本残差块：两层 3x3 卷积 + 一条"抄近道"的恒等映射"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 维度/尺寸对不上时，恒等支路也要用一个 1x1 卷积投影
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity              # ← 残差连接：梯度可以抄近道
        return self.relu(out)


class ResNet18Manual(nn.Module):
    """手动实现的 ResNet18，层名与 torchvision.models.resnet18 完全一致"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 与 torchvision 一致的初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_pretrained_resnet18(num_classes=2, freeze=True):
    """加载 ImageNet 预训练权重到手动 ResNet18，并替换分类头。

    注意：**不能**写成
        net = ResNet18Manual(num_classes=2)
        net.load_state_dict(official.state_dict(), strict=False)
    因为 strict=False 只忽略"缺失/多余"的键，**形状不匹配照样报错** ——
    fc 是 512->2 而官方是 512->1000，会直接 RuntimeError。
    正确做法和第二步一样：先建 1000 类结构，加载，再换头。
    """
    official = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net = ResNet18Manual(num_classes=1000)
    net.load_state_dict(official.state_dict())
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.fc.parameters():
            p.requires_grad = True
    return net
