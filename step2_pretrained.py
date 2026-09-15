# -*- coding: utf-8 -*-
"""
第二步：在手动模型基础上加载 ImageNet 预训练参数（对应指导书 2.1 / 2.2）

核心三步：
    1. 先建 1000 类的完整结构
    2. 加载官方权重（此时形状完全一致，strict=True 也能过）
    3. 再把分类头换成 2 类，新头自动随机初始化

运行：
    python step2_pretrained.py                      # 冻结卷积，只训分类头（默认）
    python step2_pretrained.py --epochs 5
    python step2_pretrained.py --finetune           # 进阶：解冻全部层，lr=1e-5 微调
"""
import argparse
import os

import torch
import torch.nn as nn
from torchvision import models

from common import get_loaders, train_and_eval
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, SEED, set_seed


class AlexNetManual(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 卷积部分必须命名为 features，且层顺序与 torchvision 一致
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类部分必须命名为 classifier，索引 6 是最后一层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # 索引 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 索引 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6 <- 换这里
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_pretrained_alexnet(num_classes=2, finetune=False):
    """加载 ImageNet 预训练 AlexNet，并把分类头换成 num_classes 类"""
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)          # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())     # 2. 形状一致，strict=True 也能加载
    net.classifier[6] = nn.Linear(4096, num_classes)   # 3. 换成 2 类（新头随机初始化）

    if not finetune:
        # 冻结：只训练 classifier
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--finetune", action="store_true",
                        help="解冻全部层，用小学习率微调（进阶尝试）")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.finetune:
        args.lr = 1e-5
        args.epochs = min(args.epochs, 5)
        print(">>> 微调模式：全部层解冻，lr=1e-5")

    # 统一随机性顺序：先播种 -> 建模型 -> 建 DataLoader -> 训练
    net = build_pretrained_alexnet(num_classes=2, finetune=args.finetune)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"可训练参数 {trainable:,} / 总参数 {total:,}")

    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size,
                                           seed=args.seed)
    print(f"训练 {len(train_loader.dataset)} 张 / 验证 {len(val_loader.dataset)} 张")

    acc = train_and_eval(net, train_loader, val_loader,
                         epochs=args.epochs, lr=args.lr, device=args.device)

    tag = "finetune" if args.finetune else "frozen"
    save_path = os.path.join(OUT_DIR, f"step2_alexnet_{tag}.pth")
    torch.save(net.state_dict(), save_path)
    print(f"\n【第二步 预训练{'微调' if args.finetune else '冻结'}】最终准确率: {acc:.4f}")
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
