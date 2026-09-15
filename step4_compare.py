# -*- coding: utf-8 -*-
"""
第四步：调用 torchvision.models 封装好的模型横向对比（对应指导书 4.2）

相对指导书的修正：
  1. 补 `import torch`（原文只有 `import torch.nn as nn`，后面却用了 torch.save）
  2. build_model 里加了未知模型名的兜底报错，避免 name 打错时静默出错
  3. 顺手记录每轮训练用时 —— 指导书那张对比表要填"每轮训练用时"这一列

运行：
    python step4_compare.py
    python step4_compare.py --models resnet18 mobilenet_v3_small
    python step4_compare.py --no-freeze     # 全模型微调（慢，慎用）
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torchvision import models

from common import evaluate, get_loaders
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, SEED, set_seed

CONSTRUCTORS = {
    "resnet18":           lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
    "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    "efficientnet_b0":    lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
}


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头"""
    if name not in CONSTRUCTORS:
        raise KeyError(f"不支持的模型: {name}，可选: {list(CONSTRUCTORS)}")
    net = CONSTRUCTORS[name]()

    # 找到并替换分类头（不同模型的头名字不一样）
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "efficientnet_b0":
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        head_params = net.classifier[1].parameters()

    if freeze:  # 冻结特征提取层，只训练分类头（CPU 友好）
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


def train_and_eval_timed(net, train_loader, val_loader, epochs, lr, device, seed=None):
    """训练 + 计时，返回 (acc, 每轮平均秒数)"""
    if seed is not None:
        set_seed(seed)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
        m = evaluate(net, val_loader, device)
        print(f"  Epoch [{epoch + 1}/{epochs}] Val Acc: {m['acc']:.4f}")
    sec_per_epoch = (time.time() - t0) / epochs

    m = evaluate(net, val_loader, device)
    print(f"  准确率 {m['acc']:.4f} | 精确率 {m['precision']:.4f} "
          f"| 召回率 {m['recall']:.4f} | F1 {m['f1']:.4f}")
    return m["acc"], sec_per_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"])
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size,
                                           seed=args.seed)
    print(f"训练 {len(train_loader.dataset)} 张 / 验证 {len(val_loader.dataset)} 张"
          f" | device={args.device} | seed={args.seed}")

    results = {}
    for name in args.models:
        print(f"\n===== 模型: {name} =====")
        # 每个模型都从同一个随机状态出发，初始化和打乱顺序完全一致
        set_seed(args.seed)
        net = build_model(name, freeze=not args.no_freeze)
        n_param = sum(p.numel() for p in net.parameters())
        print(f"  参数量: {n_param / 1e6:.2f}M")
        acc, sec = train_and_eval_timed(net, train_loader, val_loader,
                                        args.epochs, args.lr, args.device)
        results[name] = (acc, sec, n_param)
        torch.save(net.state_dict(), os.path.join(OUT_DIR, f"model_{name}.pth"))

    print("\n===== 横向对比结果 =====")
    print(f"{'模型':24s} {'参数量':>10s} {'准确率':>8s} {'每轮用时':>10s}")
    for name, (acc, sec, n_param) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{name:24s} {n_param / 1e6:9.2f}M {acc:8.4f} {sec:9.2f}s")

    print("\n> 小数据集上三个模型准确率可能相差不大，此时推理速度和模型体积才是选型关键。")


if __name__ == "__main__":
    main()
