# -*- coding: utf-8 -*-
"""
第五步（上）：训练时保存最佳模型（对应指导书 5.2）

相对指导书的修正：
  1. 原文 device 写死 "cpu"，且训练循环里 `net(X)` 没把 X 搬到 device ——
     一旦换 GPU 就会报 "expected all tensors to be on the same device"。
     这里统一搬到 device。
  2. 加 --model 参数，可选 resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0，
     和第四步选出来的模型保持一致（第五、六步默认用 resnet18）。

运行：
    python step5_train_save.py
    python step5_train_save.py --model mobilenet_v3_small --epochs 10
输出：
    outputs/best_model.pth   ← 验证集准确率最高的那一份权重
"""
import argparse
import os

import torch
import torch.nn as nn

from common import evaluate, get_loaders
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, set_seed
from step4_compare import build_model


def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3,
                         device="cpu", save_path="best_model.pth"):
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)      # ← 修正：搬到同一设备
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        # 验证
        metrics = evaluate(net, val_loader, device)
        acc = metrics["acc"]
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: "
              f"{running_loss / len(train_loader.dataset):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:                      # 只保留历史最佳（早停思想）
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")

    print(f"最佳准确率: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small",
                                 "shufflenet_v2_x1_0", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--save-name", default="best_model.pth")
    args = parser.parse_args()

    set_seed()
    train_loader, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size)

    net = build_model(args.model, num_classes=2, freeze=True)
    save_path = os.path.join(OUT_DIR, args.save_name)
    # 默认保存名 best_model.pth 对应的是 resnet18。换模型时不换名字会把它覆盖掉，
    # 之后 step5_predict.py 按默认 --model resnet18 加载就会键名不匹配。
    if args.model != "resnet18" and args.save_name == "best_model.pth":
        print(f"提示: 当前模型是 {args.model}，保存名仍是 best_model.pth（默认对应 resnet18）。")
        print(f"      建议改用 --save-name best_model_{args.model}.pth，"
              f"并在推理时配 --model {args.model}。\n")
    print(f"模型: {args.model} | device={args.device} | 保存到 {save_path}\n")
    train_with_best_save(net, train_loader, val_loader, epochs=args.epochs,
                         lr=args.lr, device=args.device, save_path=save_path)


if __name__ == "__main__":
    main()
