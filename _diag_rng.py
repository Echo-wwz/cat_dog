# -*- coding: utf-8 -*-
"""
_diag_rng.py —— 诊断：为什么 step1_scratch.py --seed 42 与消融里的
1_scratch_alexnet seed=42 结果不同？

假设：step1 的训练循环**每轮都跑一次验证**，而消融的循环不验证。
      验证会为 val_loader 新建一个 DataLoader 迭代器，而 DataLoader 的
      每个迭代器都会用全局随机数生成 base_seed —— 于是从第 2 轮起，
      两条路径的 shuffle 顺序就分叉了。

本脚本直接验证这个假设：跑三种变体，看哪种能复现 step1 的 0.80。
"""
import torch
import torch.nn as nn

from common import evaluate, get_loaders
from config import BATCH_SIZE, DATA_DIR, DEVICE, set_seed
from step1_scratch import AlexNet, init_weight

EPOCHS = 30


def run(seed, validate_each_epoch, label):
    """复刻 step1 的训练协议，只切换'每轮是否验证'"""
    set_seed(seed)
    net = AlexNet()
    net.apply(init_weight)
    net.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    train_iter, test_iter = get_loaders(DATA_DIR, batch_size=BATCH_SIZE, seed=seed)

    first_batch = None
    for epoch in range(EPOCHS):
        net.train()
        for X, Y in train_iter:
            if first_batch is None:
                first_batch = Y.tolist()[:5]
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(net(X), Y).backward()
            optimizer.step()
        if validate_each_epoch:
            evaluate(net, test_iter, DEVICE)

    acc = evaluate(net, test_iter, DEVICE)["acc"]
    print(f"{label:34s} seed={seed} acc={acc:.4f} 首个 batch 前 5 个标签={first_batch}")
    return acc


if __name__ == "__main__":
    print("=" * 78)
    a = run(42, True, "每轮验证（= step1_scratch.py）")
    b = run(42, False, "不验证（= 消融 run_once）")
    print("-" * 78)
    print(f"两者是否一致: {'是' if abs(a - b) < 1e-9 else '否'}")
    print("=" * 78)
