# -*- coding: utf-8 -*-
"""
step8_crossval.py —— K 折交叉验证：把验证集从 20 张提到 100 张

为什么需要它：
    单次 80/20 划分下，验证集只有 20 张 -> 1 个样本 = 5%。
    0.85 / 0.90 / 0.95 之间只差 1 个样本，无法分辨真实差异（见 step7 消融）。

    K 折交叉验证让**每张图恰好被预测一次**：5 折 × 20 张 = 100 个预测，
    于是池化准确率的分辨率变成 1%，比单次划分细 5 倍。

    做法：按类别分层切成 K 折（保证每折猫狗比例一致），
    轮流拿 1 折当验证集、其余 K-1 折训练，最后把所有验证预测池在一起算准确率。

运行：
    python step8_crossval.py                       # 6 配置 × 3 种子 × 5 折 = 90 次训练
    python step8_crossval.py --seeds 42 1          # 加快
    python step8_crossval.py --folds 5 --configs 3_resnet18_noaug 4_resnet18_aug
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from common import FileNameDataset, default_val_transform, evaluate
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, set_seed
from step7_ablation import CONFIGS, build_net, pick_transform


def stratified_folds(labels, k, seed):
    """按类别分层切 k 折：每个类别各自均分，保证各折类别比例一致"""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    for idxs in by_class.values():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for j, part in enumerate(np.array_split(idxs, k)):
            folds[j].extend(part.tolist())
    return [sorted(f) for f in folds]


def run_cv(cfg, seed, k, device, verbose=False):
    """返回 (池化准确率, 各折准确率, 用时)"""
    base = FileNameDataset(DATA_DIR)
    labels = [0 if "cat" in n.lower() else 1 for n in base.images]
    n = len(labels)
    folds = stratified_folds(labels, k, seed)

    train_tf = pick_transform(cfg["aug"]) or default_val_transform()
    val_tf = default_val_transform()

    pooled_pred = [None] * n
    fold_accs = []
    t0 = time.time()

    for fi, val_idx in enumerate(folds):
        val_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_set]

        # 每折独立播种：模型初始化 + shuffle 都只受本折 seed 影响
        set_seed(seed * 1000 + fi)
        net = build_net(cfg["kind"])
        net.to(device)

        train_loader = DataLoader(
            Subset(FileNameDataset(DATA_DIR, transform=train_tf), train_idx),
            batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            Subset(FileNameDataset(DATA_DIR, transform=val_tf), val_idx),
            batch_size=BATCH_SIZE, shuffle=False)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=cfg.get("lr", 1e-3))

        for _ in range(cfg["epochs"]):
            net.train()
            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                loss_fn(net(X), Y).backward()
                optimizer.step()
            # 与 step1~step4 协议一致：每轮验证一次（消耗一次 DataLoader 迭代器）
            evaluate(net, val_loader, device)

        # 收集本折预测，按 val_idx 顺序回填到全局下标
        net.eval()
        got = 0
        with torch.no_grad():
            for X, _ in val_loader:
                preds = net(X.to(device)).argmax(1).cpu().tolist()
                for p in preds:
                    pooled_pred[val_idx[got]] = p
                    got += 1
        fold_accs.append(accuracy_score([labels[i] for i in val_idx],
                                        [pooled_pred[i] for i in val_idx]))
        if verbose:
            print(f"    fold {fi + 1}/{k} acc={fold_accs[-1]:.4f}")

    assert all(p is not None for p in pooled_pred), "有样本没被预测到"
    pooled = accuracy_score(labels, pooled_pred)
    return pooled, fold_accs, time.time() - t0


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


DEFAULT_CONFIGS = ["1_scratch_alexnet", "2_alexnet_frozen", "3_resnet18_noaug",
                   "4_resnet18_aug", "6_mobilenet_aug", "7_shufflenet_aug"]

PAIRS = [
    ("4_resnet18_aug", "3_resnet18_noaug", "数据增强 (aug - noaug)"),
    ("2_alexnet_frozen", "1_scratch_alexnet", "迁移学习 (预训练 - 从零)"),
    ("6_mobilenet_aug", "4_resnet18_aug", "MobileNet vs ResNet18"),
    ("7_shufflenet_aug", "4_resnet18_aug", "ShuffleNet vs ResNet18"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "crossval_results.csv"))
    args = parser.parse_args()

    n = len(FileNameDataset(DATA_DIR))
    print("=" * 78)
    print(f"{args.folds} 折交叉验证 | 配置 {len(args.configs)} 个 × 种子 {len(args.seeds)} 个"
          f" = {len(args.configs) * len(args.seeds) * args.folds} 次训练")
    print(f"池化后验证样本 {n} 张 -> 分辨率 {1 / n:.1%}（单次划分只有 20 张 / 5%）")
    print(f"device={args.device}")
    print("=" * 78)

    rows, per_seed = [], {}
    for name in args.configs:
        cfg = CONFIGS[name]
        pooled_list, fold_all, secs = [], [], []
        for s in args.seeds:
            p, fa, sec = run_cv(cfg, s, args.folds, args.device)
            pooled_list.append(p)
            fold_all.extend(fa)
            secs.append(sec)
            print(f"  {name:22s} seed={s:<3d} 池化准确率={p:.4f} "
                  f"(各折 {[f'{a:.2f}' for a in fa]})  {sec:5.1f}s")
        per_seed[name] = pooled_list
        rows.append({
            "config": name, "epochs": cfg["epochs"], "lr": cfg.get("lr", 1e-3),
            "aug": cfg["aug"], "folds": args.folds,
            "pooled_mean": mean(pooled_list), "pooled_std": std(pooled_list),
            "pooled_min": min(pooled_list), "pooled_max": max(pooled_list),
            "fold_mean": mean(fold_all), "fold_std": std(fold_all),
            "sec_mean": mean(secs),
            "pooled_per_seed": ";".join(f"{a:.4f}" for a in pooled_list),
        })
        print(f"  -> {name:22s} 池化 {mean(pooled_list):.4f} ± {std(pooled_list):.4f}"
              f" | 各折均值 {mean(fold_all):.4f} ± {std(fold_all):.4f}\n")

    print("=" * 78)
    print(f"{'配置':24s}{'轮数':>5s}{'池化均值':>10s}{'标准差':>9s}"
          f"{'最小':>8s}{'最大':>8s}{'各折均值':>10s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['config']:24s}{r['epochs']:5d}{r['pooled_mean']:10.4f}"
              f"{r['pooled_std']:9.4f}{r['pooled_min']:8.2f}{r['pooled_max']:8.2f}"
              f"{r['fold_mean']:10.4f}")

    print("\n" + "=" * 78)
    print("配对比较（同种子逐对，基于池化准确率）")
    print("-" * 78)
    pairs = []
    for a, b, label in PAIRS:
        if a not in per_seed or b not in per_seed:
            continue
        d = [x - y for x, y in zip(per_seed[a], per_seed[b])]
        wins = sum(1 for v in d if v > 0)
        print(f"{label:34s} {mean(d):+.4f} ± {std(d):.4f}   "
              f"前者更好的种子 {wins}/{len(d)}")
        pairs.append({"compare": label, "delta_mean": mean(d), "delta_std": std(d),
                      "wins": wins, "n": len(d)})
    print(f"\n注：池化后 1 个样本 = {1 / n:.0%}，比单次划分的 5% 细 5 倍。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "pairs": pairs, "seeds": args.seeds,
                   "folds": args.folds, "n": n}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
