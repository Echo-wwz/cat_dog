# -*- coding: utf-8 -*-
"""
step7_ablation.py —— 多随机种子消融实验（补强统计可信度）

为什么需要它：
    验证集只有 20 张 -> 1 个样本 = 5%。单次运行的 0.85/0.90/0.95 之间
    只差 1 个样本，换种子就可能翻转。要判断"增强到底有没有用"，
    必须跑多个种子看 mean ± std，并做**配对比较**（同一划分下逐种子对比）。

本脚本一次性在同一个进程里跑完整矩阵，输出：
    1. 每个配置的 mean / std / min / max
    2. 配对差值（增强 - 无增强，逐种子配对）
    3. CSV + JSON 落盘，供报告引用

运行：
    python step7_ablation.py                      # 7 配置 × 5 种子
    python step7_ablation.py --seeds 42 1 2       # 自定义种子
    python step7_ablation.py --configs 4_resnet18_aug 3_resnet18_noaug
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torchvision import transforms

from common import IMAGENET_MEAN, IMAGENET_STD, evaluate, get_loaders
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, SEED, set_seed
from models_manual import load_pretrained_resnet18
from step1_scratch import AlexNet, init_weight
from step2_pretrained import build_pretrained_alexnet
from step3_augment import build_augment_transform
from step4_compare import build_model

# ---------------------------------------------------------------- 配置矩阵
# lr 必须和对应脚本里的默认值一致，否则就不是"同一个实验"了：
#   从零训练 / 预训练冻结 -> 1e-4（指导书值）；ResNet18 等 -> 1e-3；微调 -> 1e-5
CONFIGS = {
    "1_scratch_alexnet":    {"kind": "scratch",    "aug": "none",  "epochs": 30,
                             "lr": 1e-4},
    "2_alexnet_frozen":     {"kind": "alexnet",    "aug": "none",  "epochs": 5,
                             "lr": 1e-4},
    "3_resnet18_noaug":     {"kind": "resnet18",   "aug": "none",  "epochs": 5,
                             "lr": 1e-3},
    "4_resnet18_aug":       {"kind": "resnet18",   "aug": "basic", "epochs": 5,
                             "lr": 1e-3},
    "5_resnet18_aug_erase": {"kind": "resnet18",   "aug": "erase", "epochs": 5,
                             "lr": 1e-3},
    "6_mobilenet_aug":      {"kind": "mobilenet",  "aug": "basic", "epochs": 5,
                             "lr": 1e-3},
    "7_shufflenet_aug":     {"kind": "shufflenet", "aug": "basic", "epochs": 5,
                             "lr": 1e-3},
    # 与 2_alexnet_frozen 同轮数，才能公平比较"冻结 vs 微调"
    "8_alexnet_finetune":   {"kind": "alexnet_ft", "aug": "none",  "epochs": 5,
                             "lr": 1e-5},
    # 轮数加长版：检验"增强在 5 轮时拖后腿"是不是因为模型还没学会
    "9_resnet18_noaug_15":  {"kind": "resnet18",   "aug": "none",  "epochs": 15,
                             "lr": 1e-3},
    "10_resnet18_aug_15":   {"kind": "resnet18",   "aug": "basic", "epochs": 15,
                             "lr": 1e-3},
}


def pick_transform(aug):
    if aug == "none":
        return None
    if aug == "basic":
        return build_augment_transform(False)
    if aug == "erase":
        return build_augment_transform(True)
    raise ValueError(aug)


def build_net(kind):
    if kind == "scratch":
        net = AlexNet()
        net.apply(init_weight)
        return net
    if kind == "alexnet":
        return build_pretrained_alexnet(num_classes=2, finetune=False)
    if kind == "alexnet_ft":
        return build_pretrained_alexnet(num_classes=2, finetune=True)
    if kind == "resnet18":
        return load_pretrained_resnet18(num_classes=2, freeze=True)
    if kind == "mobilenet":
        return build_model("mobilenet_v3_small", freeze=True)
    if kind == "shufflenet":
        return build_model("shufflenet_v2_x1_0", freeze=True)
    raise ValueError(kind)


def run_once(cfg, seed, device):
    """跑一次，返回 (val_acc, final_train_loss, 秒数)

    随机性顺序必须和 step1~step4 一致，否则同一个 seed 会跑出不同结果：
        set_seed(seed) -> 构建模型 -> 构建 DataLoader -> 训练
    """
    set_seed(seed)

    net = build_net(cfg["kind"])
    net.to(device)

    train_loader, val_loader = get_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, train_transform=pick_transform(cfg["aug"]),
        seed=seed)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-3))

    t0 = time.time()
    train_loss = 0.0
    for _ in range(cfg["epochs"]):
        net.train()
        running = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)
        # 每轮验证一次 —— 必须和 step1~step4 的协议一致。
        # DataLoader 每个迭代器都会用全局随机数生成 base_seed，
        # 少一次迭代（=少消耗一次随机数）就会让后续 shuffle 顺序分叉，
        # 同一个 seed 跑出不同结果。（已用 _diag_rng.py 验证）
        evaluate(net, val_loader, device)
    secs = time.time() - t0

    return evaluate(net, val_loader, device)["acc"], train_loss, secs


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "ablation_results.csv"))
    args = parser.parse_args()

    n_val = int(0.2 * 100)
    print("=" * 78)
    print(f"消融实验 | 配置 {len(args.configs)} 个 × 种子 {len(args.seeds)} 个 "
          f"= {len(args.configs) * len(args.seeds)} 次运行")
    print(f"device={args.device} | 验证集 {n_val} 张 -> 1 个样本 = {1 / n_val:.0%}")
    print("=" * 78)

    rows, detail = [], {}
    for name in args.configs:
        cfg = CONFIGS[name]
        accs, losses, secs = [], [], []
        for s in args.seeds:
            acc, tl, sec = run_once(cfg, s, args.device)
            accs.append(acc)
            losses.append(tl)
            secs.append(sec)
            print(f"  {name:22s} seed={s:<3d} acc={acc:.4f} "
                  f"train_loss={tl:.4f} {sec:5.1f}s")
        detail[name] = {"accs": accs, "seeds": args.seeds}
        rows.append({
            "config": name, "epochs": cfg["epochs"], "lr": cfg.get("lr", 1e-3),
            "aug": cfg["aug"],
            "acc_mean": mean(accs), "acc_std": std(accs),
            "acc_min": min(accs), "acc_max": max(accs),
            "train_loss_mean": mean(losses), "sec_mean": mean(secs),
            "accs": ";".join(f"{a:.4f}" for a in accs),
        })
        print(f"  -> {name:22s} mean={mean(accs):.4f} ± {std(accs):.4f} "
              f"(min {min(accs):.2f} / max {max(accs):.2f})\n")

    # ---------------------------------------------------------- 汇总表
    print("=" * 78)
    print(f"{'配置':24s}{'轮数':>5s}{'lr':>8s}{'均值':>9s}{'标准差':>9s}"
          f"{'最小':>8s}{'最大':>8s}{'训练loss':>10s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['config']:24s}{r['epochs']:5d}{r['lr']:8.0e}{r['acc_mean']:9.4f}"
              f"{r['acc_std']:9.4f}{r['acc_min']:8.2f}{r['acc_max']:8.2f}"
              f"{r['train_loss_mean']:10.4f}")

    # ---------------------------------------------------------- 配对比较
    def paired(a, b):
        """a - b，逐种子配对（必须同种子，划分才一致）"""
        da = detail[a]["accs"]
        db = detail[b]["accs"]
        diffs = [x - y for x, y in zip(da, db)]
        wins = sum(1 for d in diffs if d > 0)
        return mean(diffs), std(diffs), wins, len(diffs)

    pairs = [
        ("4_resnet18_aug", "3_resnet18_noaug", "数据增强 (aug - noaug)"),
        ("5_resnet18_aug_erase", "4_resnet18_aug", "RandomErasing (erase - aug)"),
        ("2_alexnet_frozen", "1_scratch_alexnet", "迁移学习 (预训练 - 从零)"),
        ("2_alexnet_frozen", "8_alexnet_finetune", "冻结 vs 微调（同为 5 轮）"),
        ("10_resnet18_aug_15", "9_resnet18_noaug_15", "数据增强 @15 轮 (aug - noaug)"),
    ]
    print("\n" + "=" * 78)
    print("配对比较（同一划分下逐种子对比，差值 mean ± std）")
    print("-" * 78)
    summary = []
    for a, b, label in pairs:
        if a not in detail or b not in detail:
            continue
        d, sd, wins, n = paired(a, b)
        verdict = "无显著差异" if abs(d) < 0.05 else ("A 更好" if d > 0 else "B 更好")
        print(f"{label:34s} {d:+.4f} ± {sd:.4f}   "
              f"前者更好的种子 {wins}/{n}   -> {verdict}")
        summary.append({"compare": label, "delta_mean": d, "delta_std": sd,
                        "wins": wins, "n": n, "verdict": verdict})

    print("\n注：差值小于 1 个验证样本（{:.0%}）时，不应作为结论。".format(1 / n_val))

    # ---------------------------------------------------------- 落盘
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "pairs": summary, "seeds": args.seeds,
                   "n_val": n_val}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
