# -*- coding: utf-8 -*-
"""
第一步：手动搭建 AlexNet，从零训练（对应指导书 1.2 + "完整代码"）

相对指导书的修正：
  1. 补上 `class AlexNet(nn.Module)` 缺的冒号和类体（原文那行是占位符）
  2. `random.sample` 前固定随机种子 —— 否则每次划分都变，"控制变量"无从谈起
  3. 默认改用 common.get_loaders 做数据划分，这样第一步~第六步用的是
     **同一份 80/20 划分**，横向对比才成立（指导书里 step1 的 txt 划分和
     后面几步的 random_split 其实是两份不同的数据）
     想跑指导书原版 txt 索引写法：加 --split txt
  4. 自动选设备，--device cpu 可强制复现课堂 CPU 环境

运行：
    python step1_scratch.py                  # 默认 30 轮
    python step1_scratch.py --epochs 5       # 快速冒烟
    python step1_scratch.py --split txt      # 指导书原版 txt 索引写法
"""
import argparse
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import get_loaders
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, SEED, set_seed


# ============ 1. 定义模型 AlexNet（指导书 1.2 的类体）============
class AlexNet(nn.Module):
    def __init__(self):
        """初始化"""
        super().__init__()
        self.net = nn.Sequential(
            # 11*11 的大窗口捕捉对象，步幅 4 压缩高宽，通道数远大于 LeNet
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，padding=2 保持高宽，增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 三个连续卷积 + 小窗口，通道数进一步增加
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 全连接层用 dropout 减轻过拟合。224 输入 -> 256*5*5 = 6400
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 输出层：猫狗二分类
            nn.Linear(4096, 2),
        )

    def forward(self, X):
        """前向传播"""
        return self.net(X)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


# ============ 2'. 指导书原版：从 txt 索引读数据集 ============
class DatasetLoader(Dataset):
    """txt 每行形如：D:\\path\\cat.1.jpg 0"""

    def __init__(self, dataset_path):
        self.images_path = []
        self.labels = []
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    self.images_path.append(parts[0])
                    self.labels.append(int(parts[1]))

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return image_transform(image)

    def __getitem__(self, index):
        return self.preprocess_image(self.images_path[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)


def build_index_files(base_path, test_ratio=0.2, seed=SEED):
    imgs_abs_path = sorted(
        os.path.join(base_path, i) for i in os.listdir(base_path)
        if i.lower().endswith((".jpg", ".jpeg", ".png")))
    random.seed(seed)                       # ← 修正：固定划分
    test_files = set(random.sample(imgs_abs_path, int(test_ratio * len(imgs_abs_path))))

    train_txt = os.path.join(OUT_DIR, "trainDatasets.txt")
    test_txt = os.path.join(OUT_DIR, "testDatasets.txt")
    with open(test_txt, "w", encoding="utf-8") as f1, \
         open(train_txt, "w", encoding="utf-8") as f2:
        for i in imgs_abs_path:
            label = 0 if "cat" in os.path.basename(i).lower() else 1
            (f1 if i in test_files else f2).write(f"{i} {label}\n")
    return train_txt, test_txt


def dataloader_from_txt(base_path, batch_size=BATCH_SIZE, seed=SEED):
    train_txt, test_txt = build_index_files(base_path, seed=seed)
    train_ds, test_ds = DatasetLoader(train_txt), DatasetLoader(test_txt)
    print(f"训练 {len(train_ds)} 张 / 测试 {len(test_ds)} 张（txt 索引）")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", choices=["torch", "txt"], default="torch",
                        help="torch=统一划分(推荐) / txt=指导书原版 txt 索引")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device

    # 2. 实例化自定义 AlexNet 模型
    net = AlexNet()
    net.apply(init_weight)
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 3. 加载数据集
    print(f"数据目录: {args.data_dir}")
    if args.split == "txt":
        train_iter, test_iter = dataloader_from_txt(args.data_dir, args.batch_size,
                                                    seed=args.seed)
    else:
        train_iter, test_iter = get_loaders(args.data_dir, batch_size=args.batch_size,
                                            seed=args.seed)
        print(f"训练 {len(train_iter.dataset)} 张 / 测试 {len(test_iter.dataset)} 张")

    # 4. 模型训练循环
    print(f"开始训练... (device={device}, epochs={args.epochs}, lr={args.lr})")
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_iter:
            X, Y = X.to(device), Y.to(device)
            l = loss_fn(net(X), Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item() * X.size(0)

        # 5. 验证与评价指标计算
        net.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat = net(X)
                val_loss += loss_fn(y_hat, Y).item() * X.size(0)
                all_preds.extend(y_hat.argmax(1).cpu().numpy())
                all_labels.extend(Y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0)

        print(f"EPOCH: {epoch + 1}/{args.epochs}")
        print(f"  -> Train Loss: {running_loss / len(train_iter.dataset):.4f}")
        print(f"  -> Test Loss: {val_loss / len(test_iter.dataset):.4f}")
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} "
              f"| Recall: {recall:.4f} | F1-Score: {f1:.4f}\n")

    # 保存时按划分方式区分文件名。
    # 原来两种划分都写 step1_alexnet_scratch.pth，导致 run_all 跑完任务 1 再跑 1b 时，
    # 默认划分的模型被 --split txt 的模型覆盖（两者准确率不同，0.80 vs 0.75），
    # 后续推理/分析会静默地用错模型。
    if args.split == "txt":
        save_name = "step1_alexnet_scratch_txt.pth"
    else:
        save_name = "step1_alexnet_scratch.pth"
    save_path = os.path.join(OUT_DIR, save_name)
    torch.save(net.state_dict(), save_path)
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
