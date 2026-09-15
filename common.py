# -*- coding: utf-8 -*-
"""
common.py —— 公共代码（对应指导书 0.2 / 0.3 节）

提供：
    FileNameDataset   从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1
    get_loaders       80% 训练 / 20% 验证，固定 seed，训练集可套数据增强
    train_and_eval    训练 + 评估（准确率/精确率/召回率/F1）
    evaluate          只评估，不训练
"""
import os

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from config import BATCH_SIZE, DATA_DIR, SEED, VAL_RATIO, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not self.images:
            raise FileNotFoundError(
                f"{image_dir} 里没有找到任何 .jpg/.jpeg/.png，请检查数据集路径")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = 0 if "cat" in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label


def default_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                train_transform=None, val_transform=None,
                val_ratio=VAL_RATIO, seed=SEED, num_workers=0):
    """统一划分：80% 训练 / 20% 验证，固定随机种子保证各实验可比。

    注意：增强只作用于训练集，验证集永远用干净的 Resize + Normalize。
    """
    if train_transform is None:
        train_transform = default_train_transform()
    if val_transform is None:
        val_transform = default_val_transform()

    names = FileNameDataset(data_dir).images        # 只为了拿到文件名列表
    n_total = len(names)
    n_train = int((1 - val_ratio) * n_total)
    n_val = n_total - n_train

    train_set = FileNameDataset(image_dir=data_dir, transform=train_transform)
    val_set = FileNameDataset(image_dir=data_dir, transform=val_transform)

    # 只对索引做一次划分，训练/验证各取各的 transform
    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val = random_split(range(n_total), [n_train, n_val], generator=g)

    train_loader = DataLoader(Subset(train_set, idx_train.indices),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(Subset(val_set, idx_val.indices),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(net, loader, device="cpu", loss_fn=None):
    """跑一遍验证集，返回指标字典"""
    net.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = net(X)
        if loss_fn is not None:
            total_loss += loss_fn(logits, Y).item() * X.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(Y.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    return {
        "acc": acc, "precision": p, "recall": r, "f1": f1,
        "loss": (total_loss / n) if loss_fn is not None else None,
    }


def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3,
                   device="cpu", verbose=True, seed=None):
    """训练 epochs 轮，最后返回验证集准确率（兼容指导书 0.3 的用法）。

    关于随机性：本函数**默认不再重新播种**。调用方必须遵守统一约定 ——

        set_seed(seed)  ->  构建模型  ->  构建 DataLoader  ->  train_and_eval()

    顺序很重要：`nn.init.xavier_uniform_` 和 DataLoader 的 shuffle 都消耗全局
    随机数，谁先谁后会让"同一个 seed"跑出不同结果。只有固定这个顺序，
    step1~step7 之间同 seed 的结果才能对齐。
    """
    if seed is not None:          # 仅在你确实想在这里重新播种时传
        set_seed(seed)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        if verbose:
            train_loss = running_loss / len(train_loader.dataset)
            m = evaluate(net, val_loader, device)
            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f} "
                  f"| Val Acc: {m['acc']:.4f} | F1: {m['f1']:.4f}")

    metrics = evaluate(net, val_loader, device)
    if verbose:
        print(f"准确率: {metrics['acc']:.4f} | 精确率: {metrics['precision']:.4f} "
              f"| 召回率: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    return metrics["acc"]


if __name__ == "__main__":
    train_loader, val_loader = get_loaders()
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张")
