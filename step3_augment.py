# -*- coding: utf-8 -*-
"""
第三步：加入数据增强（对应指导书 3.2）

三个实验横向对比建议：
    A 从零训练（step1）
    B 预训练+冻结（step2）
    C 预训练+冻结+数据增强（本文件）

相对指导书的修正：
  1. `net.load_state_dict(official.state_dict(), strict=False)` 在 fc 形状
     不一致时会**直接报错**（strict=False 不豁免 size mismatch）。
     改成先建 1000 类结构 → 加载 → 再换头（见 models_manual.py 的说明）。
  2. `from step1_scratch import ResNet18Manual` 是悬空引用，改从
     models_manual.py 导入（那个文件里才是真的 ResNet18Manual）。
  3. 验证集保持干净的 Resize+Normalize，增强只作用于训练集。

运行：
    python step3_augment.py
    python step3_augment.py --epochs 10
    python step3_augment.py --no-augment      # 对照组：不加增强，直接对比
"""
import argparse
import os

from torchvision import transforms

from common import IMAGENET_MEAN, IMAGENET_STD, get_loaders, train_and_eval
from config import BATCH_SIZE, DATA_DIR, DEVICE, OUT_DIR, SEED, set_seed
from models_manual import load_pretrained_resnet18
import torch


def build_augment_transform(random_erasing=False):
    """训练集增强：只造假样本给训练用，验证集绝不能用"""
    ops = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),                    # 随机水平翻转
        transforms.ColorJitter(0.2, 0.2, 0.2),                # 随机亮度/对比度/饱和度
        transforms.RandomRotation(10),                        # 随机小角度旋转
        transforms.ToTensor(),
    ]
    if random_erasing:      # 思考题 2：随机遮挡一块，逼模型别只看局部
        ops.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)))
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-augment", action="store_true", help="对照组：关掉增强")
    parser.add_argument("--random-erasing", action="store_true",
                        help="在增强基础上再加 RandomErasing（思考题 2）")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.no_augment:
        train_transform, tag = None, "noaug"
    elif args.random_erasing:
        train_transform, tag = build_augment_transform(True), "erase"
    else:
        train_transform, tag = build_augment_transform(False), "aug"

    # 统一随机性顺序：先播种 -> 建模型 -> 建 DataLoader -> 训练
    net = load_pretrained_resnet18(num_classes=2, freeze=True)

    train_loader, val_loader = get_loaders(
        args.data_dir, batch_size=args.batch_size,
        train_transform=train_transform, seed=args.seed)
    print(f"训练 {len(train_loader.dataset)} 张 / 验证 {len(val_loader.dataset)} 张"
          f" | 增强配置: {tag} | seed={args.seed}")

    acc = train_and_eval(net, train_loader, val_loader,
                         epochs=args.epochs, lr=args.lr, device=args.device)

    save_path = os.path.join(OUT_DIR, f"step3_resnet18_{tag}.pth")
    torch.save(net.state_dict(), save_path)
    print(f"\n【第三步 预训练+{tag}】最终准确率: {acc:.4f}")
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
