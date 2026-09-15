# -*- coding: utf-8 -*-
"""
第五步（下）：单张图片推理测试（对应指导书 5.3）

用法（参数顺序与指导书一致）：
    python step5_predict.py                                    # 默认权重 + 默认图片
    python step5_predict.py outputs/best_model.pth data/dog.77.jpg
    python step5_predict.py --model mobilenet_v3_small --dir "data"

--model 必须与权重文件对应的架构一致，对照表：

    outputs/best_model.pth              --model resnet18            (默认)
    outputs/model_resnet18.pth          --model resnet18
    outputs/model_mobilenet_v3_small.pth --model mobilenet_v3_small
    outputs/model_shufflenet_v2_x1_0.pth --model shufflenet_v2_x1_0
    outputs/step1_alexnet_scratch.pth   --model alexnet_scratch    ← 手写版
    outputs/step2_alexnet_frozen.pth    --model alexnet            ← torchvision 版
    outputs/step2_alexnet_finetune.pth  --model alexnet
    outputs/step3_resnet18_aug.pth      --model resnet18
    outputs/step3_resnet18_noaug.pth    --model resnet18

相对指导书的修正：
  1. `models.resnet18(weights=None)` 写死了 resnet18；这里加 --model，
     权重文件是哪个模型训的就选哪个，否则 state_dict 键名对不上。
  2. 默认图片路径改成真实存在的数据集目录。
  3. 增加 --dir 批量模式：跑完整个文件夹并给出准确率，方便验收。
  4. 补上 AlexNet 支持（原版只能跑 torchvision 四模型，训好的 AlexNet 权重没法推理）。
     注意 AlexNet 有两套键名，见 build_net 的说明。
  5. --model 与权重不匹配时给出明确提示，而不是抛一大段 state_dict 报错。
"""
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import DATA_DIR, DEVICE, OUT_DIR

CLASSES = ["cat 猫", "dog 狗"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_net(model_name="resnet18", num_classes=2):
    """按权重文件对应的架构重建模型。

    ⚠️ AlexNet 有两套结构，state_dict 键名不同，选错会加载失败：
      alexnet         -> torchvision 版（features.* / classifier.*）
                         配 outputs/step2_alexnet_frozen.pth / step2_alexnet_finetune.pth
      alexnet_scratch -> 手写版（net.*），通道数 96/256/384/384/256，flatten 6400
                         配 outputs/step1_alexnet_scratch.pth
    """
    if model_name == "alexnet":
        from step2_pretrained import AlexNetManual
        return AlexNetManual(num_classes=num_classes)
    if model_name == "alexnet_scratch":
        from step1_scratch import AlexNet
        return AlexNet()
    from step4_compare import build_model
    return build_model(model_name, num_classes=num_classes, freeze=False)


def load_model(weight_path, model_name="resnet18", device=DEVICE):
    net = build_net(model_name, num_classes=2)
    state = torch.load(weight_path, map_location="cpu")
    try:
        net.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(
            f"\n权重与模型结构不匹配（--model {model_name}）\n"
            f"  权重文件 : {weight_path}\n"
            f"  首行错误 : {str(e).splitlines()[0]}\n"
            "请确认 --model 与训练该权重时用的模型一致，对照表见本文件头部注释。")
    net.to(device)          # ← 必须和 predict 里 x 的 device 一致，否则报
    net.eval()              #   "Input type (torch.cuda.FloatTensor) and weight
    return net              #    type (torch.FloatTensor) should be the same"


def predict(image_path, net, device=DEVICE):
    img = Image.open(image_path).convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0)             # 增加 batch 维 -> [1,3,224,224]
    with torch.no_grad():
        probs = torch.softmax(net(x.to(device)), dim=1)[0].cpu()   # 得分转概率
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", nargs="?", default=os.path.join(OUT_DIR, "best_model.pth"))
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("--model", default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0",
                                 "efficientnet_b0", "alexnet", "alexnet_scratch"],
                        help="必须与权重文件对应的架构一致（默认 resnet18）")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dir", default=None, help="批量测试整个文件夹")
    args = parser.parse_args()

    if not os.path.exists(args.weight):
        raise SystemExit(
            f"找不到权重文件 {args.weight}\n请先跑 python step5_train_save.py 生成。")

    net = load_model(args.weight, args.model, args.device)

    # ---- 批量模式：跑完整个目录，输出准确率 ----
    if args.dir:
        files = sorted(f for f in os.listdir(args.dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        correct = 0
        print(f"批量测试 {len(files)} 张（{args.dir}）\n")
        for f in files:
            label, conf, _ = predict(os.path.join(args.dir, f), net, args.device)
            gt = "cat 猫" if "cat" in f.lower() else "dog 狗"
            ok = label == gt
            correct += ok
            print(f"  {'OK ' if ok else 'ERR'} {f:16s} 预测: {label} ({conf:.2%})"
                  f"{'' if ok else f'  真实: {gt}'}")
        print(f"\n准确率: {correct / len(files):.4f} ({correct}/{len(files)})")
        return

    # ---- 单张模式 ----
    path = args.image or os.path.join(DATA_DIR, "cat.1.jpg")
    label, conf, probs = predict(path, net, args.device)
    print(f"图片: {path}")
    print(f"预测: {label}，置信度 {conf:.2%}")
    print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")


if __name__ == "__main__":
    main()
