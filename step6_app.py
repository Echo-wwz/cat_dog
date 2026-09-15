# -*- coding: utf-8 -*-
"""
第六步：封装成可拍照分类的小程序（Gradio）（对应指导书 6.2）

运行：
    python step6_app.py
    python step6_app.py --model mobilenet_v3_small --port 7860

启动后：
    本机浏览器      http://127.0.0.1:7860
    手机（同一 Wi-Fi）http://<电脑IP>:7860   ← 脚本会直接把 IP 打出来

依赖：pip install gradio
"""
import argparse
import os
import socket

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import DATA_DIR, DEVICE, OUT_DIR

CLASSES = {0: "🐱 猫", 1: "🐶 狗"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_lan_ip():
    """拿本机在局域网里的 IP，方便手机访问"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_interface(net, device):
    import gradio as gr

    # ---------- 定义预测函数（每次拍照/上传都会调用） ----------
    def predict(img: Image.Image):
        if img is None:
            return "请先拍照或上传一张图片", {}
        img = img.convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(net(x.to(device)), dim=1)[0].cpu()
        conf, pred = probs.max(0)
        detail = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        return f"识别结果：{CLASSES[int(pred)]}（置信度 {conf:.2%}）", detail

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传"),
        outputs=[gr.Textbox(label="识别结果"), gr.Label(label="各类别概率")],
        title="🐱🐶 猫狗分类器",
        description="基于 ResNet18 迁移学习训练。对准猫或狗拍照，或上传图片即可识别。",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=os.path.join(OUT_DIR, "best_model.pth"))
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = parser.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit("缺少 gradio，请先安装：pip install gradio")

    if not os.path.exists(args.weight):
        raise SystemExit(
            f"找不到权重文件 {args.weight}\n请先跑 python step5_train_save.py 生成。")

    # ---------- 1. 加载模型（启动时只做一次） ----------
    from step4_compare import build_model
    net = build_model(args.model, num_classes=2, freeze=False)
    net.load_state_dict(torch.load(args.weight, map_location="cpu"))
    net.to(args.device).eval()
    print(f"已加载权重: {args.weight}（模型 {args.model}）")

    # ---------- 2. 搭建界面并启动 ----------
    demo = build_interface(net, args.device)
    print(f"\n本机访问:      http://127.0.0.1:{args.port}")
    print(f"手机同 Wi-Fi:  http://{get_lan_ip()}:{args.port}\n")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
