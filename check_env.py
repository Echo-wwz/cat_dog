# -*- coding: utf-8 -*-
"""
check_env.py —— 跑实验前先自检：依赖、设备、数据、文件数

    python check_env.py
"""
import importlib
import os
import sys

from config import DATA_DIR, DEVICE, OUT_DIR

REQUIRED = ["torch", "torchvision", "PIL", "sklearn", "numpy"]
OPTIONAL = ["gradio"]

ok = True

print("=" * 56)
print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
print("-" * 56)
for m in REQUIRED:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"[OK]   {m:12s} {ver}")
    except ImportError as e:
        ok = False
        print(f"[MISS] {m:12s} -> {e}")
for m in OPTIONAL:
    try:
        mod = importlib.import_module(m)
        print(f"[OK]   {m:12s} {getattr(mod, '__version__', '?')}  (第六步需要)")
    except ImportError:
        print(f"[可选] {m:12s} 未安装 -> pip install gradio  (第六步需要)")

print("-" * 56)
print(f"torch 设备  : {DEVICE}")
try:
    import torch
    print(f"CUDA 可用   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
except Exception:
    pass

print("-" * 56)
print(f"数据目录    : {DATA_DIR}")
if os.path.isdir(DATA_DIR):
    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    cats = sum(1 for f in files if "cat" in f.lower())
    dogs = sum(1 for f in files if "dog" in f.lower())
    print(f"图片总数    : {len(files)}  (猫 {cats} / 狗 {dogs})")
    if len(files) == 0:
        ok = False
        print("  !! 目录里没有图片")
    elif cats == 0 or dogs == 0:
        ok = False
        print("  !! 两类不全，文件名需形如 cat.1.jpg / dog.1.jpg")
else:
    ok = False
    print("  !! 目录不存在。改 config.py 里的 DATA_DIR，"
          "或设置环境变量 CATDOG_DATA 指向数据集目录")

print("-" * 56)
# ---- 预训练权重缓存（torch.hub 决定，见 TORCH_HOME）----
try:
    import torch as _torch
    ckpt_dir = os.path.join(_torch.hub.get_dir(), "checkpoints")
    if os.environ.get("TORCH_HOME"):
        src = f"环境变量 TORCH_HOME={os.environ['TORCH_HOME']}"
    elif os.environ.get("XDG_CACHE_HOME"):
        src = f"环境变量 XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}"
    else:
        src = "默认位置 ~/.cache（未设 TORCH_HOME）"
    print(f"权重缓存    : {ckpt_dir}")
    print(f"  来源      : {src}")

    WANT = {
        "alexnet-owt-7be5be79.pth": "alexnet（第一/二步）",
        "resnet18-f37072fd.pth": "resnet18（第三/四步）",
        "mobilenet_v3_small-047dcff4.pth": "mobilenet（第四步）",
        "shufflenetv2_x1-5666bf0f80.pth": "shufflenet（第四步）",
    }
    missing = []
    for fn, desc in WANT.items():
        p = os.path.join(ckpt_dir, fn)
        if os.path.isfile(p):
            print(f"  [有] {fn:38s} {os.path.getsize(p) / 1e6:6.1f} MB  {desc}")
        else:
            missing.append(fn)
            print(f"  [缺] {fn:38s} {'--':>6s}      {desc}")
    if missing:
        print(f"  -> 缺 {len(missing)} 个；第一次跑对应步骤会联网下载，属正常")
        print("  -> 想换位置：设 TORCH_HOME，checkpoints 落在 $TORCH_HOME/hub/checkpoints")
except Exception as e:
    print(f"权重缓存    : 探测失败 -> {e}")

print("-" * 56)
print(f"输出目录    : {OUT_DIR}")
print("=" * 56)
print("环境检查通过 ✔" if ok else "环境有问题 ✘，请先按上面的提示处理")
sys.exit(0 if ok else 1)
