# -*- coding: utf-8 -*-
"""
全局配置：路径 / 超参数 / 设备

设计原则：数据集和权重缓存都放在项目内，整个 cat_dog 文件夹可以随便搬。
"""
import os
import random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 权重缓存：必须在任何 models.xxx(weights=DEFAULT) 调用之前设好 TORCH_HOME
# ---------------------------------------------------------------------------
# torch 解析规则（实测 2.13.0）：
#     checkpoints 目录 = $TORCH_HOME/hub/checkpoints      <- 注意多一层 hub
#     torch.hub.set_dir(d) 则落在  d/checkpoints          <- 少一层，别混用
# 优先级：TORCH_HOME > XDG_CACHE_HOME > ~/.cache
#
# 这里优先用项目自带的 torch_cache（self-contained，换机器直接拷文件夹）。
# 想临时换到别处，设环境变量 CATDOG_TORCH_HOME 即可覆盖。
_LOCAL_CACHE = os.path.join(HERE, "torch_cache")
_LOCAL_CKPT = os.path.join(_LOCAL_CACHE, "hub", "checkpoints")

_override = os.environ.get("CATDOG_TORCH_HOME")
if _override:
    os.environ["TORCH_HOME"] = _override
elif os.path.isdir(_LOCAL_CKPT):
    os.environ["TORCH_HOME"] = _LOCAL_CACHE

# 放到 import torch 之前，稳妥（实测 get_dir() 是惰性的，放后面其实也行）
import torch  # noqa: E402

CKPT_DIR = os.path.join(torch.hub.get_dir(), "checkpoints")

OUT_DIR = os.path.join(HERE, "outputs")          # 权重、索引文件、日志都丢这儿
os.makedirs(OUT_DIR, exist_ok=True)


def _find_data_dir():
    """按优先级找一个存在的数据集目录（全部用绝对路径，和 CWD 无关）"""
    candidates = [
        os.environ.get("CATDOG_DATA"),
        os.path.join(HERE, "data"),        # 本项目：cat.1.jpg / dog.1.jpg
        os.path.join(HERE, "datasets"),    # 指导书里的 ./datasets
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return candidates[1]      # 都没有就返回 ./data，让报错信息直观一点


DATA_DIR = _find_data_dir()

# ---- 控制变量：全程固定同一份数据划分 ----
SEED = 42
VAL_RATIO = 0.2
BATCH_SIZE = 16

# ---- 设备：有 GPU 就用 GPU，100 张图在 CPU 上跑 AlexNet×30 轮会等到怀疑人生 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED):
    """一次性固定所有随机源，保证每次划分/初始化一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"OUT_DIR  = {OUT_DIR}")
    print(f"CKPT_DIR = {CKPT_DIR}")
    print(f"TORCH_HOME = {os.environ.get('TORCH_HOME', '(未设，用默认 ~/.cache)')}")
    print(f"DEVICE   = {DEVICE}")
    print(f"torch    = {torch.__version__}")
