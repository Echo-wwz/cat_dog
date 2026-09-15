# 猫狗分类实验 · 可直接运行的代码包
一页纸速查：[`项目说明.md`](项目说明.md) · 结果分析：[`实验结果报告.html`](实验结果报告.html)
数据集：`data\`（100 张，猫 `cat.1.jpg`~`cat.50.jpg` / 狗 `dog.51.jpg`~`dog.100.jpg`）
权重缓存：`torch_cache\`（`config.py` 自动指向，可随项目搬迁）

---

## 一、怎么跑（先看这一节）

本机已有一个装好 torch 的 conda 环境 `dl`：
Python 3.12.14 + torch 2.13.0+cu130 + torchvision 0.28 + scikit-learn 1.9 + **gradio 6.27**，CUDA 可用（RTX 5070 Laptop）。
（gradio 是为此项目新装的，只新增了包，没动 torch 等已有依赖。）

```bash
cd "D:/CV/cat_dog"
PY="C:/Users/Echo/.conda/envs/dl/python.exe"

$PY check_env.py              # ⓪ 环境 + 数据集 + 权重缓存自检
$PY run_all.py --dry-run      # ① 先看计划，不执行
$PY run_all.py --list         # ② 列出所有任务编号
$PY run_all.py single         # ③ 跑六步单次运行（约 4 分钟）
$PY run_all.py ablation       # ④ 跑多随机种子消融（约 6 分钟）
$PY run_all.py crossval       # ⑤ 跑 5 折交叉验证（约 9 分钟，见第七节）
$PY run_all.py all            # 全部（不含 Gradio 小程序）
$PY run_all.py app            # 单独启动 Gradio 小程序（前台阻塞，Ctrl+C 退出）
```

`run_all.py` 会把每个任务的输出**同时打到屏幕并写入 `outputs/logs/<编号>_<名称>.log`**，
最后打印一张"成功 / 失败 / 耗时"汇总表。某个任务失败不会中断后续任务。

只想跑其中一两个：

```bash
$PY run_all.py single --only 1 4        # 只跑第一步和第四步
$PY run_all.py ablation --only 7a       # 只跑消融主矩阵
```

### 任务清单与预估耗时

| 组 | 编号 | 任务 | 预估 |
|---|---|---|---|
| single | `0` | 环境自检（依赖 / GPU / 数据） | 10 秒 |
| | `1` | 第一步：AlexNet 从零训练 30 轮 | 40 秒 |
| | `1b` | 第一步：指导书原版 txt 索引划分 | 40 秒 |
| | `2` | 第二步：预训练冻结，30 轮 | 70 秒 |
| | `2b` | 第二步进阶：全解冻微调 5 轮 | 20 秒 |
| | `3` | 第三步：数据增强 5 轮 | 15 秒 |
| | `3b` | 第三步对照组：关掉增强 | 15 秒 |
| | `4` | 第四步：三模型横向对比 | 30 秒 |
| | `5a` | 第五步：训练并保存 `best_model.pth` | 20 秒 |
| | `5b` | 第五步：全量 100 张推理 | 20 秒 |
| ablation | `7a` | 消融主矩阵：7 配置 × 5 种子 | 3 分钟 |
| | `7b` | 消融补充：4 配置 × 5 种子 | 2 分钟 |
| | `7c` | 随机性诊断 `_diag_rng.py` | 1 分钟 |
| crossval | `8` | 5 折交叉验证：6 配置 × 3 种子 × 5 折 | 9 分钟 |
| app | `6` | Gradio 拍照小程序 | 阻塞 |

> **权重缓存在项目内**：`torch_cache\hub\checkpoints\`，4 个预训练权重已放好。
> `config.py` 启动时自动把 `TORCH_HOME` 指向它（在 `import torch` 之前设好），
> 所以**第一次跑就能离线完成**，整个 `cat_dog` 文件夹也可以直接拷到别的机器。
> 想临时换到别处：设 `CATDOG_TORCH_HOME` 覆盖。当前生效目录用 `check_env.py` 查看。

> **只用 CPU 跑也行**：所有脚本加 `--device cpu` 即可复现课堂环境，但 AlexNet 30 轮会很慢，
> 建议配合 `--epochs 5`。

### 想单独跑某个脚本

每个脚本都能独立运行，也都支持 `--epochs / --lr / --batch-size / --device / --seed / --data-dir`：

```bash
$PY step1_scratch.py                    # 默认 30 轮，seed=42
$PY step1_scratch.py --split txt        # 指导书原版 txt 索引划分
$PY step3_augment.py --no-augment       # 对照组
$PY step3_augment.py --random-erasing   # 再加 RandomErasing
$PY step2_pretrained.py --finetune      # 解冻全部层微调
$PY step5_predict.py --dir "data"       # 批量推理整个数据集
$PY stepX.py --help                     # 看全部参数
```

### step5_predict.py 的 `--model` 必须与权重匹配

这是最容易踩的坑：**权重是哪个架构训的，`--model` 就得写哪个**，否则 state_dict 键名对不上。

| 权重文件 | `--model` |
|---|---|
| `outputs/best_model.pth` | `resnet18`（默认） |
| `outputs/model_resnet18.pth` | `resnet18` |
| `outputs/model_mobilenet_v3_small.pth` | `mobilenet_v3_small` |
| `outputs/model_shufflenet_v2_x1_0.pth` | `shufflenet_v2_x1_0` |
| `outputs/step3_resnet18_aug.pth` | `resnet18` |
| `outputs/step3_resnet18_noaug.pth` | `resnet18` |
| `outputs/step1_alexnet_scratch.pth` | `alexnet_scratch` ← 手写版，默认划分 |
| `outputs/step1_alexnet_scratch_txt.pth` | `alexnet_scratch` ← 手写版，`--split txt` |
| `outputs/step2_alexnet_frozen.pth` | `alexnet` ← torchvision 版 |
| `outputs/step2_alexnet_finetune.pth` | `alexnet` |

> AlexNet 在指导书里有两套结构（手写版通道 96/256/384/384/256 用 `net.*` 键；
> torchvision 版 64/192/384/256/256 用 `features.*`/`classifier.*` 键），
> 所以分成 `alexnet_scratch` 和 `alexnet` 两个名字。选错会给出明确提示，不会抛一大段报错。
>
> `step1_scratch.py` 按划分方式分文件保存：默认划分 → `step1_alexnet_scratch.pth`，
> `--split txt` → `step1_alexnet_scratch_txt.pth`。两者准确率不同（0.80 vs 0.75），
> 早期版本共用一个文件名，跑完 `1` 再跑 `1b` 会互相覆盖。

---

## 二、跑完之后看什么

| 产物 | 位置 | 内容 |
|---|---|---|
| 权重 | `outputs/*.pth` | `best_model.pth` 是第五步存的最佳模型；`model_<名称>.pth` 是第四步的三模型 |
| 消融数据 | `outputs/ablation_results.csv` / `.json` | 7 配置 × 5 种子的 mean / std / 逐种子结果 |
| 消融补充 | `outputs/ablation_extra.csv` / `.json` | 轮数对等与长轮数对照 |
| 交叉验证 | `outputs/crossval_results.csv` / `.json` | 6 配置 × 3 种子 × 5 折：池化准确率 + 配对比较 |
| 全部日志 | `outputs/logs/` | 每个任务的完整 stdout（用 `run_all.py` 跑才有） |
| txt 索引 | `outputs/trainDatasets.txt` / `testDatasets.txt` | `--split txt` 时生成 |

**自检要点**：消融矩阵里 `seed=42` 那一格，应该**精确等于**该脚本单独跑的结果
（当前对齐情况：`1→0.80、2→0.95、3→0.90、4→0.90、6→0.90、7→0.75`）。
对不上就说明随机性约定被破坏了，见第六节。

---

## 三、目录结构

```
cat_dog/
├── run_all.py           # ★ 总控脚本：一条命令跑完全部实验
├── config.py            # 全局配置：数据集路径 / 权重缓存 / 超参 / 设备
├── check_env.py         # 环境自检（含权重缓存位置与完整性）
├── common.py            # 公共代码 = 指导书 0.2 + 0.3
├── models_manual.py     # 手写 ResNet18（补上指导书悬空引用的 ResNet18Manual）
├── step1_scratch.py     # 第一步：AlexNet 从零训练（两种划分分文件保存）
├── step2_pretrained.py  # 第二步：AlexNet + ImageNet 预训练 + 冻结 / 微调
├── step3_augment.py     # 第三步：ResNet18 + 数据增强（含 RandomErasing）
├── step4_compare.py     # 第四步：torchvision 三模型对比
├── step5_train_save.py  # 第五步上：保存最佳模型
├── step5_predict.py     # 第五步下：单张 / 批量推理（支持 6 种架构）
├── step6_app.py         # 第六步：Gradio 小程序
├── step7_ablation.py    # 加餐①：多随机种子消融（mean ± std + 配对比较）
├── step8_crossval.py    # 加餐②：5 折交叉验证（把验证集从 20 张提到 100 张）
├── _diag_rng.py         # 随机性诊断脚本（见第六节）
├── data/                # 数据集：cat.1.jpg~cat.50.jpg / dog.51.jpg~dog.100.jpg
├── torch_cache/         # 预训练权重缓存（config.py 自动指向，可随项目搬迁）
│   └── hub/checkpoints/ #   ← 4 个 .pth 放这里（注意多一层 hub）
├── 项目说明.md            # 一页纸速查：环境 / 路径 / 任务 / 产物
├── 实验结果报告.html      # 完整结果报告（可打印）
└── outputs/             # 权重、日志、CSV/JSON（自动创建）
```

> `data/` 和 `torch_cache/` 都在项目内，所以整个文件夹**拷到哪都能直接跑**。

---

## 四、数据集路径怎么改

`config.py` 里按优先级自动找，找到第一个存在的就用（**全部绝对路径，和当前工作目录无关**）：

1. 环境变量 `CATDOG_DATA`
2. `data/` ← 当前生效（项目内，100 张）
3. `datasets/`（指导书里的 `./datasets`）

换数据集只需：

```bash
# 临时换
set CATDOG_DATA=D:\your\path && python step1_scratch.py

# 或直接改 config.py 的 DATA_DIR
```

---

## 五、六步在做什么（含本机实测结果）

本机实测环境：RTX 5070 Laptop + torch 2.13.0+cu130，100 张图，80/20 划分。

⚠️ **先看这一条**：验证集只有 20 张，**1 个样本 = 5%**。所以下面"单次运行"那一列仅供参考，
真正可信的是 `step7_ablation.py` 跑出的 **5 种子均值 ± 标准差**。

| 步骤 | 脚本 | 关键改动 | 指导书预期 | 单跑(seed=42) | **5 种子均值 ± std** |
|---|---|---|---|---|---|
| 1 | `step1_scratch.py` | 随机权重，30 轮 | 0.65~0.80 | 0.80 | **0.68 ± 0.10** |
| 2 | `step2_pretrained.py` | 灌 ImageNet 权重、冻结卷积 | 0.90~1.00 | 0.95 | **0.91 ± 0.05** |
| 3 | `step3_augment.py` | ResNet18 + 数据增强，5 轮 | ≈步骤2 或更稳 | 0.90 | **0.88 ± 0.08** |
| 4 | `step4_compare.py` | 三模型对比，各 5 轮 | 都在 0.9+ | 0.90 / 0.90 / 0.75 | **0.88 / 0.89 / 0.74** |
| 5 | `step5_train_save.py` + `step5_predict.py` | 保存 best + 推理 | — | 0.90；全量 **91%** | — |
| 6 | `step6_app.py` | Gradio 摄像头界面 | — | HTTP 200 | — |

> 第一步单跑 0.80，但 5 个种子里只有 1 次摸到 0.80（其余 0.55~0.75）——
> 它恰好落在指导书预期的 0.65~0.80 区间里，是**运气**。
> 真实水平低于预期下限，反而更支持"小数据集从零训练学不出通用特征"这个论断。

第四步实测（batch=16，GPU）：

| 模型 | 参数量 | 单次划分 5 种子 | **5 折池化（100 张）** | 每轮用时 |
|---|---|---|---|---|
| resnet18 | 11.18M | 0.8800 ± 0.0758 | 0.9100 | 2.72s |
| mobilenet_v3_small | 1.52M | 0.8900 ± 0.0548 | 0.9100 | 2.62s |
| shufflenet_v2_x1_0 | 1.26M | 0.7400 ± 0.0822 | **0.9033** | 2.66s |

> ⚠️ **注意 ShuffleNet 那一行**：单次划分看着差 0.14，池化 100 张后只差 0.007（1 张图）。
> 那 0.14 是 20 张验证集抽样的运气，**不是模型真的差**。详见第七节。

> resnet18 与 mobilenet 差 0.01，**在噪声内，不能说谁更强**；
> 但 mobilenet 只用 **1/7** 的参数达到同样水平，工程上是明确的胜利。
> ShuffleNet 落后 0.14（约 3 个样本）且训练 loss 高一个数量级，**这个差距是真的**。

### 配对比较（同一划分下逐种子对比）

| 对比 | 差值 mean ± std | 前者更好的种子 | 判定 |
|---|---|---|---|
| **迁移学习**（预训练 − 从零） | **+0.2300 ± 0.1151** | **5 / 5** | ✅ 显著有效 |
| 数据增强（5 轮） | −0.0200 ± 0.0447 | 0 / 5 | ❌ 无差异 |
| 数据增强（15 轮） | −0.0100 ± 0.0548 | 1 / 5 | ❌ 无差异 |
| RandomErasing | +0.0100 ± 0.0224 | 1 / 5 | ❌ 无差异 |
| 冻结 vs 微调（同为 5 轮） | +0.0500 ± 0.0612 | 3 / 5 | ❌ 无差异 |

**五组对照里只有"迁移学习"是真的。** 指导书"六步对比表"里那些 0.85/0.90/0.95 的差异，
绝大多数是 20 张验证集的噪声。数据增强在这份数据上甚至**略微拖后腿**
（训练 loss 0.267 → 0.317，任务变难但信息量没增加）。

📄 **完整报告见 [`实验结果报告.html`](实验结果报告.html)**（含逐条解读、可复现性坑位、思考题解答）。
原始日志在 `outputs/logs/`，结构化数据在 `outputs/ablation_results.csv`。

> **控制变量**：所有步骤共用 `common.get_loaders` 的同一份 80/20 划分（seed=42），
> 每次只改一个变量，对比才公平。
> （指导书里第一步用的是 `random.sample` 的 txt 划分，和后面几步的 `random_split`
> 其实是两份不同的数据 —— 这是原文一个隐藏的不公平点。想看原版写法加 `--split txt`。）

---

## 六、随机性约定（改代码前务必读）

`step1`~`step8` 全部遵守同一条顺序，**同一个 seed 才能跑出同一个结果**：

```
set_seed(seed)  ->  构建模型  ->  构建 DataLoader  ->  训练（每轮验证一次）
```

四条容易踩的规则：

1. **`train_and_eval()` 不再自己播种。** 调用方负责在构建模型前 `set_seed(seed)`。
   以前函数内部会再播一次种，导致模型初始化用的是一个随机状态、训练又换一个，无法复现。
2. **构建模型的顺序要在 DataLoader 之前。** `nn.init.xavier_uniform_` 和 DataLoader 的
   shuffle 都消耗全局随机数，谁先谁后会让"同一个 seed"跑出不同结果。
3. **每轮必须验证一次。** DataLoader 的每个迭代器都会用全局随机数生成 `base_seed`，
   少迭代一次（比如训练循环里不跑验证），从第 2 轮起 shuffle 顺序就会分叉 ——
   实测同一个 seed 会跑出 0.80 vs 0.75。
   验证方法见 `_diag_rng.py`。
4. **同一脚本的不同配置不能写同一个输出文件名。** `step1_scratch.py` 曾经两种划分都存
   `step1_alexnet_scratch.pth`，`run_all.py single` 跑完任务 `1` 再跑 `1b`，
   后者的模型就**静默覆盖**了前者（0.75 盖掉 0.80），之后所有推理都在用错的模型。
   现已按划分分文件：默认 → `step1_alexnet_scratch.pth`，`--split txt` → `step1_alexnet_scratch_txt.pth`。
   `step2`/`step3` 早已用 `step2_alexnet_{frozen|finetune}.pth` 这样区分。

消融矩阵 `step7_ablation.py` 的配置表里，`epochs` 和 `lr` **必须显式写出并与对应脚本一致**。
第一版就是漏写了 lr，让"从零训练"走了 1e-3（正确值是 1e-4，差 10 倍），结果算成 0.52。

**自检方法**：消融矩阵里 `seed=42` 的那一格，应该精确等于该脚本单独跑的结果。
当前状态：`1→0.80、2→0.95、3→0.90、4→0.90、6→0.90、7→0.75`，全部对齐 ✅
（本机重跑 5 个种子后逐种子数值与文档完全一致，验证了整套协议。）

---

## 七、5 折交叉验证（**已跑完**）

### 为什么还要它

单次 80/20 划分下验证集只有 20 张，**1 个样本 = 5%**，所以消融里五组对照有四组落在噪声内。
K 折交叉验证让**每张图恰好被预测一次**：

```
5 折 × 每折 20 张 = 100 个预测  ->  分辨率 1%（比单次划分细 5 倍）
```

做法：按类别**分层**切 5 折（保证每折猫狗比例一致），轮流拿 1 折当验证集、
其余 4 折训练，最后把所有验证预测池在一起算准确率。这是本实验能给出的最细的估计。

### 怎么跑

```bash
$PY run_all.py crossval          # 6 配置 × 3 种子 × 5 折 = 90 次训练，约 9 分钟
# 或单独跑：
$PY step8_crossval.py
$PY step8_crossval.py --seeds 42 1 --configs 3_resnet18_noaug 4_resnet18_aug   # 加快
$PY step8_crossval.py --folds 10                                                # 更细
```

输出到 `outputs/crossval_results.csv` 和 `.json`。

### 跑完怎么读

脚本会给出两组数字，含义不同，别混：

| 列 | 含义 | 用途 |
|---|---|---|
| **池化准确率**（`pooled_*`） | 100 个预测一起算，分辨率 1% | **这是要写进报告的主数字** |
| 各折准确率（`fold_*`） | 5 折各自 20 张的准确率 | 看折间波动，本身仍是 5% 分辨率 |

配对比较同样是同种子逐对，看"前者更好的种子数"。
判读标准：池化后 **1 个样本 = 1%**，所以差值小于 0.03（3 个样本）仍不宜下结论。

### 本机实测结果

| 配置 | 池化准确率 | ± std | min–max | 逐种子 | 秒/折 |
|---|---|---|---|---|---|
| 从零训练 AlexNet | 0.6533 | 0.0473 | 0.60–0.69 | .69/.67/.60 | 54.5 |
| **AlexNet 预训练冻结** | **0.9233** | 0.0252 | 0.90–0.95 | .92/.95/.90 | 10.9 |
| ResNet18 无增强 | 0.9133 | 0.0115 | 0.90–0.92 | .92/.90/.92 | 10.0 |
| ResNet18 + 增强 | 0.9100 | 0.0100 | 0.90–0.92 | .92/.91/.90 | 14.3 |
| MobileNetV3-S + 增强 | 0.9100 | 0.0100 | 0.90–0.92 | .92/.90/.91 | 13.9 |
| ShuffleNetV2 + 增强 | 0.9033 | 0.0058 | 0.90–0.91 | .90/.91/.90 | 14.2 |

配对比较（3 个种子）：

| 对照 | Δ 池化准确率 | wins | 判定 |
|---|---|---|---|
| **迁移学习（预训练 − 从零）** | **+0.2700 ± 0.0361** | **3/3** | ✅ **显著** |
| 数据增强（aug − noaug） | −0.0033 ± 0.0153 | 1/3 | 噪声内 |
| MobileNet vs ResNet18 | 0.0000 ± 0.0100 | 1/3 | 完全一致 |
| ShuffleNet vs ResNet18 | −0.0067 ± 0.0115 | 0/3 | 略低，幅度极小 |

### ⚠️ 交叉验证推翻了单次划分的一个结论

单次划分（20 张）下 **ShuffleNet 看着明显更差**：0.74 vs ResNet18 的 0.88，差 0.14。
池化 100 张后实际是 **0.9033 vs 0.9133，只差 1 张图**。

| 配置 | 单次划分（20 张） | 5 折池化（100 张） | 差异 |
|---|---|---|---|
| ResNet18 + 增强 | 0.88 ± 0.08 | 0.9100 | +0.03 |
| ShuffleNetV2 + 增强 | 0.74 ± 0.08 | 0.9033 | **+0.16** |
| *两者差距* | *−0.14* | *−0.0067* | *缩小 20 倍* |

原因就是本节开头说的粒度陷阱——ShuffleNet 恰好在那 20 张上错了 5 张，就被判成"差 0.14"。

**结论**：四个预训练模型全部落在 **0.90–0.92**（最大差距 2 张图），
**模型选型在这份数据上没有意义**；只有**迁移学习**是唯一站得住的结论。
详见 `实验结果报告.html` 第六节。

---

## 八、常见问题

**Q：下载预训练权重卡住 / 失败？**

本项目**不需要下载** —— 4 个权重已在 `torch_cache\hub\checkpoints\`，
`config.py` 会自动把 `TORCH_HOME` 指过去。跑 `python check_env.py` 可确认。

万一换机器丢了，**手动下载 `.pth` 放进 `torch_cache\hub\checkpoints\`** 是最稳的办法：

| 文件 | 来源（`torchvision` 官方） |
|---|---|
| `alexnet-owt-7be5be79.pth` | `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth` |
| `resnet18-f37072fd.pth` | `https://download.pytorch.org/models/resnet18-f37072fd.pth` |
| `mobilenet_v3_small-047dcff4.pth` | `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth` |
| `shufflenetv2_x1-5666bf0f80.pth` | `https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth` |

文件名必须**一模一样**（含哈希后缀），torch 按文件名查缓存。

**Q：想把权重缓存换到别处？**

两种方式，**注意路径差一层 `hub`**：

| 方式 | 怎么设 | checkpoints 落在 |
|---|---|---|
| 环境变量 `CATDOG_TORCH_HOME`（本项目推荐） | `set CATDOG_TORCH_HOME=D:\别的盘\torch_cache` | `D:\别的盘\torch_cache\hub\checkpoints\` |
| 环境变量 `TORCH_HOME`（通用，但会被上面覆盖） | `setx TORCH_HOME "D:\torch_cache"` | `$TORCH_HOME\hub\checkpoints\` |

⚠️ `TORCH_HOME` 是 `hub` 的**父目录**，不是 hub 目录本身 —— 多一层或少一层都会让
torch 以为没缓存、重新下载。若在代码里写，`torch.hub.set_dir(d)` 则是 `d\checkpoints`（**不带 hub**）。

优先级（实测）：`CATDOG_TORCH_HOME` > 项目内 `torch_cache` > `TORCH_HOME` > `XDG_CACHE_HOME` > `~/.cache`。

跑 `python check_env.py` 会打印**实际生效的缓存目录**和四个权重是否齐全，改完拿它验证。

**Q：显存 / 内存不够？**
把 `--batch-size` 调小（8 或 4）。

**Q：第六步手机连不上？**
1. 手机和电脑必须**同一个 Wi-Fi**；
2. Windows 防火墙要放行 Python（首次启动会弹窗，勾"专用网络"）；
3. 脚本启动时会打印真实局域网 IP，用它而不是 `127.0.0.1`。

**Q：想快速验收，不想等？**
```bash
python step4_compare.py --epochs 2
python step5_train_save.py --epochs 2
```
