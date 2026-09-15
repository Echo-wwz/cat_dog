# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完全部实验（总控脚本）

用法：
    python run_all.py --dry-run     # 只打印计划，不执行（先看这个）
    python run_all.py all           # 全部：单跑 + 消融 + 交叉验证（约 15~20 分钟）
    python run_all.py single        # 只跑六步单次运行（约 4 分钟）
    python run_all.py ablation      # 只跑多随机种子消融（约 6 分钟）
    python run_all.py crossval      # 只跑 5 折交叉验证（约 9 分钟）
    python run_all.py app           # 启动 Gradio 小程序（前台阻塞，Ctrl+C 退出）

    python run_all.py ablation --only 7a        # 只跑某个任务
    python run_all.py --list                    # 列出所有任务编号

输出：
    每个任务的完整 stdout 会同时打到屏幕并保存到 outputs/logs/<编号>_<名称>.log
    最后打印一张"成功 / 失败 / 耗时"汇总表。

依赖：环境见 README 第一节（本机用 C:/Users/Echo/.conda/envs/dl/python.exe）
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "outputs", "logs")

# ---------------------------------------------------------------- 任务清单
# 格式： (编号, 说明, 命令参数, 预估耗时)
GROUPS = {
    "single": [
        ("0",  "环境自检（依赖 / GPU / 数据）", ["check_env.py"], "10 秒"),
        ("1",  "第一步：AlexNet 从零训练 30 轮", ["step1_scratch.py"], "40 秒"),
        ("1b", "第一步：指导书原版 txt 索引划分", ["step1_scratch.py", "--split", "txt"], "40 秒"),
        ("2",  "第二步：预训练冻结，30 轮", ["step2_pretrained.py"], "70 秒"),
        ("2b", "第二步进阶：全解冻微调 5 轮", ["step2_pretrained.py", "--finetune", "--epochs", "5"], "20 秒"),
        ("3",  "第三步：数据增强 5 轮", ["step3_augment.py"], "15 秒"),
        ("3b", "第三步对照组：关掉增强", ["step3_augment.py", "--no-augment"], "15 秒"),
        ("4",  "第四步：三模型横向对比", ["step4_compare.py"], "30 秒"),
        ("5a", "第五步：训练并保存 best_model.pth", ["step5_train_save.py"], "20 秒"),
        ("5b", "第五步：全量 100 张推理", ["step5_predict.py", "--dir", "DATASET"], "20 秒"),
    ],
    "ablation": [
        ("7a", "消融主矩阵：7 配置 × 5 种子",
         ["step7_ablation.py", "--out", "outputs/ablation_results.csv"], "3 分钟"),
        ("7b", "消融补充：4 配置 × 5 种子（轮数对等 / 长轮数对照）",
         ["step7_ablation.py",
          "--configs", "2_alexnet_frozen", "8_alexnet_finetune",
          "9_resnet18_noaug_15", "10_resnet18_aug_15",
          "--out", "outputs/ablation_extra.csv"], "2 分钟"),
        ("7c", "随机性诊断：验证「每轮验证会改变 shuffle 顺序」",
         ["_diag_rng.py"], "1 分钟"),
    ],
    "crossval": [
        ("8", "5 折交叉验证：6 配置 × 3 种子 × 5 折 = 90 次训练",
         ["step8_crossval.py"], "9 分钟"),
    ],
    "app": [
        ("6", "第六步：Gradio 拍照小程序（前台阻塞，Ctrl+C 退出）",
         ["step6_app.py"], "—"),
    ],
}

ORDER = ["single", "ablation", "crossval", "app"]


def resolve(args):
    """把 DATASET 占位符换成真实数据集路径"""
    from config import DATA_DIR
    return [DATA_DIR if a.upper() == "DATASET" else a for a in args]


def all_tasks():
    tasks = []
    for g in ORDER:
        tasks.extend((g,) + t for t in GROUPS[g])
    return tasks


def run_one(num, desc, cmd_args, python_exe):
    """执行一个任务，输出同时打到屏幕和日志文件"""
    os.makedirs(LOG_DIR, exist_ok=True)
    safe = desc.split("：")[0].replace(" ", "_").replace("/", "-")
    log_path = os.path.join(LOG_DIR, f"{num}_{safe}.log")

    print(f"\n{'=' * 78}")
    print(f"[{num}] {desc}")
    print(f"      $ {os.path.basename(python_exe)} {' '.join(cmd_args)}")
    print(f"      日志 -> {os.path.relpath(log_path, HERE)}")
    print("=" * 78, flush=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            [python_exe] + cmd_args, cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode, time.time() - t0, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", nargs="?", default=None,
                        choices=ORDER + ["all", None],
                        help="要跑的组；all = 全部（不含 app）")
    parser.add_argument("--only", nargs="+", default=None,
                        help="只跑指定编号，如 --only 7a 7b")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    parser.add_argument("--list", action="store_true", help="列出全部任务编号")
    parser.add_argument("--python", default=sys.executable,
                        help="指定解释器（默认当前解释器）")
    parser.add_argument("--keep-going", action="store_true", default=True,
                        help="某个任务失败后继续跑后面的（默认开启）")
    args = parser.parse_args()

    if args.list:
        for g in ORDER:
            print(f"\n[{g}]")
            for num, desc, _, est in GROUPS[g]:
                print(f"  {num:3s}  {desc}   ({est})")
        return

    if args.group == "all":
        tasks = [(g,) + t for g in ("single", "ablation", "crossval") for t in GROUPS[g]]
    elif args.group:
        tasks = [(args.group,) + t for t in GROUPS[args.group]]
    else:
        tasks = all_tasks()

    if args.only:
        tasks = [t for t in tasks if t[1] in args.only]
        if not tasks:
            raise SystemExit(f"--only 没匹配到任何任务，可用编号见 python run_all.py --list")

    print("=" * 78)
    print("猫狗分类实验 · 总控脚本")
    print(f"工作目录 : {HERE}")
    print(f"解释器   : {args.python}")
    print(f"任务数   : {len(tasks)}")
    print("=" * 78)
    for g, num, desc, _, est in tasks:
        print(f"  [{g:9s}] {num:3s} {desc}   ({est})")
    print("=" * 78)
    if args.dry_run:
        print("\n（--dry-run：只打印计划，未执行）")
        return

    if args.group == "app" or (args.only and "6" in args.only):
        print("\n提示：第六步会占用终端，Ctrl+C 退出。")

    results = []
    for g, num, desc, cmd_args, _ in tasks:
        rc, secs, log_path = run_one(num, desc, resolve(cmd_args), args.python)
        results.append((num, desc, rc, secs, log_path))
        if rc != 0:
            print(f"\n!! [{num}] 失败（退出码 {rc}），日志见 {log_path}")
            if not args.keep_going:
                break

    print(f"\n{'=' * 78}")
    print("汇总")
    print("-" * 78)
    print(f"{'编号':5s}{'状态':8s}{'耗时':>9s}  说明")
    for num, desc, rc, secs, _ in results:
        status = "成功" if rc == 0 else f"失败({rc})"
        print(f"{num:5s}{status:8s}{secs:8.1f}s  {desc}")
    ok = sum(1 for r in results if r[2] == 0)
    print("-" * 78)
    print(f"共 {len(results)} 个任务，成功 {ok} 个，失败 {len(results) - ok} 个")
    print(f"日志目录: {LOG_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
