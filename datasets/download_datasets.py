#!/usr/bin/env python3
"""
下载灵巧操作开源数据集

运行方式:
    cd datasets/
    python download_datasets.py

支持下载的数据集:
    1. DexGraspNet  - Shadow Hand 仿真抓取 (CC-BY-NC 4.0)
    2. robomimic     - Shadow Hand 仿真操作 (MIT)
    3. DEXCap        - LEAP Hand 真实双手 (MIT, HuggingFace)
    4. LIBERO        - Franka 操作基准 (CC-BY 4.0, HuggingFace)
    5. BridgeData V2 - WidowX 操作 (MIT)
    6. InterHand2.6M - 人手 3D 姿态 (CC-BY-NC 4.0, 需申请)

需要额外工具:
    - HuggingFace: pip install huggingface_hub
    - Open X-Embodiment: pip install gsutil  (需 Google Cloud 认证)
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATASETS = {
    "DexGraspNet": {
        "url": "https://github.com/PKU-EPIC/DexGraspNet.git",
        "type": "git",
        "desc": "132万+ Shadow Hand 仿真抓取姿态 (NeurIPS 2023)",
        "license": "CC-BY-NC 4.0",
        "post_download": "cd DexGraspNet && python scripts/download_dataset.py",
    },
    "robomimic": {
        "url": "https://github.com/ARISE-Initiative/robomimic.git",
        "type": "git",
        "desc": "Shadow Hand 仿真操作数据 (SHAPES, lift 等任务, CoRL 2021)",
        "license": "MIT",
        "post_download": "cd robomimic && python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph",
    },
    "LIBERO": {
        "type": "huggingface",
        "repo_id": "yifengzhu-hf/LIBERO-datasets",
        "desc": "130 个操作任务的遥操作演示 (NeurIPS 2023)",
        "license": "CC-BY 4.0",
    },
    "BridgeData_V2": {
        "url": "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/",
        "type": "manual",
        "desc": "WidowX 二指操作 (CoRL 2023)",
        "license": "MIT",
        "note": "需手动从浏览器下载 demos*.zip 和 scripted*.zip",
    },
    "InterHand2.6M": {
        "url": "https://mks0601.github.io/InterHand2.6M/",
        "type": "manual",
        "desc": "260万帧人手 3D 关键点标注 (ECCV 2020)",
        "license": "CC-BY-NC 4.0",
        "note": "需注册申请后从项目页下载",
    },
}


def run_cmd(cmd: str, cwd=None):
    """运行 shell 命令"""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  错误: {result.stderr[:200]}")
        return False
    return True


def git_clone(name: str, url: str):
    """Git clone 数据集仓库"""
    target = BASE_DIR / name
    if target.exists():
        print(f"[{name}] 已存在，跳过")
        return True
    print(f"\n[{name}] Git clone...")
    return run_cmd(f"git clone --depth 1 {url}", cwd=BASE_DIR)


def huggingface_download(name: str, repo_id: str):
    """从 HuggingFace 下载数据集"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f"[{name}] 需要 huggingface_hub: pip install huggingface_hub")
        return False

    target = BASE_DIR / name
    if target.exists() and any(target.iterdir()):
        print(f"[{name}] 已存在，跳过")
        return True

    print(f"\n[{name}] HuggingFace 下载: {repo_id}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target))
        print(f"  完成")
        return True
    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    print("=" * 60)
    print("灵巧操作数据集下载脚本")
    print("=" * 60)
    print(f"下载目录: {BASE_DIR}\n")

    success = []
    failed = []
    manual = []

    for name, info in DATASETS.items():
        dtype = info["type"]
        if dtype == "git":
            ok = git_clone(name, info["url"])
        elif dtype == "huggingface":
            ok = huggingface_download(name, info["repo_id"])
        elif dtype == "manual":
            manual.append((name, info))
            continue
        else:
            continue

        if ok:
            success.append(name)
            # 执行 post_download 命令
            if "post_download" in info:
                print(f"\n[{name}] 执行数据下载脚本...")
                run_cmd(info["post_download"], cwd=BASE_DIR / name)
        else:
            failed.append(name)

    # 打印结果
    print("\n" + "=" * 60)
    print(f"成功: {len(success)}/{len(DATASETS)}")
    for n in success:
        print(f"  [OK] {n}")
    if failed:
        print(f"失败: {len(failed)}")
        for n in failed:
            print(f"  [FAIL] {n}")

    if manual:
        print("\n" + "=" * 60)
        print("需手动下载的数据集")
        print("=" * 60)
        for name, info in manual:
            print(f"\n{name} ({info['desc']})")
            print(f"  许可证: {info['license']}")
            print(f"  下载: {info['url']}")
            if "note" in info:
                print(f"  注意: {info['note']}")
            print(f"  放置: datasets/{name}/")


if __name__ == "__main__":
    main()