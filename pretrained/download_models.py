#!/usr/bin/env python3
"""
下载 AnyTeleop + HaMeR 复现所需的全部模型文件

运行方式:
    cd pretrained/
    python download_models.py

需要手动注册下载的文件（见下方注释）：
    - MANO_RIGHT.pkl: https://mano.is.tue.mpg.de/
    - SMPLX_NEUTRAL.pkl: https://smpl-x.is.tue.mpg.de/
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).parent

FILES = {
    # HaMeR 主模型权重包
    "hamer/hamer_demo_data.tar.gz": {
        "url": "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz",
        "size_mb": 1700,  # 估算
        "desc": "HaMeR 主模型权重包 (hamer.ckpt + model_config.yaml + mano_mean_params.npz + ViTPose-H 权重)",
    },
    # ViTDet 人体检测权重
    "hamer/model_final_f05665.pkl": {
        "url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
        "size_mb": 2600,
        "desc": "Detectron2 ViTDet-H 人体检测器权重",
    },
    # FrankMocap 模型文件
    "anyteleop/frankmocap/extra_data/hand_module/SMPLX_HAND_INFO.pkl": {
        "url": "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/SMPLX_HAND_INFO.pkl",
        "size_mb": 0.1,
        "desc": "SMPL-X 手部元数据",
    },
    "anyteleop/frankmocap/extra_data/hand_module/mean_mano_params.pkl": {
        "url": "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/mean_mano_params.pkl",
        "size_mb": 0.001,
        "desc": "MANO 均值参数",
    },
    "anyteleop/frankmocap/extra_data/hand_module/pretrained_weights/pose_shape_best.pth": {
        "url": "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/checkpoints_best/pose_shape_best.pth",
        "size_mb": 102,
        "desc": "FrankMocap 手部姿态回归权重",
    },
}


def download_file(rel_path: str, url: str, desc: str):
    """带进度条和断点续传的文件下载"""
    full_path = BASE_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.exists():
        current_size = full_path.stat().st_size
        print(f"[{rel_path}] 已存在 ({current_size / 1024 / 1024:.1f} MB)，尝试断点续传...")
        headers = {"Range": f"bytes={current_size}-"}
    else:
        current_size = 0
        headers = {}
        print(f"\n[{rel_path}] 开始下载: {desc}")

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            if response.status == 416:  # Range Not Satisfiable = 已完整
                print(f"  文件已完整，跳过")
                return True

            mode = "ab" if current_size > 0 else "wb"
            downloaded = current_size

            with open(full_path, mode) as f:
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / (current_size + total_size) * 100
                        print(f"  {downloaded / 1024 / 1024:.1f} MB / {(current_size + total_size) / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="\r")
                    else:
                        print(f"  {downloaded / 1024 / 1024:.1f} MB", end="\r")

            print()  # newline
            print(f"  完成: {full_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True

    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    print("=" * 60)
    print("AnyTeleop + HaMeR 模型文件下载脚本")
    print("=" * 60)

    success = []
    failed = []

    for rel_path, info in FILES.items():
        ok = download_file(rel_path, info["url"], info["desc"])
        if ok:
            success.append(rel_path)
        else:
            failed.append(rel_path)

    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    print(f"成功: {len(success)}/{len(FILES)}")
    for f in success:
        print(f"  [OK] {f}")
    if failed:
        print(f"失败: {len(failed)}")
        for f in failed:
            print(f"  [FAIL] {f}")

    print("\n" + "=" * 60)
    print("需手动注册下载的文件（无法通过脚本自动获取）")
    print("=" * 60)
    print("""
1. MANO_RIGHT.pkl
   地址: https://mano.is.tue.mpg.de/
   步骤: 注册账号 → 同意许可证 → 下载 MANO v1.2 → 解压得到 MANO_RIGHT.pkl
   放置: pretrained/mano/MANO_RIGHT.pkl
   许可证: 非商业科研用途

2. SMPLX_NEUTRAL.pkl (仅纯 RGB 模式需要)
   地址: https://smpl-x.is.tue.mpg.de/
   步骤: 注册账号 → 同意许可证 → 下载 SMPL-X 模型
   放置: pretrained/anyteleop/frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl
   许可证: 非商业科研用途
""")

    if failed:
        print("部分文件下载失败，建议：")
        print("  1. 检查网络连接（需能访问 dl.fbaipublicfiles.com 和 utexas.edu）")
        print("  2. 重新运行本脚本（支持断点续传）")
        print("  3. 或使用代理/VPN 下载")
        sys.exit(1)
    else:
        print("全部文件下载完成！")
        sys.exit(0)


if __name__ == "__main__":
    main()
