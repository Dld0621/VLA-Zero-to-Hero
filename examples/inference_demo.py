#!/usr/bin/env python3
"""
inference_demo.py
=================
使用 OpenVLA 进行真实推理的完整示例。

需要：
  - GPU with >= 16GB VRAM (for 7B model)
  - pip install transformers accelerate torch

如果显存不足，可改用 Octo（27M-93M 参数）或加载 4-bit 量化的 OpenVLA。
"""

import torch
from PIL import Image
import argparse


def load_openvla(device: str = "cuda"):
    """加载 OpenVLA 模型和处理器。"""
    print("Loading OpenVLA-7B... (this may take a few minutes)")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, processor


def run_inference(
    model,
    processor,
    image_path: str,
    instruction: str,
    unnorm_key: str = "bridge",
    device: str = "cuda",
):
    """
    运行单步 VLA 推理。

    Args:
        model: OpenVLA 模型
        processor: 对应的处理器
        image_path: 输入图像路径
        instruction: 自然语言指令
        unnorm_key: 反归一化键（对应训练数据集）
        device: 计算设备

    Returns:
        action: numpy 数组，动作向量
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 构建 prompt（OpenVLA 的特定格式）
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # 预处理
    inputs = processor(prompt, image).to(device)

    # 推理
    with torch.no_grad():
        action = model.predict_action(inputs, unnorm_key=unnorm_key)

    return action


def print_action(action, action_names=None):
    """美化打印动作向量。"""
    if action_names is None:
        # OpenVLA 默认输出 7 维 delta 位姿
        action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    print("\nPredicted Action:")
    print("-" * 30)
    for name, val in zip(action_names, action):
        print(f"  {name:12s}: {val:+.6f}")
    print("-" * 30)

    # 动作语义解释
    print("\nInterpretation:")
    pos = action[:3]
    rot = action[3:6]
    gripper = action[6]

    pos_norm = sum(x**2 for x in pos) ** 0.5
    print(f"  Position delta magnitude: {pos_norm:.4f} m")
    print(f"  Rotation delta magnitude: {sum(x**2 for x in rot)**0.5:.4f} rad")
    print(f"  Gripper: {'OPEN' if gripper > 0 else 'CLOSE'} ({gripper:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--instruction", type=str, required=True, help="语言指令")
    parser.add_argument("--unnorm_key", type=str, default="bridge", help="反归一化键")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenVLA Inference Demo")
    print("=" * 60)

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU (very slow)")
        args.device = "cpu"

    # 加载模型
    try:
        model, processor = load_openvla(device=args.device)
    except Exception as e:
        print(f"\nERROR: Failed to load OpenVLA: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have installed transformers>=4.40.0")
        print("  2. Run: pip install transformers accelerate")
        print("  3. Ensure you have >= 16GB GPU memory")
        print("  4. For CPU/limited GPU, use examples/minimal_vla.py instead")
        return

    # 运行推理
    print(f"\nInput:")
    print(f"  Image: {args.image}")
    print(f"  Instruction: '{args.instruction}'")
    print(f"  Unnorm key: {args.unnorm_key}")

    action = run_inference(
        model,
        processor,
        image_path=args.image,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        device=args.device,
    )

    # 打印结果
    print_action(action)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
