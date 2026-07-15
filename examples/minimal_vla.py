#!/usr/bin/env python3
"""
minimal_vla.py
==============
一个最简化的 VLA 推理示例。
不依赖预训练大模型，仅用 CLIP + 小型 MLP 演示 VLA 的 pipeline。

适合理解 VLA 的基本架构，不适合实际机器人控制。
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse


class MinimalVLA(nn.Module):
    """
    最小 VLA 模型：
    - 视觉编码器：随机初始化的 CNN（实际应用中使用 CLIP/DINOv2）
    - 文本编码器：随机初始化的 Embedding + GRU（实际应用中使用 LLaMA/T5）
    - 融合层：简单拼接 + MLP
    - 策略头：MLP 回归输出动作
    """

    def __init__(
        self,
        image_size: int = 224,
        action_dim: int = 7,
        vocab_size: int = 1000,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim

        # --- 视觉编码器（简化版 CNN）---
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

        # --- 文本编码器（简化版）---
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_encoder = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # --- 融合层 ---
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # --- 策略头：输出动作 ---
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """将文本 token 编码为特征向量。"""
        emb = self.text_embedding(text_tokens)
        _, hidden = self.text_encoder(emb)
        return hidden.squeeze(0)  # [B, embed_dim]

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            image: [B, 3, H, W] RGB 图像
            text_tokens: [B, seq_len] 文本 token IDs

        Returns:
            action: [B, action_dim] 归一化动作（范围 [-1, 1]）
        """
        # 编码视觉和文本
        vis_feat = self.vision_encoder(image)        # [B, embed_dim]
        txt_feat = self.encode_text(text_tokens)      # [B, embed_dim]

        # 融合
        fused = torch.cat([vis_feat, txt_feat], dim=-1)  # [B, embed_dim*2]
        fused = self.fusion(fused)                       # [B, embed_dim]

        # 生成动作
        action = self.policy_head(fused)  # [B, action_dim]
        return action


def simple_tokenize(text: str, max_len: int = 16, vocab_size: int = 1000) -> torch.Tensor:
    """
    极度简化的 tokenizer：将每个字符映射为一个整数 ID。
    实际应用中使用 HuggingFace Tokenizer。
    """
    tokens = [ord(c) % vocab_size for c in text.lower()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))  # pad
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def preprocess_image(image_path: str, size: int = 224) -> torch.Tensor:
    """加载并预处理图像。"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    # 简单归一化到 [-1, 1]
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def denormalize_action(action: np.ndarray) -> dict:
    """
    将 [-1, 1] 的归一化动作映射到实际物理量。
    这只是示例值，实际机器人需根据硬件调整。
    """
    # 假设 action_dim=7: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    return {
        "delta_position": action[:3] * 0.05,      # 最大 5cm 移动
        "delta_rotation": action[3:6] * 0.1,      # 最大 0.1 rad 旋转
        "gripper": (action[6] + 1) / 2,           # 映射到 [0, 1]
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal VLA Demo")
    parser.add_argument("--image_path", type=str, default=None, help="输入图像路径")
    parser.add_argument("--instruction", type=str, default="pick up the object", help="语言指令")
    parser.add_argument("--action_dim", type=int, default=7, help="动作维度")
    args = parser.parse_args()

    print("=" * 60)
    print("Minimal VLA Demo")
    print("=" * 60)

    # 初始化模型
    model = MinimalVLA(action_dim=args.action_dim)
    model.eval()

    # 准备输入
    if args.image_path:
        image = preprocess_image(args.image_path)
        print(f"\n[Input] Image: {args.image_path}")
    else:
        # 使用随机图像作为占位
        image = torch.randn(1, 3, 224, 224)
        print("\n[Input] Random placeholder image (use --image_path for real image)")

    text_tokens = simple_tokenize(args.instruction)
    print(f"[Input] Instruction: '{args.instruction}'")
    print(f"[Input] Text tokens: {text_tokens.squeeze().tolist()[:10]}...")

    # 推理
    with torch.no_grad():
        action = model(image, text_tokens)

    action_np = action.squeeze().numpy()
    print(f"\n[Output] Raw action (normalized, [-1, 1]): {action_np}")

    # 反归一化到物理量
    phys = denormalize_action(action_np)
    print(f"\n[Output] Physical action:")
    print(f"  Delta position (m): {phys['delta_position']}")
    print(f"  Delta rotation (rad): {phys['delta_rotation']}")
    print(f"  Gripper (0=close, 1=open): {phys['gripper']:.3f}")

    print("\n" + "=" * 60)
    print("Note: This is a RANDOMLY INITIALIZED model for architecture demo only.")
    print("For real VLA inference, use OpenVLA or Octo (see docs/02-key-papers.md)")
    print("=" * 60)


if __name__ == "__main__":
    main()
