#!/usr/bin/env python3
"""
dreamer_rssm.py
================
Dreamer V3 核心架构 RSSM（Recurrent State-Space Model）的简化实现。

RSSM 是世界模型最重要的架构之一，核心思想是 **分离确定性和随机性**：
  - 确定性部分（GRU）：记忆历史，捕捉可预测规律
  - 随机性部分（Gaussian latent）：捕捉不可预测的不确定性

本文档用合成数据演示 RSSM 的训练和推理，帮助理解 Dreamer V3 的核心。

对应理论文档：docs/07-world-models-for-vla.md 第 5.1 节
对应论文：Mastering Diverse Domains through World Models (Hafner et al., 2023)

依赖：pip install torch matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 合成数据：带噪声的 2D 轨迹
# ============================================================

class NoisyTrajectoryDataset(Dataset):
    """
    合成数据集：模拟在 2D 平面上移动的 Agent。
    确定性部分：恒定速度 + 动作控制
    随机性部分：高斯噪声（模拟摩擦、碰撞等不确定因素）
    """

    def __init__(self, num_samples=3000, seq_len=20, dt=0.1, noise_std=0.05):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt

        self.observations = []  # [N, T, obs_dim]
        self.actions = []       # [N, T, act_dim]

        for _ in range(num_samples):
            obs_seq = []
            act_seq = []

            # 随机初始状态
            x, y = np.random.randn(2) * 2.0
            vx, vy = np.random.randn(2) * 0.3

            for t in range(seq_len):
                # 观测 = 位置 + 速度
                obs = np.array([x, y, vx, vy], dtype=np.float32)
                obs_seq.append(obs)

                # 动作 = 速度调整
                action = np.random.randn(2).astype(np.float32) * 0.2
                act_seq.append(action)

                # 确定性转移
                vx = vx + action[0] * dt
                vy = vy + action[1] * dt
                x = x + vx * dt
                y = y + vy * dt

                # 随机性（模拟不确定的环境因素）
                x += np.random.randn() * noise_std
                y += np.random.randn() * noise_std

            self.observations.append(np.array(obs_seq, dtype=np.float32))
            self.actions.append(np.array(act_seq, dtype=np.float32))

        self.observations = np.array(self.observations)  # [N, T, 4]
        self.actions = np.array(self.actions)             # [N, T, 2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.observations[idx]),  # [T, 4]
            torch.FloatTensor(self.actions[idx]),        # [T, 2]
        )


# ============================================================
# 2. RSSM 架构
# ============================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model（简化版）。

    核心变量：
      h_t: 确定性隐状态（GRU 输出），记忆历史信息
      z_t: 随机隐状态（Gaussian），捕捉当前不确定性
      s_t = (h_t, z_t): 完整的 RSSM 状态

    两个核心函数：
      prior:    p(z_t | h_t)           — 不依赖观测，只看历史
      posterior: q(z_t | h_t, o_t)     — 依赖观测，更准确
    """

    def __init__(self, obs_dim=4, act_dim=2, stoch_dim=16, deter_dim=64):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # --- 确定性部分：GRU ---
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        # 输入到 GRU 的投影
        self.act_proj = nn.Linear(act_dim, deter_dim)
        self.z_proj = nn.Linear(stoch_dim, deter_dim)

        # --- 随机性部分：Gaussian ---
        # Prior: h_t → (mu, logstd)
        self.prior_net = nn.Linear(deter_dim, stoch_dim * 2)
        # Posterior: h_t + o_t → (mu, logstd)
        self.posterior_net = nn.Linear(deter_dim + obs_dim, stoch_dim * 2)

        # --- 观测重建 ---
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

    def get_stoch_state(self, mean, logstd):
        """从高斯分布采样。训练时加噪声，推理时用均值。"""
        std = torch.exp(logstd.clamp(-5, 2))  # 限制范围防止数值问题
        if self.training:
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        return z

    def prior(self, h):
        """Prior: p(z_t | h_t)，不依赖观测。"""
        out = self.prior_net(h)
        mu = out[:, :self.stoch_dim]
        logstd = out[:, self.stoch_dim:]
        z = self.get_stoch_state(mu, logstd)
        return z, mu, logstd

    def posterior(self, h, obs):
        """Posterior: q(z_t | h_t, o_t)，依赖观测（更准确）。"""
        x = torch.cat([h, obs], dim=-1)
        out = self.posterior_net(x)
        mu = out[:, :self.stoch_dim]
        logstd = out[:, self.stoch_dim:]
        z = self.get_stoch_state(mu, logstd)
        return z, mu, logstd

    def imagine_step(self, h, z, action):
        """
        想象一步（推理/规划时用 prior，不依赖真实观测）。
        对应 Dreamer V3 中的 "imagination rollout"。
        """
        gru_input = self.act_proj(action) + self.z_proj(z)
        h_next = self.gru(gru_input, h)
        z_next, _, _ = self.prior(h_next)
        return h_next, z_next

    def reconstruct(self, h, z):
        """从 RSSM 状态重建观测。"""
        x = torch.cat([h, z], dim=-1)
        return self.obs_decoder(x)


# ============================================================
# 3. 训练
# ============================================================

def train_rssm(model, dataloader, epochs=25, lr=3e-4, device="cpu", kl_balance=0.5):
    """
    RSSM 训练循环。

    损失 = 观测重建 + KL(posterior || prior) + RSSM 预测
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"recon_loss": [], "kl_loss": [], "total_loss": []}

    for epoch in range(epochs):
        model.train()
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for obs_seq, act_seq in dataloader:
            # obs_seq: [B, T, obs_dim], act_seq: [B, T, act_dim]
            B, T, obs_dim = obs_seq.shape
            obs_seq = obs_seq.to(device)
            act_seq = act_seq.to(device)

            recon_loss = 0.0
            kl_loss = 0.0

            # 初始化 RSSM 状态
            h = torch.zeros(B, model.deter_dim, device=device)
            z = torch.zeros(B, model.stoch_dim, device=device)

            for t in range(T):
                obs_t = obs_seq[:, t, :]
                act_t = act_seq[:, t, :]

                # 1. Prior
                z_prior, mu_prior, logstd_prior = model.prior(h)

                # 2. Posterior
                z_post, mu_post, logstd_post = model.posterior(h, obs_t)

                # 3. GRU 更新确定性状态
                gru_input = model.act_proj(act_t) + model.z_proj(z_post)
                h = model.gru(gru_input, h)

                # 4. 观测重建
                obs_recon = model.reconstruct(h, z_post)

                # 5. 损失
                recon_loss = recon_loss + F.mse_loss(obs_recon, obs_t)

                # KL 散度: KL(q(z|h,o) || p(z|h))
                # = log(p/q) = logstd_post - logstd_prior + (var_post + (mu_post - mu_prior)^2) / (2*var_prior) - 0.5
                var_post = torch.exp(2 * logstd_post)
                var_prior = torch.exp(2 * logstd_prior)
                kl = 0.5 * (
                    (var_post + (mu_post - mu_prior) ** 2) / (var_prior + 1e-8)
                    - 1
                    + 2 * (logstd_prior - logstd_post)
                )
                kl_loss = kl_loss + kl.mean()

                z = z_post  # 后续步骤用 posterior

            # 平均到序列长度
            recon_loss = recon_loss / T
            kl_loss = kl_loss / T

            # 自由比特（free nats）：KL 低于阈值时停止更新，防止 posterior 坍缩到 prior
            free_nats = 1.0
            kl_loss = torch.clamp(kl_loss, min=free_nats)

            loss = recon_loss + kl_balance * kl_loss

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（Dreamer 系列的标准做法）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["total_loss"].append(avg_recon + kl_balance * avg_kl)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_recon + kl_balance * avg_kl:.4f}")

    return history


# ============================================================
# 4. 想象展开（Imagination Rollout）
# ============================================================

def imagine_rollout(model, obs_seq, act_seq, device="cpu"):
    """
    演示 RSSM 的核心能力：想象展开（不依赖真实观测，用 prior 预测未来）。

    对比 posterior 轨迹（用真实观测）vs prior 轨迹（纯想象）的差异。
    """
    model.eval()
    with torch.no_grad():
        obs_seq = obs_seq.unsqueeze(0).to(device)  # [1, T, obs_dim]
        act_seq = act_seq.unsqueeze(0).to(device)  # [1, T, act_dim]
        B, T, _ = obs_seq.shape

        # --- Posterior 轨迹（依赖真实观测，更准确） ---
        h_post = torch.zeros(B, model.deter_dim, device=device)
        z_post = torch.zeros(B, model.stoch_dim, device=device)
        post_recons = []

        for t in range(T):
            z_post, _, _ = model.posterior(h_post, obs_seq[:, t, :])
            gru_input = model.act_proj(act_seq[:, t, :]) + model.z_proj(z_post)
            h_post = model.gru(gru_input, h_post)
            recon = model.reconstruct(h_post, z_post)
            post_recons.append(recon.cpu().numpy())

        # --- Prior 轨迹（纯想象，不依赖观测） ---
        h_pri = torch.zeros(B, model.deter_dim, device=device)
        z_pri = torch.zeros(B, model.stoch_dim, device=device)
        pri_recons = []

        for t in range(T):
            h_pri, z_pri = model.imagine_step(h_pri, z_pri, act_seq[:, t, :])
            recon = model.reconstruct(h_pri, z_pri)
            pri_recons.append(recon.cpu().numpy())

        # 计算误差
        post_recons = np.array(post_recons).squeeze(1)  # [T, obs_dim]
        pri_recons = np.array(pri_recons).squeeze(1)
        ground_truth = obs_seq.squeeze(0).cpu().numpy()  # [T, obs_dim]

        post_err = np.linalg.norm(post_recons - ground_truth, axis=-1)
        pri_err = np.linalg.norm(pri_recons - ground_truth, axis=-1)

        return post_err, pri_err


def visualize_rssm(history, post_err, pri_err):
    """可视化训练过程和 Posterior vs Prior 对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 损失曲线
    axes[0].plot(history["recon_loss"], label="Reconstruction")
    axes[0].plot(history["kl_loss"], label="KL (posterior || prior)")
    axes[0].plot(history["total_loss"], label="Total", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("RSSM Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Posterior vs Prior 重建误差
    steps = range(len(post_err))
    axes[1].plot(steps, post_err, label="Posterior (用真实观测)", linewidth=2)
    axes[1].plot(steps, pri_err, label="Prior (纯想象)", linewidth=2)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Reconstruction Error (L2)")
    axes[1].set_title("Posterior vs Prior: 误差随时间累积")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Posterior/Prior 差距随时间的变化
    axes[2].fill_between(steps, post_err, pri_err, alpha=0.3, color="red", label="差距")
    axes[2].plot(steps, post_err, color="blue", label="Posterior")
    axes[2].plot(steps, pri_err, color="orange", label="Prior")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Error")
    axes[2].set_title("确定性 vs 随机性的差距")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rssm_training_analysis.png", dpi=150)
    print("\n[Saved] rssm_training_analysis.png")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dreamer V3 RSSM Demo")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stoch_dim", type=int, default=16)
    parser.add_argument("--deter_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("Dreamer V3 RSSM (Recurrent State-Space Model) Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # --- 1. 数据 ---
    print(f"\n[Data] 生成 {3000} 条带噪声轨迹 (seq_len={args.seq_len})...")
    dataset = NoisyTrajectoryDataset(num_samples=3000, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  观测维度: 4 (x, y, vx, vy)")
    print(f"  动作维度: 2")
    print(f"  噪声 std: 0.05")

    # --- 2. 模型 ---
    model = RSSM(
        obs_dim=4,
        act_dim=2,
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] RSSM")
    print(f"  确定性维度 h: {args.deter_dim}")
    print(f"  随机性维度 z: {args.stoch_dim}")
    print(f"  总参数量: {total_params:,}")

    # --- 3. 训练 ---
    print(f"\n[Train] 开始训练 ({args.epochs} epochs)...")
    print("-" * 60)
    history = train_rssm(model, dataloader, epochs=args.epochs, device=device)
    print("-" * 60)

    # --- 4. 想象展开 ---
    print("\n[Imagine] Posterior vs Prior 对比...")
    test_obs, test_act = dataset[0]
    post_err, pri_err = imagine_rollout(model, test_obs, test_act, device=device)

    print(f"  Posterior 平均误差: {post_err.mean():.4f} (用真实观测)")
    print(f"  Prior 平均误差:    {pri_err.mean():.4f} (纯想象)")
    print(f"  差距:              {pri_err.mean() - post_err.mean():.4f}")
    print()
    print("  → Posterior 比 Prior 准确（因为它能'看到'真实观测）")
    print("  → Prior 的误差随时间累积更快（只能靠历史'猜'未来）")
    print("  → 这就是 Dreamer 用 imagination + critic 训练策略的原因")

    # --- 5. 可视化 ---
    visualize_rssm(history, post_err, pri_err)

    # --- 6. 总结 ---
    print("\n" + "=" * 60)
    print("RSSM 核心概念回顾：")
    print("=" * 60)
    print("1. h_t (确定性 GRU): 记忆历史 → 捕捉可预测的运动学规律")
    print("2. z_t (随机性 Gaussian): 捕捉不确定的碰撞/摩擦/滑动")
    print("3. Prior:     p(z_t | h_t)       — 规划/想象时用（不看观测）")
    print("4. Posterior: q(z_t | h_t, o_t)   — 训练/更新时用（看观测）")
    print("5. KL(posterior || prior): 让 prior 学会预测 posterior")
    print("6. 想象展开: 用 prior 自回归预测未来 → 在'脑中'模拟环境")
    print()
    print("与 VLA 的关系：")
    print("  VLA 用 Transformer 编码历史 → RSSM 用 GRU 编码历史")
    print("  两者都可以用于策略学习，但 RSSM 更高效（隐藏维度更小）")
    print("=" * 60)


if __name__ == "__main__":
    main()