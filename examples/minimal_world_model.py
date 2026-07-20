#!/usr/bin/env python3
"""
minimal_world_model.py
======================
一个最简化的世界模型实现：用 MLP 在 latent space 预测下一个状态。

不依赖任何预训练模型或仿真环境，用合成数据演示世界模型的核心思想：
  1. 观测编码器：将高维观测压缩到低维 latent space
  2. 转移模型（Transition Model）：在 latent space 预测 (o_t, a_t) → z_{t+1}
  3. 观测解码器：从 latent 重建观测

适合理解世界模型的基本架构和训练方式。
依赖：pip install torch matplotlib
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 合成数据集：模拟简单的 2D 点mass 环境
# ============================================================

class PointMassDataset(Dataset):
    """
    合成 2D 点质量数据集。
    状态 = [x, y, vx, vy]，动作 = [ax, ay]（加速度）。
    物理规则：s_{t+1} = s_t + v_t * dt + 0.5 * a_t * dt^2, v_{t+1} = v_t + a_t * dt
    """

    def __init__(self, num_samples=5000, seq_len=10, dt=0.1, noise_std=0.02):
        self.seq_len = seq_len
        self.dt = dt
        self.num_samples = num_samples

        # 生成轨迹
        self.trajectories = []  # list of (seq_len+1, 4) — state 长度为 seq_len+1
        self.actions = []       # list of (seq_len, 2) — action 长度为 seq_len

        for _ in range(num_samples):
            # 随机初始状态
            state = np.random.randn(4).astype(np.float32) * 0.5  # [x, y, vx, vy]

            trajectory = [state.copy()]
            actions_seq = []

            for t in range(seq_len):
                # 随机动作（加速度）
                action = np.random.randn(2).astype(np.float32) * 0.5  # [ax, ay]

                # 物理更新
                pos = state[:2]
                vel = state[2:]
                new_pos = pos + vel * dt + 0.5 * action * dt ** 2
                new_vel = vel + action * dt
                state = np.concatenate([new_pos, new_vel])

                # 加噪声模拟真实世界不确定性
                state += np.random.randn(4).astype(np.float32) * noise_std

                trajectory.append(state.copy())
                actions_seq.append(action)

            self.trajectories.append(np.array(trajectory, dtype=np.float32))
            self.actions.append(np.array(actions_seq, dtype=np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        traj = self.trajectories[idx]  # (seq_len+1, 4)
        act = self.actions[idx]       # (seq_len, 2)
        return traj, act


# ============================================================
# 2. 世界模型架构
# ============================================================

class Encoder(nn.Module):
    """将状态向量映射到低维 latent space。"""
    def __init__(self, state_dim=4, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2),  # 输出 mu 和 logvar
        )
        self.latent_dim = latent_dim

    def forward(self, state):
        """返回 (mu, logvar)，用于重参数化采样。"""
        out = self.net(state)  # [B, latent_dim*2]
        mu = out[:, :self.latent_dim]
        logvar = out[:, self.latent_dim:]
        return mu, logvar


class TransitionModel(nn.Module):
    """
    转移模型：在 latent space 预测下一状态。
    对应 RSSM 中的确定性部分（简化版）。
    """
    def __init__(self, latent_dim=16, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # 预测 delta_z
        )

    def forward(self, z_t, action):
        """
        Args:
            z_t: [B, latent_dim] 当前 latent state
            action: [B, action_dim] 当前动作
        Returns:
            z_pred: [B, latent_dim] 预测的下一时刻 latent state
        """
        x = torch.cat([z_t, action], dim=-1)
        delta_z = self.net(x)
        return z_t + delta_z  # 残差连接：预测变化量，加上当前状态


class RewardPredictor(nn.Module):
    """奖励预测头（MuZero 风格，不预测观测只预测 reward/value）。"""
    def __init__(self, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t):
        """预测标量 reward。"""
        return self.net(z_t).squeeze(-1)


class MinimalWorldModel(nn.Module):
    """
    最小世界模型 = Encoder + Transition + RewardPredictor。
    """
    def __init__(self, state_dim=4, action_dim=2, latent_dim=16):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim)
        self.transition = TransitionModel(latent_dim, action_dim)
        self.reward_head = RewardPredictor(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, state):
        """编码状态到 latent space（使用均值，用于推理）。"""
        mu, logvar = self.encoder(state)
        return mu

    def predict_next(self, z_t, action):
        """在 latent space 预测下一状态。"""
        return self.transition(z_t, action)

    def predict_reward(self, z_t):
        """从 latent state 预测 reward。"""
        return self.reward_head(z_t)


# ============================================================
# 3. 训练循环
# ============================================================

def train(world_model, dataloader, epochs=30, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)

    # 损失函数
    transition_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()

    history = {"transition_loss": [], "reward_loss": []}

    for epoch in range(epochs):
        world_model.train()
        total_trans_loss = 0.0
        total_rew_loss = 0.0
        n_batches = 0

        for trajectories, actions in dataloader:
            # trajectories: [B, seq_len+1, 4]
            # actions: [B, seq_len, 2]
            B, T_plus_1, state_dim = trajectories.shape
            T = T_plus_1 - 1

            trajectories = trajectories.to(device)
            actions = actions.to(device)

            trans_loss = 0.0
            rew_loss = 0.0

            for t in range(T):
                state_t = trajectories[:, t, :]      # [B, 4]
                state_tp1 = trajectories[:, t + 1, :] # [B, 4]
                action_t = actions[:, t, :]           # [B, 2]

                # 编码
                z_t = world_model.encode(state_t)
                z_tp1 = world_model.encode(state_tp1)

                # 预测
                z_pred = world_model.predict_next(z_t, action_t)

                # 构造伪 reward：目标靠近原点时 reward 高
                target_pos = state_tp1[:, :2]
                reward = -torch.norm(target_pos, dim=-1)  # 距离原点越近 reward 越高
                reward_pred = world_model.predict_reward(z_tp1)

                # 累积损失
                trans_loss = trans_loss + transition_loss_fn(z_pred, z_tp1)
                rew_loss = rew_loss + reward_loss_fn(reward_pred, reward)

            trans_loss = trans_loss / T
            rew_loss = rew_loss / T

            loss = trans_loss + 0.1 * rew_loss  # reward loss 权重较小

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_trans_loss += trans_loss.item()
            total_rew_loss += rew_loss.item()
            n_batches += 1

        avg_trans = total_trans_loss / max(n_batches, 1)
        avg_rew = total_rew_loss / max(n_batches, 1)
        history["transition_loss"].append(avg_trans)
        history["reward_loss"].append(avg_rew)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Trans Loss: {avg_trans:.4f} | Rew Loss: {avg_rew:.4f}")

    return history


# ============================================================
# 4. 多步展开预测 + 可视化
# ============================================================

def multi_step_predict(world_model, trajectory, actions, steps=10, device="cpu"):
    """
    用训练好的世界模型从初始状态开始，自主展开 N 步预测。
    """
    world_model.eval()
    trajectory = trajectory.to(device)
    actions = actions.to(device)

    # 编码初始状态
    z = world_model.encode(trajectory[:, 0, :])  # [1, latent_dim]

    predictions = []
    z_current = z

    for t in range(steps):
        z_pred = world_model.predict_next(z_current, actions[:, t, :])
        predictions.append(z_pred.detach().cpu().numpy())
        z_current = z_pred  # 自回归展开（误差会累积！）

    return np.array(predictions)  # [steps, 1, latent_dim]


def visualize_prediction(history):
    """绘制训练损失曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["transition_loss"], label="Transition Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Latent Transition Prediction Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["reward_loss"], label="Reward Prediction Loss", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Reward Prediction Loss (MuZero-style)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("world_model_training_loss.png", dpi=150)
    print("\n[Saved] world_model_training_loss.png")


# ============================================================
# 5. 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Minimal World Model Demo")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent space 维度")
    parser.add_argument("--seq_len", type=int, default=10, help="序列长度")
    args = parser.parse_args()

    print("=" * 60)
    print("Minimal World Model Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # --- 1. 创建数据 ---
    print(f"\n[Data] 生成 {5000} 条合成轨迹 (seq_len={args.seq_len})...")
    dataset = PointMassDataset(num_samples=5000, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  状态维度: 4 (x, y, vx, vy)")
    print(f"  动作维度: 2 (ax, ay)")
    print(f"  轨迹数: {len(dataset)}")

    # --- 2. 创建模型 ---
    print(f"\n[Model] MinimalWorldModel (latent_dim={args.latent_dim})")
    model = MinimalWorldModel(
        state_dim=4, action_dim=2, latent_dim=args.latent_dim
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    # --- 3. 训练 ---
    print(f"\n[Train] 开始训练 ({args.epochs} epochs)...")
    print("-" * 60)
    history = train(model, dataloader, epochs=args.epochs, device=device)
    print("-" * 60)
    print("[Train] 训练完成!")

    # --- 4. 多步展开预测 ---
    print("\n[Predict] 多步展开预测演示...")
    test_traj, test_act = dataset[0]
    test_traj = test_traj.unsqueeze(0)  # [1, seq_len+1, 4]
    test_act = test_act.unsqueeze(0)   # [1, seq_len, 2]

    predictions = multi_step_predict(model, test_traj, test_act, steps=args.seq_len, device=device)

    # 计算 latent prediction error
    z_ground_truth = []
    for t in range(args.seq_len):
        z_gt = model.encode(test_traj[:, t + 1, :].to(device))
        z_ground_truth.append(z_gt.detach().cpu().numpy())
    z_ground_truth = np.array(z_ground_truth)  # [steps, 1, latent_dim]

    per_step_error = np.linalg.norm(predictions - z_ground_truth, axis=-1)  # [steps, 1]
    print(f"  单步预测误差: {per_step_error[0, 0]:.4f}")
    print(f"  5步后误差:   {per_step_error[4, 0]:.4f}")
    print(f"  10步后误差:  {per_step_error[9, 0]:.4f}")
    print("  → 误差随步数累积，这是世界模型的核心挑战（compounding error）")

    # --- 5. 可视化 ---
    visualize_prediction(history)

    # --- 6. 关键总结 ---
    print("\n" + "=" * 60)
    print("核心概念回顾：")
    print("=" * 60)
    print("1. Encoder: 高维状态 → 低维 latent（信息压缩）")
    print("2. Transition Model: 在 latent space 预测动力学（z_t, a_t → z_{t+1}）")
    print("3. Reward Head: 从 latent 预测 reward（MuZero 风格，无需重建像素）")
    print("4. 多步展开: 误差累积是世界模型的核心挑战")
    print()
    print("对比 VLA：")
    print("  VLA:     观测 + 指令 → 动作（策略）")
    print("  世界模型: 观测 + 动作 → 下一观测（动力学）")
    print("  两者互补：世界模型帮 VLA '预判未来'")
    print("=" * 60)


if __name__ == "__main__":
    main()