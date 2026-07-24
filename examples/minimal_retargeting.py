#!/usr/bin/env python3
"""
minimal_retargeting.py
======================
最简化的 IK Retargeting 演示，对比三种方法：

1. Rule-based: 直接角度映射
2. Vector Optimization: scipy least_squares 任务空间 IK
3. Learning-based: 简单的 MLP 映射（需要训练数据）

使用模拟的 2D 手指模型演示核心思想。
"""

import numpy as np
from scipy.optimize import least_squares
import argparse


class SimpleFinger:
    """
    简化的 2D 手指模型（2 个关节）
    
    人手: 3 段骨骼 (MCP-PIP-DIP-TIP)
    机器人: 2 段骨骼 (MCP-PIP-TIP)，忽略 DIP
    """
    
    def __init__(self, l1=1.0, l2=0.8, l3=0.6):
        """
        Args:
            l1: MCP 到 PIP 长度（人手）
            l2: PIP 到 DIP 长度（人手）/ PIP 到 TIP 长度（机器人）
            l3: DIP 到 TIP 长度（人手，机器人无此段）
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
    
    def forward_kinematics_human(self, theta1, theta2, theta3):
        """人手 FK（3 关节）"""
        x0, y0 = 0.0, 0.0
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        x3 = x2 + self.l3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.l3 * np.sin(theta1 + theta2 + theta3)
        return np.array([x3, y3])
    
    def forward_kinematics_robot(self, theta1, theta2):
        """机器人 FK（2 关节）"""
        x0, y0 = 0.0, 0.0
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        return np.array([x2, y2])


def rule_based_retargeting(human_theta1, human_theta2, human_theta3, scale=1.0):
    """
    Rule-based: 直接映射前两个关节角，忽略人手第三个关节
    """
    robot_theta1 = human_theta1 * scale
    robot_theta2 = human_theta2 * scale
    
    # 关节限位
    robot_theta1 = np.clip(robot_theta1, 0, np.pi)
    robot_theta2 = np.clip(robot_theta2, 0, np.pi)
    
    return robot_theta1, robot_theta2


def vector_optimization_retargeting(finger, human_theta1, human_theta2, human_theta3):
    """
    Vector Optimization: 让机器人 fingertip 对齐人手指尖
    """
    # 目标：人手指尖位置
    target_tip = finger.forward_kinematics_human(human_theta1, human_theta2, human_theta3)
    
    def objective(robot_angles):
        robot_tip = finger.forward_kinematics_robot(robot_angles[0], robot_angles[1])
        return robot_tip - target_tip
    
    result = least_squares(
        objective,
        x0=[0.5, 0.5],
        bounds=([0, 0], [np.pi, np.pi]),
        method='trf',
        ftol=1e-8,
    )
    
    return result.x[0], result.x[1]


def generate_training_data(finger, n_samples=1000):
    """生成训练数据（人手关节 → 机器人关节的最优映射）"""
    X = []
    y = []
    
    for _ in range(n_samples):
        # 随机人手姿态
        h1 = np.random.uniform(0, np.pi)
        h2 = np.random.uniform(0, np.pi)
        h3 = np.random.uniform(0, np.pi)
        
        # 用 Vector Optimization 生成标签
        r1, r2 = vector_optimization_retargeting(finger, h1, h2, h3)
        
        X.append([h1, h2, h3])
        y.append([r1, r2])
    
    return np.array(X), np.array(y)


class SimpleMLPRetargeting:
    """简单的 MLP 学习映射"""
    
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        h = self.relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out
    
    def train(self, X, y, epochs=500, lr=0.01):
        """简单梯度下降训练"""
        n = len(X)
        
        for epoch in range(epochs):
            # 前向
            h = self.relu(X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            
            # 损失
            loss = np.mean((pred - y) ** 2)
            
            # 反向
            d_pred = 2 * (pred - y) / n
            dW2 = h.T @ d_pred
            db2 = np.sum(d_pred, axis=0)
            
            d_h = d_pred @ self.W2.T
            d_h[h <= 0] = 0  # ReLU 梯度
            
            dW1 = X.T @ d_h
            db1 = np.sum(d_h, axis=0)
            
            # 更新
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        return loss


def compare_methods():
    """对比三种 retargeting 方法"""
    
    finger = SimpleFinger(l1=1.0, l2=0.8, l3=0.6)
    
    # 测试人手姿态
    test_cases = [
        (0.3, 0.5, 0.2, "半弯曲"),
        (0.8, 1.0, 0.6, "较弯曲"),
        (1.2, 1.3, 0.8, "接近握拳"),
    ]
    
    print("=" * 70)
    print("Retargeting 方法对比")
    print("=" * 70)
    
    # 训练 Learning-based 模型
    print("\n[1] 生成训练数据并训练 MLP...")
    X_train, y_train = generate_training_data(finger, n_samples=2000)
    mlp = SimpleMLPRetargeting(input_dim=3, hidden_dim=32, output_dim=2)
    final_loss = mlp.train(X_train, y_train, epochs=500, lr=0.01)
    print(f"  最终训练损失: {final_loss:.6f}")
    
    print("\n[2] 测试对比")
    print("-" * 70)
    print(f"{'姿态':<10} {'人手指尖':<20} {'Rule-based':<20} {'Vector Opt':<20} {'MLP':<20}")
    print("-" * 70)
    
    for h1, h2, h3, name in test_cases:
        # 人手指尖
        human_tip = finger.forward_kinematics_human(h1, h2, h3)
        
        # Rule-based
        rb1, rb2 = rule_based_retargeting(h1, h2, h3, scale=1.0)
        rb_tip = finger.forward_kinematics_robot(rb1, rb2)
        rb_error = np.linalg.norm(rb_tip - human_tip)
        
        # Vector Optimization
        vo1, vo2 = vector_optimization_retargeting(finger, h1, h2, h3)
        vo_tip = finger.forward_kinematics_robot(vo1, vo2)
        vo_error = np.linalg.norm(vo_tip - human_tip)
        
        # MLP
        mlp_out = mlp.forward(np.array([h1, h2, h3]))
        mlp1, mlp2 = np.clip(mlp_out[0], 0, np.pi), np.clip(mlp_out[1], 0, np.pi)
        mlp_tip = finger.forward_kinematics_robot(mlp1, mlp2)
        mlp_error = np.linalg.norm(mlp_tip - human_tip)
        
        print(f"{name:<10} ({human_tip[0]:.2f},{human_tip[1]:.2f})       "
              f"err={rb_error:.3f}            err={vo_error:.3f}            err={mlp_error:.3f}")
    
    print("-" * 70)
    print("\n结论:")
    print("  - Rule-based: 最简单，但误差大（无法补偿尺寸差异）")
    print("  - Vector Opt: 精度最高，但需要实时求解优化问题")
    print("  - MLP: 推理快，但需要训练数据，精度取决于数据覆盖")


def main():
    parser = argparse.ArgumentParser(description="Minimal Retargeting Demo")
    parser.add_argument("--method", choices=["rule", "vector", "mlp", "compare"], 
                        default="compare", help="Retargeting method")
    args = parser.parse_args()
    
    if args.method == "compare":
        compare_methods()
    else:
        print(f"单独运行 {args.method} 模式（详见源码）")


if __name__ == "__main__":
    main()
