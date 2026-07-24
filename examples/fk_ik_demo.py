#!/usr/bin/env python3
"""
fk_ik_demo.py
=============
正运动学（FK）与逆运动学（IK）基础演示。

使用简单的 2-DOF 平面机械臂演示：
- FK: 已知关节角 → 计算末端位置
- IK: 已知末端位置 → 求解关节角（解析法 + 数值法）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


class PlanarArm2D:
    """
    2-DOF 平面机械臂
    
    关节 1: 基座旋转 (theta1)
    关节 2: 肘部旋转 (theta2)
    """
    
    def __init__(self, link1_length=1.0, link2_length=1.0):
        self.l1 = link1_length
        self.l2 = link2_length
    
    def forward_kinematics(self, theta1, theta2):
        """
        正运动学
        
        Returns:
            (x, y): 末端执行器位置
            joints: [(x0,y0), (x1,y1), (x2,y2)] 各关节位置
        """
        x0, y0 = 0.0, 0.0
        
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        
        return (x2, y2), [(x0, y0), (x1, y1), (x2, y2)]
    
    def inverse_kinematics_analytical(self, x, y):
        """
        解析法逆运动学（2-DOF 平面臂有闭式解）
        
        Returns:
            (theta1, theta2): 关节角解（可能有多个解）
        """
        # 到目标的距离
        d = np.sqrt(x**2 + y**2)
        
        # 检查可达性
        if d > self.l1 + self.l2 or d < abs(self.l1 - self.l2):
            return None
        
        # 余弦定理求 theta2
        cos_theta2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        
        theta2_1 = np.arccos(cos_theta2)   # "肘部向上"
        theta2_2 = -np.arccos(cos_theta2)  # "肘部向下"
        
        # 求 theta1
        alpha = np.arctan2(y, x)
        beta = np.arctan2(self.l2 * np.sin(theta2_1), self.l1 + self.l2 * np.cos(theta2_1))
        
        theta1_1 = alpha - beta
        theta1_2 = alpha + beta  # 对应另一个 theta2 解
        
        return [(theta1_1, theta2_1), (theta1_2, theta2_2)]
    
    def inverse_kinematics_numerical(self, x, y, initial_guess=(0.5, 0.5), max_iter=100):
        """
        数值法逆运动学（Jacobian 迭代）
        
        适用于更复杂的机械臂（无闭式解时）
        """
        theta = np.array(initial_guess)
        target = np.array([x, y])
        
        for _ in range(max_iter):
            (xe, ye), _ = self.forward_kinematics(theta[0], theta[1])
            error = target - np.array([xe, ye])
            
            if np.linalg.norm(error) < 1e-4:
                return tuple(theta)
            
            # 计算 Jacobian
            J = self._jacobian(theta[0], theta[1])
            
            # 阻尼最小二乘
            damping = 0.1
            delta_theta = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(2)) @ error
            
            theta += delta_theta
        
        return tuple(theta)
    
    def _jacobian(self, theta1, theta2):
        """计算 Jacobian 矩阵"""
        J = np.array([
            [-self.l1*np.sin(theta1) - self.l2*np.sin(theta1+theta2), 
             -self.l2*np.sin(theta1+theta2)],
            [self.l1*np.cos(theta1) + self.l2*np.cos(theta1+theta2), 
             self.l2*np.cos(theta1+theta2)]
        ])
        return J


def demo_fk():
    """演示正运动学"""
    arm = PlanarArm2D(link1_length=1.0, link2_length=0.8)
    
    print("=" * 50)
    print("正运动学 (FK) 演示")
    print("=" * 50)
    
    test_angles = [
        (0.0, 0.0),
        (np.pi/4, np.pi/4),
        (np.pi/2, -np.pi/4),
        (np.pi/3, np.pi/3),
    ]
    
    for t1, t2 in test_angles:
        (x, y), joints = arm.forward_kinematics(t1, t2)
        print(f"\ntheta1={t1:.3f}, theta2={t2:.3f}")
        print(f"  末端位置: ({x:.3f}, {y:.3f})")
        print(f"  关节位置: {[(round(x,3), round(y,3)) for x,y in joints]}")


def demo_ik():
    """演示逆运动学"""
    arm = PlanarArm2D(link1_length=1.0, link2_length=0.8)
    
    print("\n" + "=" * 50)
    print("逆运动学 (IK) 演示")
    print("=" * 50)
    
    test_targets = [(1.2, 0.5), (0.5, 1.0), (1.5, 0.0)]
    
    for tx, ty in test_targets:
        print(f"\n目标位置: ({tx}, {ty})")
        
        # 解析法
        solutions = arm.inverse_kinematics_analytical(tx, ty)
        if solutions:
            for i, (t1, t2) in enumerate(solutions):
                (x, y), _ = arm.forward_kinematics(t1, t2)
                print(f"  解析法解 {i+1}: theta1={t1:.3f}, theta2={t2:.3f}")
                print(f"    验证 FK: ({x:.3f}, {y:.3f})")
        else:
            print("  目标不可达")
        
        # 数值法
        t1_num, t2_num = arm.inverse_kinematics_numerical(tx, ty)
        (x, y), _ = arm.forward_kinematics(t1_num, t2_num)
        print(f"  数值法解: theta1={t1_num:.3f}, theta2={t2_num:.3f}")
        print(f"    验证 FK: ({x:.3f}, {y:.3f})")


def animate_ik(target_x=1.2, target_y=0.5):
    """IK 求解动画演示"""
    arm = PlanarArm2D(link1_length=1.0, link2_length=0.8)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('2-DOF Arm IK Animation')
    
    # 目标点
    ax.plot(target_x, target_y, 'r*', markersize=15, label='Target')
    
    line, = ax.plot([], [], 'o-', linewidth=2, markersize=8, label='Arm')
    
    # 解析法求解
    solutions = arm.inverse_kinematics_analytical(target_x, target_y)
    if not solutions:
        print("目标不可达")
        return
    
    # 使用第一个解
    theta1, theta2 = solutions[0]
    
    # 从初始姿态插值到目标姿态
    n_frames = 60
    theta1_path = np.linspace(0, theta1, n_frames)
    theta2_path = np.linspace(0, theta2, n_frames)
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(frame):
        t1 = theta1_path[frame]
        t2 = theta2_path[frame]
        _, joints = arm.forward_kinematics(t1, t2)
        xs = [j[0] for j in joints]
        ys = [j[1] for j in joints]
        line.set_data(xs, ys)
        return line,
    
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=50, blit=True)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('fk_ik_animation.png', dpi=150)
    print("动画帧已保存为 fk_ik_animation.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="FK/IK Demo for 2-DOF Planar Arm")
    parser.add_argument("--mode", choices=["fk", "ik", "animate"], default="fk",
                        help="演示模式")
    parser.add_argument("--target", type=str, default="1.2,0.5",
                        help="IK 目标位置 (x,y)")
    args = parser.parse_args()
    
    if args.mode == "fk":
        demo_fk()
    elif args.mode == "ik":
        demo_ik()
    elif args.mode == "animate":
        tx, ty = map(float, args.target.split(','))
        animate_ik(tx, ty)


if __name__ == "__main__":
    main()
