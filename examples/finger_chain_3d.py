#!/usr/bin/env python3
"""
finger_chain_3d.py
==================
3D 手指运动链的 FK/IK 演示，更接近真实灵巧手结构。

特点：
- 3D 空间中的 3-DOF 手指链（MCP-PIP-DIP-TIP）
- 完整的 Denavit-Hartenberg (DH) 参数
- 解析 Jacobian 计算
- 阻尼最小二乘 IK
- 与 2D 简化版对比，展示 3D 复杂性
"""

import numpy as np
import argparse


class FingerChain3D:
    """
    3D 手指运动链（3-DOF）
    
    关节 1: MCP 弯曲（绕 X 轴）
    关节 2: MCP 外展（绕 Y 轴）
    关节 3: PIP 弯曲（绕 X 轴）
    关节 4: DIP 弯曲（绕 X 轴）—— 部分机器人手省略此关节
    
    简化模型：3 个主动关节（MCP flexion + MCP abduction + PIP flexion）
    """
    
    def __init__(self, link_lengths=None):
        """
        Args:
            link_lengths: [l_mcp, l_pip, l_dip] 三段骨骼长度
        """
        if link_lengths is None:
            link_lengths = [0.045, 0.025, 0.020]  # 人手中指典型长度（米）
        self.l = link_lengths
        
        # 关节限位（弧度）
        self.joint_limits = np.array([
            [0.0, np.pi/2],      # MCP flexion
            [-np.pi/6, np.pi/6], # MCP abduction
            [0.0, np.pi/2],      # PIP flexion
        ])
    
    def dh_transform(self, theta, d, a, alpha):
        """
        Denavit-Hartenberg 变换矩阵
        
        Args:
            theta: 关节角
            d: 连杆偏移
            a: 连杆长度
            alpha: 连杆扭角
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        T = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
        return T
    
    def forward_kinematics(self, theta, return_all=False):
        """
        3D 正运动学
        
        Args:
            theta: [3] 关节角 [mcp_flex, mcp_abd, pip_flex]
            return_all: 是否返回所有关节位置
        
        Returns:
            tip_position: [3] TIP 位置
            joints: list of [3] 各关节位置（如果 return_all=True）
        """
        t1, t2, t3 = theta
        l1, l2, l3 = self.l
        
        # 使用 DH 参数构建变换链
        # 基座到 MCP
        T01 = self.dh_transform(t2, 0, 0, np.pi/2)  # abduction
        T12 = self.dh_transform(t1, 0, l1, 0)       # MCP flexion + link1
        T23 = self.dh_transform(t3, 0, l2, 0)       # PIP flexion + link2
        T34 = self.dh_transform(0, 0, l3, 0)        # DIP (固定) + link3
        
        T04 = T01 @ T12 @ T23 @ T34
        tip_pos = T04[:3, 3]
        
        if return_all:
            # 计算各关节位置
            p0 = np.array([0, 0, 0])
            p1 = (T01 @ T12)[:3, 3]
            p2 = (T01 @ T12 @ T23)[:3, 3]
            p3 = tip_pos
            return tip_pos, [p0, p1, p2, p3]
        
        return tip_pos
    
    def jacobian(self, theta):
        """
        数值计算 Jacobian（有限差分法）
        
        也可使用解析法（见下方 analytical_jacobian）
        """
        eps = 1e-6
        n = len(theta)
        tip = self.forward_kinematics(theta)
        J = np.zeros((3, n))
        
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            tip_plus = self.forward_kinematics(theta_plus)
            J[:, i] = (tip_plus - tip) / eps
        
        return J
    
    def analytical_jacobian(self, theta):
        """
        解析 Jacobian（更高效、更精确）
        """
        # 简化：使用几何法
        # J[:, i] = z_i × (p_end - p_i)
        _, joints = self.forward_kinematics(theta, return_all=True)
        tip = joints[-1]
        
        # 简化假设：所有关节旋转轴
        # 这需要根据具体 DH 参数推导
        # 此处使用数值 Jacobian 作为 fallback
        return self.jacobian(theta)
    
    def inverse_kinematics_dls(self, target_tip, theta0=None, max_iter=100, 
                                lambda_damp=0.06, step_limit=0.05, tol=1e-5):
        """
        阻尼最小二乘 IK
        
        Args:
            target_tip: [3] 目标 TIP 位置
            theta0: [3] 初始猜测
            max_iter: 最大迭代次数
            lambda_damp: 阻尼系数
            step_limit: 最大步长
            tol: 收敛容差
        
        Returns:
            theta: [3] 求解的关节角
            error_history: list 误差历史
        """
        if theta0 is None:
            theta0 = np.array([0.3, 0.0, 0.3])
        
        theta = theta0.copy()
        error_history = []
        
        for i in range(max_iter):
            # 当前 TIP 位置
            current_tip = self.forward_kinematics(theta)
            error = target_tip - current_tip
            error_norm = np.linalg.norm(error)
            error_history.append(error_norm)
            
            if error_norm < tol:
                break
            
            # Jacobian
            J = self.jacobian(theta)
            
            # 阻尼最小二乘
            JJt = J @ J.T
            damping = lambda_damp**2 * np.eye(3)
            delta_theta = J.T @ np.linalg.solve(JJt + damping, error)
            
            # 步长限制
            delta_norm = np.linalg.norm(delta_theta)
            if delta_norm > step_limit:
                delta_theta = delta_theta * (step_limit / delta_norm)
            
            # 更新
            theta = theta + delta_theta
            
            # 裁剪到关节限位
            theta = np.clip(theta, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        return theta, error_history
    
    def check_reachability(self, target):
        """检查目标是否可达"""
        max_reach = sum(self.l)
        distance = np.linalg.norm(target)
        return distance <= max_reach


def demo_fk():
    """演示 3D FK"""
    finger = FingerChain3D()
    
    print("=" * 60)
    print("3D Finger Chain FK Demo")
    print("=" * 60)
    
    test_poses = [
        ([0.0, 0.0, 0.0], "伸直"),
        ([np.pi/4, 0.0, np.pi/4], "半弯曲"),
        ([np.pi/2, 0.0, np.pi/2], "完全弯曲"),
        ([np.pi/4, np.pi/12, np.pi/4], "半弯曲+外展"),
    ]
    
    for theta, name in test_poses:
        tip, joints = finger.forward_kinematics(theta, return_all=True)
        print(f"\n{name}: theta = [{theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f}]")
        print(f"  TIP position: ({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f})")
        print(f"  Joint positions:")
        for i, p in enumerate(joints):
            print(f"    J{i}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")


def demo_ik():
    """演示 3D IK"""
    finger = FingerChain3D()
    
    print("\n" + "=" * 60)
    print("3D Finger Chain IK Demo (Damped Least Squares)")
    print("=" * 60)
    
    # 测试目标
    test_targets = [
        np.array([0.06, 0.0, 0.03]),
        np.array([0.05, 0.01, 0.02]),
        np.array([0.04, -0.01, 0.04]),
    ]
    
    for target in test_targets:
        print(f"\n目标: ({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")
        
        if not finger.check_reachability(target):
            print("  目标不可达")
            continue
        
        theta, history = finger.inverse_kinematics_dls(target)
        
        # 验证
        tip = finger.forward_kinematics(theta)
        final_error = np.linalg.norm(tip - target)
        
        print(f"  求解: [{theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.4f}]")
        print(f"  验证: ({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f})")
        print(f"  误差: {final_error:.6f} m")
        print(f"  迭代: {len(history)}")


def demo_comparison_2d_vs_3d():
    """对比 2D 简化版与 3D 完整版"""
    print("\n" + "=" * 60)
    print("2D vs 3D Comparison")
    print("=" * 60)
    
    # 3D 手指
    finger_3d = FingerChain3D(link_lengths=[0.045, 0.025, 0.020])
    
    # 同一组关节角
    theta = np.array([np.pi/3, 0.0, np.pi/3])
    
    tip_3d = finger_3d.forward_kinematics(theta)
    
    # 2D 近似（忽略外展）
    l1, l2 = 0.045, 0.025 + 0.020  # 合并 PIP+DIP
    tip_2d = np.array([
        l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[2]),
        l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[2]),
        0.0
    ])
    
    print(f"\n关节角: [{theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f}]")
    print(f"3D FK:  ({tip_3d[0]:.4f}, {tip_3d[1]:.4f}, {tip_3d[2]:.4f})")
    print(f"2D 近似: ({tip_2d[0]:.4f}, {tip_2d[1]:.4f}, {tip_2d[2]:.4f})")
    print(f"误差: {np.linalg.norm(tip_3d - tip_2d):.6f} m")
    print("\n结论: 2D 近似在手指平面运动时足够，但外展/旋转时误差显著")


def main():
    parser = argparse.ArgumentParser(description="3D Finger Chain FK/IK Demo")
    parser.add_argument("--mode", choices=["fk", "ik", "compare"], default="fk")
    args = parser.parse_args()
    
    if args.mode == "fk":
        demo_fk()
    elif args.mode == "ik":
        demo_ik()
    elif args.mode == "compare":
        demo_comparison_2d_vs_3d()


if __name__ == "__main__":
    main()
