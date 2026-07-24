#!/usr/bin/env python3
"""
evaluation_framework.py
=======================
Retargeting 综合评估框架。

提供：
- 关节空间指标 (JAE, RMSE, Limit Violation)
- 任务空间指标 (FPE, Normalized FPE)
- 动态指标 (Jerk, Latency)
- 多种方法基准对比
- 可视化报告生成

使用示例：
    python evaluation_framework.py --method rule --n_samples 100
"""

import numpy as np
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    method_name: str
    n_samples: int
    
    # 关节空间
    mean_jae: float = 0.0  # rad
    max_jae: float = 0.0
    rmse_jae: float = 0.0
    limit_violation_rate: float = 0.0
    
    # 任务空间
    mean_fpe: float = 0.0  # m
    normalized_fpe: float = 0.0  # %
    per_finger_fpe: Dict[str, float] = None
    
    # 动态
    mean_jerk: float = 0.0
    mean_latency_ms: float = 0.0
    
    # 综合
    overall_score: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class SimpleHandModel:
    """
    简化机器人手模型（用于评估）
    
    5 根手指，每根 2 个关节（MCP + PIP），共 10 DOF
    """
    
    def __init__(self):
        # 手指长度（米）
        self.finger_lengths = {
            'thumb': [0.035, 0.030],
            'index': [0.045, 0.030],
            'middle': [0.050, 0.035],
            'ring': [0.048, 0.033],
            'pinky': [0.040, 0.028],
        }
        
        # 关节限位
        self.joint_limits = np.array([
            [0.0, 1.2], [0.0, 1.2],   # thumb
            [0.0, 1.2], [0.0, 1.2],   # index
            [0.0, 1.2], [0.0, 1.2],   # middle
            [0.0, 1.2], [0.0, 1.2],   # ring
            [0.0, 1.2], [0.0, 1.2],   # pinky
        ])
        
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.n_dof = 10
    
    def forward_kinematics(self, joints):
        """
        简化的 FK：计算 fingertip 位置
        
        Args:
            joints: [10] 关节角
        
        Returns:
            tips: dict {finger_name: [3] position}
        """
        tips = {}
        for i, finger in enumerate(self.finger_names):
            t1 = joints[i * 2]      # MCP
            t2 = joints[i * 2 + 1]  # PIP
            l1, l2 = self.finger_lengths[finger]
            
            # 简化 2D 平面模型（手指平面）
            x = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
            z = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y = 0.0
            
            # 每根手指有不同的基座偏移
            base_offsets = {
                'thumb': np.array([0.02, -0.02, 0.0]),
                'index': np.array([0.0, 0.01, 0.0]),
                'middle': np.array([0.0, 0.04, 0.0]),
                'ring': np.array([0.0, 0.07, 0.0]),
                'pinky': np.array([0.0, 0.10, 0.0]),
            }
            
            tips[finger] = base_offsets[finger] + np.array([x, y, z])
        
        return tips
    
    def get_fingertip_positions(self, joints):
        """返回 fingertips 的 numpy 数组"""
        tips = self.forward_kinematics(joints)
        return np.array([tips[f] for f in self.finger_names])


def generate_test_dataset(n_samples=100, seed=42):
    """
    生成测试数据集
    
    随机生成人手关节角，通过 FK 得到 landmarks，再通过优化得到 ground truth 机器人关节
    """
    np.random.seed(seed)
    dataset = []
    
    model = SimpleHandModel()
    
    for _ in range(n_samples):
        # 随机人手关节角（在合理范围内）
        human_joints = np.random.uniform(0.1, 1.0, 10)
        
        # 模拟：用 human_joints 作为 ground truth（实际应用中应为优化结果）
        gt_joints = human_joints * 0.9  # 模拟机器人关节
        
        # 生成 landmarks（从 human_joints FK）
        human_tips = model.get_fingertip_positions(human_joints)
        
        dataset.append({
            'landmarks': human_tips,  # 5 个 fingertip 位置
            'gt_joints': gt_joints,
            'human_joints': human_joints,
        })
    
    return dataset


def rule_based_retargeting(landmarks, scale=1.60):
    """Rule-based retargeting（模拟）"""
    # 从 landmarks 恢复近似的关节角
    # 简化：直接用 fingertip 距离估计弯曲程度
    joints = []
    for i in range(5):
        tip = landmarks[i]
        dist = np.linalg.norm(tip)
        # 简化的角度估计
        angle = np.clip(dist * scale, 0.0, 1.2)
        joints.extend([angle, angle * 0.8])
    return np.array(joints)


def vector_opt_retargeting(landmarks, model):
    """向量优化 retargeting（简化版）"""
    from scipy.optimize import least_squares
    
    def objective(joints):
        tips = model.get_fingertip_positions(joints)
        return (tips - landmarks).flatten()
    
    result = least_squares(
        objective,
        x0=np.ones(10) * 0.5,
        bounds=(model.joint_limits[:, 0], model.joint_limits[:, 1]),
        method='trf',
        ftol=1e-6,
    )
    return result.x


def evaluate_retargeting(
    retargeting_fn: Callable,
    test_dataset: List[Dict],
    robot_model: SimpleHandModel,
    method_name: str,
) -> EvaluationMetrics:
    """
    综合评估 retargeting 方法
    """
    n = len(test_dataset)
    
    jae_list = []
    fpe_list = []
    limit_violation_list = []
    latency_list = []
    
    per_finger_fpe = {f: [] for f in robot_model.finger_names}
    
    for sample in test_dataset:
        landmarks = sample['landmarks']
        gt_joints = sample.get('gt_joints')
        
        # 计时
        start = time.perf_counter()
        pred_joints = retargeting_fn(landmarks)
        latency = time.perf_counter() - start
        latency_list.append(latency)
        
        # 关节空间误差
        if gt_joints is not None:
            jae = np.mean(np.abs(pred_joints - gt_joints))
            jae_list.append(jae)
        
        # 任务空间误差
        pred_tips = robot_model.get_fingertip_positions(pred_joints)
        fpe = np.mean(np.linalg.norm(pred_tips - landmarks, axis=1))
        fpe_list.append(fpe)
        
        # 每根手指误差
        for i, finger in enumerate(robot_model.finger_names):
            finger_fpe = np.linalg.norm(pred_tips[i] - landmarks[i])
            per_finger_fpe[finger].append(finger_fpe)
        
        # 限位检查
        violations = np.any((pred_joints < robot_model.joint_limits[:, 0]) | 
                           (pred_joints > robot_model.joint_limits[:, 1]))
        limit_violation_list.append(1.0 if violations else 0.0)
    
    # 计算指标
    metrics = EvaluationMetrics(
        method_name=method_name,
        n_samples=n,
        mean_jae=np.mean(jae_list) if jae_list else 0.0,
        max_jae=np.max(jae_list) if jae_list else 0.0,
        rmse_jae=np.sqrt(np.mean([j**2 for j in jae_list])) if jae_list else 0.0,
        limit_violation_rate=np.mean(limit_violation_list),
        mean_fpe=np.mean(fpe_list),
        normalized_fpe=np.mean(fpe_list) / 0.09 * 100,  # 以 9cm 手长归一化
        per_finger_fpe={k: np.mean(v) * 1000 for k, v in per_finger_fpe.items()},  # mm
        mean_jerk=0.0,  # 单帧无 jerk
        mean_latency_ms=np.mean(latency_list) * 1000,
    )
    
    # 综合评分
    metrics.overall_score = compute_overall_score(metrics)
    
    return metrics


def compute_overall_score(m: EvaluationMetrics) -> float:
    """
    计算综合评分（0-100）
    """
    # JAE 评分 (0-25)
    jae_deg = m.mean_jae * 180 / np.pi
    jae_score = max(0, 25 - jae_deg * 3)
    
    # FPE 评分 (0-25)
    fpe_score = max(0, 25 - m.normalized_fpe * 2)
    
    # 限位评分 (0-25)
    limit_score = 25 * (1 - m.limit_violation_rate)
    
    # 延迟评分 (0-25)
    latency_score = max(0, 25 - m.mean_latency_ms * 0.5)
    
    return jae_score + fpe_score + limit_score + latency_score


def print_report(metrics: EvaluationMetrics):
    """打印评估报告"""
    print("=" * 60)
    print(f"Retargeting Evaluation Report")
    print(f"Method: {metrics.method_name}")
    print(f"Samples: {metrics.n_samples}")
    print("=" * 60)
    
    print(f"\nJoint Space:")
    print(f"  Mean JAE: {metrics.mean_jae:.4f} rad ({metrics.mean_jae * 180 / np.pi:.2f} deg)")
    print(f"  Max JAE:  {metrics.max_jae:.4f} rad ({metrics.max_jae * 180 / np.pi:.2f} deg)")
    print(f"  RMSE:     {metrics.rmse_jae:.4f} rad")
    print(f"  Limit Violation Rate: {metrics.limit_violation_rate * 100:.1f}%")
    
    print(f"\nTask Space:")
    print(f"  Mean FPE: {metrics.mean_fpe * 1000:.2f} mm")
    print(f"  Normalized FPE: {metrics.normalized_fpe:.1f}%")
    print(f"  Per-finger FPE (mm):")
    for finger, fpe in metrics.per_finger_fpe.items():
        print(f"    {finger.capitalize():6s}: {fpe:.2f}")
    
    print(f"\nDynamic:")
    print(f"  Mean Latency: {metrics.mean_latency_ms:.3f} ms")
    
    print(f"\nOverall Score: {metrics.overall_score:.1f}/100")
    print("=" * 60)


def benchmark_all_methods(test_dataset, robot_model):
    """对比所有方法"""
    methods = {
        'Rule-based (scale=1.0)': lambda lm: rule_based_retargeting(lm, scale=1.0),
        'Rule-based (scale=1.6)': lambda lm: rule_based_retargeting(lm, scale=1.6),
        'Vector Optimization': lambda lm: vector_opt_retargeting(lm, robot_model),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("Benchmarking All Methods")
    print("=" * 80)
    
    for name, fn in methods.items():
        print(f"\n[{name}]")
        metrics = evaluate_retargeting(fn, test_dataset, robot_model, name)
        results[name] = metrics
        print_report(metrics)
    
    # 汇总表格
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"{'Method':<30} {'JAE(deg)':<12} {'FPE(mm)':<12} {'Limit%':<10} {'Latency(ms)':<12} {'Score':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<30} {m.mean_jae * 180 / np.pi:<12.2f} {m.mean_fpe * 1000:<12.2f} "
              f"{m.limit_violation_rate * 100:<10.1f} {m.mean_latency_ms:<12.3f} {m.overall_score:<8.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Retargeting Evaluation Framework")
    parser.add_argument("--method", choices=["rule", "vector", "all"], default="all")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="保存 JSON 报告")
    args = parser.parse_args()
    
    # 生成测试数据
    print(f"Generating test dataset ({args.n_samples} samples)...")
    dataset = generate_test_dataset(n_samples=args.n_samples)
    
    # 机器人模型
    model = SimpleHandModel()
    
    if args.method == "all":
        results = benchmark_all_methods(dataset, model)
    else:
        if args.method == "rule":
            fn = lambda lm: rule_based_retargeting(lm, scale=1.6)
            name = "Rule-based"
        else:
            fn = lambda lm: vector_opt_retargeting(lm, model)
            name = "Vector Optimization"
        
        metrics = evaluate_retargeting(fn, dataset, model, name)
        print_report(metrics)
        results = {name: metrics}
    
    # 保存报告
    if args.output:
        report = {k: v.to_dict() for k, v in results.items()}
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
