#!/usr/bin/env python3
"""
landmark_to_joint.py
====================
从人手 21 点 landmarks 到机器人关节角的 Rule-based 映射演示。

演示完整的 pipeline：
1. 模拟 MediaPipe 输出的 21 点坐标
2. 坐标系转换（手腕原点 + 尺度归一化 + 左右手镜像）
3. 计算弯曲角和外展角
4. 映射到 O10 灵巧手 10 个关节角
5. 应用关节限位
"""

import numpy as np
import argparse


# ============= MediaPipe 21 点索引 =============
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# ============= 坐标系转换 =============

def to_local_coordinates(landmarks):
    """转换到手腕局部坐标系"""
    wrist = landmarks[WRIST].copy()
    return landmarks - wrist


def normalize_scale(local_landmarks):
    """尺度归一化：以手腕到中指 MCP 距离为单位"""
    wrist = local_landmarks[WRIST]
    middle_mcp = local_landmarks[MIDDLE_MCP]
    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        return local_landmarks
    return local_landmarks / scale


def mirror_left_hand(landmarks, is_left):
    """左手 Y 轴镜像"""
    if is_left:
        landmarks = landmarks.copy()
        landmarks[:, 1] *= -1
    return landmarks


def preprocess_landmarks(landmarks, is_left=False):
    """完整预处理 pipeline"""
    local = to_local_coordinates(landmarks)
    normalized = normalize_scale(local)
    mirrored = mirror_left_hand(normalized, is_left)
    return mirrored


# ============= 角度计算 =============

def compute_flexion_angle(landmarks, i, j, k):
    """
    计算弯曲角：三个连续关键点形成的夹角
    
    Args:
        landmarks: [21, 3]
        i, j, k: 三个关键点的索引，j 是顶点
    """
    p1 = landmarks[i]
    p2 = landmarks[j]
    p3 = landmarks[k]
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_abduction_angle(landmarks, mcp1, mcp2):
    """
    计算外展角：两根手指 MCP 相对手腕的张开角度
    """
    wrist = landmarks[WRIST]
    p1 = landmarks[mcp1] - wrist
    p2 = landmarks[mcp2] - wrist
    
    cos_angle = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


# ============= O10 映射 =============

def map_landmarks_to_o10(landmarks, is_left=False):
    """
    将人手 21 点坐标映射到 O10 灵巧手 10 个关节角
    
    Args:
        landmarks: [21, 3] 3D 坐标（相机坐标系）
        is_left: 是否为左手
    
    Returns:
        joint_angles: [10] 关节角（弧度）
        debug_info: dict 调试信息
    """
    # 1. 预处理
    lm = preprocess_landmarks(landmarks, is_left)
    
    # 2. 计算各手指弯曲角
    joints = []
    
    # 拇指: CMC-MCP-IP
    thumb_mcp = compute_flexion_angle(lm, THUMB_CMC, THUMB_MCP, THUMB_IP)
    thumb_ip = compute_flexion_angle(lm, THUMB_MCP, THUMB_IP, THUMB_TIP)
    joints.extend([thumb_mcp, thumb_ip])
    
    # 食指: wrist-MCP-PIP (MCP 弯曲角)
    index_mcp = compute_flexion_angle(lm, WRIST, INDEX_MCP, INDEX_PIP)
    index_pip = compute_flexion_angle(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP)
    joints.extend([index_mcp, index_pip])
    
    # 中指
    middle_mcp = compute_flexion_angle(lm, WRIST, MIDDLE_MCP, MIDDLE_PIP)
    middle_pip = compute_flexion_angle(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP)
    joints.extend([middle_mcp, middle_pip])
    
    # 无名指
    ring_mcp = compute_flexion_angle(lm, WRIST, RING_MCP, RING_PIP)
    ring_pip = compute_flexion_angle(lm, RING_MCP, RING_PIP, RING_DIP)
    joints.extend([ring_mcp, ring_pip])
    
    # 小指
    pinky_mcp = compute_flexion_angle(lm, WRIST, PINKY_MCP, PINKY_PIP)
    pinky_pip = compute_flexion_angle(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP)
    joints.extend([pinky_mcp, pinky_pip])
    
    joints = np.array(joints)
    
    # 3. 应用缩放系数（补偿 landmark → curl 的衰减）
    scale_factor = 1.60
    joints = joints * scale_factor
    
    # 4. 裁剪到 O10 关节限位
    o10_limits = np.array([
        [0.0, 1.2], [0.0, 1.2],   # 拇指
        [0.0, 1.2], [0.0, 1.2],   # 食指
        [0.0, 1.2], [0.0, 1.2],   # 中指
        [0.0, 1.2], [0.0, 1.2],   # 无名指
        [0.0, 1.2], [0.0, 1.2],   # 小指
    ])
    joints_clipped = np.clip(joints, o10_limits[:, 0], o10_limits[:, 1])
    
    debug_info = {
        "raw_angles": joints / scale_factor,
        "scaled_angles": joints,
        "clipped_angles": joints_clipped,
        "is_left": is_left,
    }
    
    return joints_clipped, debug_info


# ============= 模拟数据生成 =============

def generate_open_hand():
    """生成张开手的 landmarks（模拟）"""
    landmarks = np.zeros((21, 3))
    
    # 手腕在原点
    landmarks[WRIST] = [0, 0, 0]
    
    # 拇指（偏向一侧）
    landmarks[THUMB_CMC] = [0.02, -0.03, 0]
    landmarks[THUMB_MCP] = [0.04, -0.05, 0]
    landmarks[THUMB_IP] = [0.06, -0.06, 0]
    landmarks[THUMB_TIP] = [0.08, -0.065, 0]
    
    # 食指（张开）
    landmarks[INDEX_MCP] = [0.05, 0.01, 0]
    landmarks[INDEX_PIP] = [0.09, 0.02, 0]
    landmarks[INDEX_DIP] = [0.12, 0.025, 0]
    landmarks[INDEX_TIP] = [0.15, 0.03, 0]
    
    # 中指（最长）
    landmarks[MIDDLE_MCP] = [0.05, 0.04, 0]
    landmarks[MIDDLE_PIP] = [0.10, 0.06, 0]
    landmarks[MIDDLE_DIP] = [0.14, 0.075, 0]
    landmarks[MIDDLE_TIP] = [0.17, 0.085, 0]
    
    # 无名指
    landmarks[RING_MCP] = [0.045, 0.07, 0]
    landmarks[RING_PIP] = [0.085, 0.10, 0]
    landmarks[RING_DIP] = [0.12, 0.12, 0]
    landmarks[RING_TIP] = [0.145, 0.135, 0]
    
    # 小指
    landmarks[PINKY_MCP] = [0.035, 0.10, 0]
    landmarks[PINKY_PIP] = [0.065, 0.14, 0]
    landmarks[PINKY_DIP] = [0.09, 0.17, 0]
    landmarks[PINKY_TIP] = [0.11, 0.19, 0]
    
    return landmarks


def generate_fist():
    """生成握拳的 landmarks（模拟）"""
    landmarks = np.zeros((21, 3))
    
    landmarks[WRIST] = [0, 0, 0]
    
    # 拇指（扣在食指上）
    landmarks[THUMB_CMC] = [0.02, -0.03, 0]
    landmarks[THUMB_MCP] = [0.04, -0.02, 0]
    landmarks[THUMB_IP] = [0.05, 0.00, 0]
    landmarks[THUMB_TIP] = [0.055, 0.02, 0]
    
    # 四指卷曲
    for finger_base in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
        landmarks[finger_base] = [0.04, 0.02, 0]
        landmarks[finger_base + 1] = [0.04, 0.03, 0]
        landmarks[finger_base + 2] = [0.04, 0.035, 0]
        landmarks[finger_base + 3] = [0.04, 0.04, 0]
    
    return landmarks


# ============= 主程序 =============

def main():
    parser = argparse.ArgumentParser(description="Landmark to Joint Angle Mapping Demo")
    parser.add_argument("--hand", choices=["left", "right"], default="right",
                        help="左手或右手")
    parser.add_argument("--gesture", choices=["open", "fist"], default="open",
                        help="手势类型")
    args = parser.parse_args()
    
    is_left = (args.hand == "left")
    
    print("=" * 60)
    print(f"Landmark → Joint Mapping Demo ({args.hand} hand, {args.gesture})")
    print("=" * 60)
    
    # 生成模拟 landmarks
    if args.gesture == "open":
        landmarks = generate_open_hand()
    else:
        landmarks = generate_fist()
    
    print(f"\n[Input] 模拟 {args.gesture} hand landmarks [21, 3]")
    print(f"  Wrist: {landmarks[WRIST]}")
    print(f"  Index TIP: {landmarks[INDEX_TIP]}")
    print(f"  Middle TIP: {landmarks[MIDDLE_TIP]}")
    
    # 映射到 O10
    joints, debug = map_landmarks_to_o10(landmarks, is_left=is_left)
    
    # 输出结果
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    print(f"\n[Output] O10 Joint Angles (radians):")
    for i, name in enumerate(finger_names):
        mcp = joints[i * 2]
        pip = joints[i * 2 + 1]
        print(f"  {name:6s}: MCP={mcp:.3f}, PIP={pip:.3f}")
    
    print(f"\n[Debug] Scaling factor: 1.60")
    print(f"[Debug] Is left hand: {is_left}")
    
    print("\n" + "=" * 60)
    print("Note: This is a demo with synthetic landmarks.")
    print("For real tracking, integrate MediaPipe or InterHand.")
    print("=" * 60)


if __name__ == "__main__":
    main()
