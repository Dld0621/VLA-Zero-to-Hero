"""
Freshman Zero-to-One: Human Hand Simulation + DexMV Retargeting
================================================================
大一新生也能跑通的完整 pipeline：人手仿真 → 坐标输入 → IK 重定向 → 机器人可视化

特点：
- 零外部依赖：除 numpy/scipy/mujoco/matplotlib 外无需任何开源代码
- 自包含：人手 21 点模型、DexMV 核心算法、机器人可视化全部内嵌
- 支持坐标输入：内置手势 / 文件读取 / 命令行参数

Usage:
    # 方式 1: 使用内置手势（推荐首次运行）
    python freshman_zero_to_one.py --mode builtin --gesture open
    python freshman_zero_to_one.py --mode builtin --gesture fist
    python freshman_zero_to_one.py --mode builtin --gesture pinch
    python freshman_zero_to_one.py --mode builtin --gesture sequence

    # 方式 2: 从 JSON 文件读取 21 点坐标
    python freshman_zero_to_one.py --mode file --input my_hand.json

    # 方式 3: 从 NumPy 文件读取 (.npy, shape=[21,3])
    python freshman_zero_to_one.py --mode file --input my_hand.npy

    # 方式 4: 指定机器人模型
    python freshman_zero_to_one.py --model allegro --gesture open

    # 方式 5: 显示人手可视化（matplotlib）+ 机器人重定向
    python freshman_zero_to_one.py --gesture open --visualize-human

输出：
    - 终端显示重定向精度（FPE）
    - 保存机器人关节角到 .npy 文件
    - 可选：MuJoCo 可视化窗口 / 人手 matplotlib 图
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# =============================================================================
# Part 0: 依赖检查（友好提示）
# =============================================================================

def check_dependencies():
    """检查必要依赖，给 freshman 友好提示。"""
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import mujoco
    except ImportError:
        missing.append("mujoco")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("=" * 60)
        print("[错误] 缺少以下依赖包，请先安装：")
        print("=" * 60)
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60)
        sys.exit(1)

check_dependencies()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import mujoco


# =============================================================================
# Part 1: 人手 21 点模型与可视化
# =============================================================================

class HumanHand21:
    """
    MediaPipe 风格的 21 点人手模型。

    关键点索引:
        0     : 手腕 (Wrist)
        1-4   : 拇指 (Thumb CMC, MCP, IP, Tip)
        5-8   : 食指 (Index MCP, PIP, DIP, Tip)
        9-12  : 中指 (Middle MCP, PIP, DIP, Tip)
        13-16 : 无名指 (Ring MCP, PIP, DIP, Tip)
        17-20 : 小指 (Pinky MCP, PIP, DIP, Tip)
    """

    # 连接关系，用于画图
    CONNECTIONS = [
        # 手掌骨架
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # 拇指
        (1, 2), (2, 3), (3, 4),
        # 食指
        (5, 6), (6, 7), (7, 8),
        # 中指
        (9, 10), (10, 11), (11, 12),
        # 无名指
        (13, 14), (14, 15), (15, 16),
        # 小指
        (17, 18), (18, 19), (19, 20),
    ]

    # 颜色映射
    COLORS = {
        "thumb": "#FF6B6B",    # 红色
        "index": "#4ECDC4",    # 青色
        "middle": "#45B7D1",   # 蓝色
        "ring": "#96CEB4",     # 绿色
        "pinky": "#FFEAA7",    # 黄色
        "palm": "#DDA0DD",     # 紫色
    }

    def __init__(self):
        self.landmarks = np.zeros((21, 3))

    # ------------------------------------------------------------------
    # 预定义手势（单位：米，以手腕为原点）
    # ------------------------------------------------------------------

    @staticmethod
    def get_open_hand() -> np.ndarray:
        """张开手（五指伸直张开）。"""
        landmarks = np.array([
            # 手腕
            [0.000,  0.000, 0.000],
            # 拇指 (CMC, MCP, IP, Tip) — 拇指外展
            [0.025,  0.010, 0.005],
            [0.035,  0.025, 0.008],
            [0.042,  0.040, 0.010],
            [0.048,  0.055, 0.012],
            # 食指
            [0.020,  0.055, 0.002],
            [0.022,  0.075, 0.003],
            [0.023,  0.095, 0.004],
            [0.024,  0.115, 0.005],
            # 中指（最长）
            [0.005,  0.060, 0.000],
            [0.006,  0.085, 0.001],
            [0.007,  0.110, 0.002],
            [0.008,  0.135, 0.003],
            # 无名指
            [-0.012, 0.055, 0.000],
            [-0.014, 0.075, 0.001],
            [-0.015, 0.095, 0.002],
            [-0.016, 0.115, 0.003],
            # 小指
            [-0.025, 0.045, 0.002],
            [-0.028, 0.060, 0.003],
            [-0.030, 0.075, 0.004],
            [-0.032, 0.090, 0.005],
        ], dtype=np.float32)
        return landmarks

    @staticmethod
    def get_fist() -> np.ndarray:
        """握拳（所有手指卷曲）。"""
        landmarks = np.array([
            [0.000,  0.000, 0.000],   # wrist
            [0.025,  0.010, 0.005],   # thumb cmc
            [0.032,  0.020, 0.012],   # thumb mcp
            [0.035,  0.028, 0.018],   # thumb ip
            [0.038,  0.035, 0.022],   # thumb tip
            [0.015,  0.035, 0.008],   # index mcp
            [0.018,  0.042, 0.015],   # index pip
            [0.020,  0.045, 0.020],   # index dip
            [0.022,  0.047, 0.024],   # index tip
            [0.005,  0.038, 0.006],   # middle mcp
            [0.007,  0.044, 0.012],   # middle pip
            [0.008,  0.047, 0.017],   # middle dip
            [0.009,  0.049, 0.021],   # middle tip
            [-0.005, 0.035, 0.006],   # ring mcp
            [-0.006, 0.041, 0.012],   # ring pip
            [-0.007, 0.044, 0.017],   # ring dip
            [-0.008, 0.046, 0.021],   # ring tip
            [-0.015, 0.030, 0.008],   # pinky mcp
            [-0.017, 0.036, 0.014],   # pinky pip
            [-0.018, 0.039, 0.019],   # pinky dip
            [-0.019, 0.041, 0.023],   # pinky tip
        ], dtype=np.float32)
        return landmarks

    @staticmethod
    def get_pinch() -> np.ndarray:
        """捏合（拇指与食指接触，其余手指张开）。"""
        landmarks = np.array([
            [0.000,  0.000, 0.000],   # wrist
            [0.025,  0.010, 0.005],   # thumb cmc
            [0.032,  0.022, 0.008],   # thumb mcp
            [0.028,  0.035, 0.010],   # thumb ip
            [0.025,  0.048, 0.012],   # thumb tip
            [0.015,  0.055, 0.002],   # index mcp
            [0.018,  0.070, 0.003],   # index pip
            [0.021,  0.085, 0.004],   # index dip
            [0.024,  0.098, 0.005],   # index tip (接近拇指)
            [0.005,  0.060, 0.000],   # middle mcp
            [0.006,  0.085, 0.001],   # middle pip
            [0.007,  0.110, 0.002],   # middle dip
            [0.008,  0.135, 0.003],   # middle tip
            [-0.012, 0.055, 0.000],   # ring mcp
            [-0.014, 0.075, 0.001],   # ring pip
            [-0.015, 0.095, 0.002],   # ring dip
            [-0.016, 0.115, 0.003],   # ring tip
            [-0.025, 0.045, 0.002],   # pinky mcp
            [-0.028, 0.060, 0.003],   # pinky pip
            [-0.030, 0.075, 0.004],   # pinky dip
            [-0.032, 0.090, 0.005],   # pinky tip
        ], dtype=np.float32)
        return landmarks

    @staticmethod
    def get_ok_sign() -> np.ndarray:
        """OK 手势（拇指与食指成环，其余三指伸直）。"""
        landmarks = np.array([
            [0.000,  0.000, 0.000],
            [0.025,  0.010, 0.005],
            [0.030,  0.020, 0.008],
            [0.028,  0.032, 0.012],
            [0.025,  0.042, 0.015],   # thumb tip
            [0.018,  0.048, 0.002],
            [0.020,  0.065, 0.003],
            [0.022,  0.082, 0.004],
            [0.024,  0.098, 0.005],   # index tip (接近拇指)
            [0.005,  0.060, 0.000],
            [0.006,  0.085, 0.001],
            [0.007,  0.110, 0.002],
            [0.008,  0.135, 0.003],
            [-0.012, 0.055, 0.000],
            [-0.014, 0.075, 0.001],
            [-0.015, 0.095, 0.002],
            [-0.016, 0.115, 0.003],
            [-0.025, 0.045, 0.002],
            [-0.028, 0.060, 0.003],
            [-0.030, 0.075, 0.004],
            [-0.032, 0.090, 0.005],
        ], dtype=np.float32)
        return landmarks

    @staticmethod
    def get_pointing() -> np.ndarray:
        """食指指向前方（其余手指握拳）。"""
        landmarks = np.array([
            [0.000,  0.000, 0.000],
            [0.025,  0.010, 0.005],
            [0.032,  0.020, 0.012],
            [0.035,  0.028, 0.018],
            [0.038,  0.035, 0.022],
            [0.015,  0.035, 0.008],
            [0.018,  0.055, 0.003],
            [0.020,  0.075, 0.004],
            [0.022,  0.095, 0.005],   # index tip (伸直)
            [0.005,  0.038, 0.006],
            [0.007,  0.044, 0.012],
            [0.008,  0.047, 0.017],
            [0.009,  0.049, 0.021],
            [-0.005, 0.035, 0.006],
            [-0.006, 0.041, 0.012],
            [-0.007, 0.044, 0.017],
            [-0.008, 0.046, 0.021],
            [-0.015, 0.030, 0.008],
            [-0.017, 0.036, 0.014],
            [-0.018, 0.039, 0.019],
            [-0.019, 0.041, 0.023],
        ], dtype=np.float32)
        return landmarks

    @classmethod
    def get_gesture(cls, name: str) -> np.ndarray:
        """按名称获取手势。"""
        gestures = {
            "open": cls.get_open_hand,
            "fist": cls.get_fist,
            "pinch": cls.get_pinch,
            "ok": cls.get_ok_sign,
            "pointing": cls.get_pointing,
        }
        if name not in gestures:
            raise ValueError(f"未知手势 '{name}'，可选: {list(gestures.keys())}")
        return gestures[name]()

    @classmethod
    def get_all_gesture_names(cls) -> List[str]:
        return ["open", "fist", "pinch", "ok", "pointing"]


class HumanHandVisualizer:
    """用 matplotlib 可视化 21 点人手（无需外部模型文件）。"""

    def __init__(self, figsize=(6, 6)):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Human Hand 21-Point Landmarks')

    def plot(self, landmarks: np.ndarray, title: str = "Human Hand"):
        """绘制单帧人手。"""
        self.ax.clear()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(title)

        # 绘制连线
        for start, end in HumanHand21.CONNECTIONS:
            pts = landmarks[[start, end]]
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'gray', linewidth=1.5, alpha=0.6)

        # 按手指分组绘制点
        groups = [
            ("palm",  [0],       "o"),
            ("thumb", [1,2,3,4], "s"),
            ("index", [5,6,7,8], "^"),
            ("middle",[9,10,11,12],"v"),
            ("ring",  [13,14,15,16],"D"),
            ("pinky", [17,18,19,20],"p"),
        ]
        for name, indices, marker in groups:
            pts = landmarks[indices]
            color = HumanHand21.COLORS.get(name, "black")
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, marker=marker, s=80, label=name, edgecolors='black', linewidths=0.5)

        # 标注 fingertip 索引
        tip_indices = [4, 8, 12, 16, 20]
        tip_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for idx, name in zip(tip_indices, tip_names):
            self.ax.text(landmarks[idx, 0], landmarks[idx, 1], landmarks[idx, 2],
                        f"  {name}", fontsize=8, color='red')

        # 设置等比例
        max_range = np.array([landmarks[:, 0].max() - landmarks[:, 0].min(),
                              landmarks[:, 1].max() - landmarks[:, 1].min(),
                              landmarks[:, 2].max() - landmarks[:, 2].min()]).max() / 2.0
        mid_x = (landmarks[:, 0].max() + landmarks[:, 0].min()) * 0.5
        mid_y = (landmarks[:, 1].max() + landmarks[:, 1].min()) * 0.5
        mid_z = (landmarks[:, 2].max() + landmarks[:, 2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.legend(loc='upper left', fontsize=7)
        plt.tight_layout()

    def show(self):
        plt.show()

    def save(self, path: str):
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[HumanHandVisualizer] 已保存人手图到 {path}")


def animate_hand_sequence(landmarks_seq: np.ndarray, labels: List[str] = None,
                          interval: int = 200, save_path: str = None):
    """
    动画展示人手序列。

    Args:
        landmarks_seq: (n_frames, 21, 3)
        labels: 每帧的标签
        interval: 帧间隔 (ms)
        save_path: 保存 GIF 路径（需要 pillow 库）
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        landmarks = landmarks_seq[frame]
        label = labels[frame] if labels else f"Frame {frame}"

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Human Hand: {label}')

        for start, end in HumanHand21.CONNECTIONS:
            pts = landmarks[[start, end]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'gray', linewidth=1.5, alpha=0.6)

        groups = [
            ("palm", [0], "o"), ("thumb", [1,2,3,4], "s"),
            ("index", [5,6,7,8], "^"), ("middle", [9,10,11,12], "v"),
            ("ring", [13,14,15,16], "D"), ("pinky", [17,18,19,20], "p"),
        ]
        for name, indices, marker in groups:
            pts = landmarks[indices]
            color = HumanHand21.COLORS.get(name, "black")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker=marker,
                      s=80, label=name, edgecolors='black', linewidths=0.5)

        max_range = np.array([landmarks[:, 0].max() - landmarks[:, 0].min(),
                              landmarks[:, 1].max() - landmarks[:, 1].min(),
                              landmarks[:, 2].max() - landmarks[:, 2].min()]).max() / 2.0
        mid = landmarks.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.legend(loc='upper left', fontsize=7)

    anim = FuncAnimation(fig, update, frames=len(landmarks_seq), interval=interval, blit=False)

    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"[Animation] 已保存动画到 {save_path}")
        except Exception as e:
            print(f"[Animation] 保存失败: {e}")
            print("  提示: pip install pillow")

    plt.show()
    return anim


# =============================================================================
# Part 2: DexMV 核心重定向算法（自包含版本）
# =============================================================================

class HuberLoss:
    """Huber (Smooth L1) Loss — 对小误差敏感，对大误差鲁棒。"""

    def __init__(self, delta: float = 0.01):
        self.delta = delta

    def __call__(self, diff: np.ndarray) -> float:
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        return np.sum(0.5 * quadratic ** 2 + self.delta * linear)

    def gradient(self, diff: np.ndarray) -> np.ndarray:
        abs_diff = np.abs(diff)
        return np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff))


class DexMVRetargeter:
    """
    DexMV-Style 高精度 IK Retargeting。

    核心思想：
        直接优化机器人 fingertip 位置与目标位置的匹配，
        使用 Huber Loss + SLSQP 优化 + 解析梯度（Jacobian）。
    """

    def __init__(self, model_path: str, fingertip_body_names: List[str],
                 huber_delta: float = 0.005, smoothing_weight: float = 2e-3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到模型文件: {model_path}\n"
                f"请确保已下载 URDF 模型到 pretrained/urdf/ 目录。\n"
                f"详见: pretrained/urdf/README.md"
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.fingertip_body_names = fingertip_body_names
        self.huber = HuberLoss(delta=huber_delta)
        self.smoothing_weight = smoothing_weight

        # 获取 fingertip body IDs
        self.body_ids = []
        for name in fingertip_body_names:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                self.body_ids.append(bid)
            except Exception as e:
                available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                             for i in range(self.model.nbody)]
                raise ValueError(
                    f"找不到 body '{name}'。\n"
                    f"可用 bodies: {available[:20]}...\n"
                    f"请检查 fingertip_body_names 是否正确。"
                )

        # 提取可控关节（排除 freejoint/world）
        self.joint_ids = []
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.joint_ids.append(i)

        self.n_dofs = len(self.joint_ids)

        # 提取关节限位
        self.joint_limits = np.zeros((self.n_dofs, 2))
        for i, jnt_id in enumerate(self.joint_ids):
            if self.model.jnt_limited[jnt_id]:
                self.joint_limits[i, 0] = self.model.jnt_range[jnt_id, 0]
                self.joint_limits[i, 1] = self.model.jnt_range[jnt_id, 1]
            else:
                self.joint_limits[i, 0] = -np.pi
                self.joint_limits[i, 1] = np.pi

        print(f"[DexMVRetargeter] 加载成功: {self.n_dofs} DOFs, {len(self.body_ids)} fingertips")
        print(f"  Fingertips: {fingertip_body_names}")

    def get_fingertip_positions(self) -> np.ndarray:
        """通过 FK 获取当前 fingertip 位置。"""
        positions = np.zeros((len(self.body_ids), 3))
        for i, body_id in enumerate(self.body_ids):
            positions[i] = self.data.xpos[body_id].copy()
        return positions

    def compute_jacobian(self) -> np.ndarray:
        """计算位置 Jacobian: J.shape = (n_fingertips*3, n_dofs)。"""
        J = np.zeros((len(self.body_ids) * 3, self.n_dofs))
        for i, body_id in enumerate(self.body_ids):
            jac_body = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_body, None, body_id)
            for j, jnt_id in enumerate(self.joint_ids):
                dof_adr = self.model.jnt_dofadr[jnt_id]
                J[i * 3:(i + 1) * 3, j] = jac_body[:, dof_adr]
        return J

    def _set_qpos(self, qpos: np.ndarray):
        """设置关节角度并更新 FK。"""
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = qpos[i]
        mujoco.mj_fwdPosition(self.model, self.data)

    def _objective(self, qpos: np.ndarray, target_pos: np.ndarray, last_qpos: Optional[np.ndarray]) -> float:
        self._set_qpos(qpos)
        current_pos = self.get_fingertip_positions()
        diff = current_pos - target_pos
        loss = self.huber(diff.flatten())
        if last_qpos is not None:
            loss += self.smoothing_weight * np.sum((qpos - last_qpos) ** 2)
        return loss

    def _gradient(self, qpos: np.ndarray, target_pos: np.ndarray, last_qpos: Optional[np.ndarray]) -> np.ndarray:
        self._set_qpos(qpos)
        current_pos = self.get_fingertip_positions()
        J = self.compute_jacobian()
        diff = (current_pos - target_pos).flatten()
        huber_grad = self.huber.gradient(diff)
        grad = huber_grad @ J
        if last_qpos is not None:
            grad += 2 * self.smoothing_weight * (qpos - last_qpos)
        return grad

    def retarget_single(self, target_pos: np.ndarray, init_qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """单帧重定向。"""
        if init_qpos is None:
            init_qpos = self.joint_limits.mean(axis=1)

        result = minimize(
            lambda q: self._objective(q, target_pos, None),
            init_qpos,
            method="SLSQP",
            jac=lambda q: self._gradient(q, target_pos, None),
            bounds=[(self.joint_limits[i, 0], self.joint_limits[i, 1]) for i in range(self.n_dofs)],
            options={"ftol": 1e-5, "maxiter": 200, "disp": False},
        )
        return result.x

    def retarget_sequence(self, target_positions: np.ndarray,
                          init_qpos: Optional[np.ndarray] = None,
                          verbose: bool = True) -> np.ndarray:
        """序列重定向（带时序平滑 + warm-start）。"""
        n_frames = target_positions.shape[0]
        qpos_sequence = np.zeros((n_frames, self.n_dofs))
        last_qpos = init_qpos.copy() if init_qpos is not None else self.joint_limits.mean(axis=1)

        for i in range(n_frames):
            result = minimize(
                lambda q: self._objective(q, target_positions[i], last_qpos),
                last_qpos,
                method="SLSQP",
                jac=lambda q: self._gradient(q, target_positions[i], last_qpos),
                bounds=[(self.joint_limits[j, 0], self.joint_limits[j, 1]) for j in range(self.n_dofs)],
                options={"ftol": 1e-5, "maxiter": 200, "disp": False},
            )
            qpos_sequence[i] = result.x
            last_qpos = result.x.copy()

            if verbose and (i == 0 or (i + 1) % 10 == 0 or i == n_frames - 1):
                print(f"  Frame {i+1}/{n_frames}: loss={result.fun:.6f}")

        return qpos_sequence


# =============================================================================
# Part 3: 坐标输入接口
# =============================================================================

def load_coordinates_from_file(filepath: str) -> np.ndarray:
    """
    从文件读取 21 点坐标。

    支持格式:
        - .json: {"landmarks": [[x,y,z], ...]} 或 {"left_landmarks": ..., "right_landmarks": ...}
        - .npy:  numpy array, shape=(21,3) 或 (n_frames, 21, 3)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)

        if 'landmarks' in data:
            coords = np.array(data['landmarks'], dtype=np.float32)
        elif 'left_landmarks' in data:
            coords = np.array(data['left_landmarks'], dtype=np.float32)
        elif 'right_landmarks' in data:
            coords = np.array(data['right_landmarks'], dtype=np.float32)
        else:
            raise ValueError(f"JSON 文件缺少 'landmarks' 键，实际键: {list(data.keys())}")

    elif ext == '.npy':
        coords = np.load(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .json 和 .npy")

    # 验证形状
    if coords.ndim == 2:
        if coords.shape != (21, 3):
            raise ValueError(f"坐标形状应为 (21, 3)，实际为 {coords.shape}")
        coords = coords[np.newaxis, ...]  # 变成 (1, 21, 3)
    elif coords.ndim == 3:
        if coords.shape[1:] != (21, 3):
            raise ValueError(f"坐标形状应为 (n_frames, 21, 3)，实际为 {coords.shape}")
    else:
        raise ValueError(f"坐标维度应为 2 或 3，实际为 {coords.ndim}")

    print(f"[Input] 从 {filepath} 读取了 {len(coords)} 帧坐标，形状: {coords.shape}")
    return coords


def save_landmarks_to_json(landmarks: np.ndarray, filepath: str):
    """保存 21 点坐标到 JSON 文件（方便手动编辑或传输）。"""
    data = {"landmarks": landmarks.tolist()}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Output] 已保存坐标到 {filepath}")


# =============================================================================
# Part 4: 机器人手模型配置
# =============================================================================

def get_robot_config(model_name: str, base_path: str = None) -> dict:
    """获取机器人手模型配置。"""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "..", "pretrained", "urdf")
        base_path = os.path.abspath(base_path)

    configs = {
        "shadow": {
            "xml": os.path.join(base_path, "mujoco_menagerie", "shadow_hand", "scene_right.xml"),
            "tips": ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"],
            "n_fingertips": 5,
            "finger_names": ["Thumb", "Index", "Middle", "Ring", "Pinky"],
        },
        "allegro": {
            "xml": os.path.join(base_path, "allegro_hand_right", "allegro_hand_right.urdf"),
            "tips": ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"],
            "n_fingertips": 4,
            "finger_names": ["Thumb", "Index", "Middle", "Ring"],
        },
        "leap": {
            "xml": os.path.join(base_path, "leap_hand_sim", "assets", "leap_hand", "robot.urdf"),
            "tips": ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"],
            "n_fingertips": 4,
            "finger_names": ["Thumb", "Index", "Middle", "Ring"],
        },
    }

    if model_name not in configs:
        raise ValueError(f"未知模型 '{model_name}'，可选: {list(configs.keys())}")

    return configs[model_name]


# =============================================================================
# Part 5: 评估与可视化
# =============================================================================

def evaluate_retargeting(retargeter: DexMVRetargeter, qpos_sequence: np.ndarray,
                         target_positions: np.ndarray) -> dict:
    """评估重定向精度。"""
    n_frames = len(qpos_sequence)
    fpe_list = []

    for i in range(n_frames):
        for j, jnt_id in enumerate(retargeter.joint_ids):
            qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
            retargeter.data.qpos[qpos_adr] = qpos_sequence[i, j]
        mujoco.mj_fwdPosition(retargeter.model, retargeter.data)
        current_pos = retargeter.get_fingertip_positions()
        errors = np.linalg.norm(current_pos - target_positions[i], axis=1)
        fpe_list.append(errors)

    fpe_array = np.array(fpe_list)
    return {
        "mean_fpe_mm": np.mean(fpe_array) * 1000,
        "max_fpe_mm": np.max(fpe_array) * 1000,
        "mean_fpe_per_finger_mm": np.mean(fpe_array, axis=0) * 1000,
    }


def visualize_robot_hand(retargeter: DexMVRetargeter, qpos_sequence: np.ndarray,
                         fps: float = 10.0, duration_sec: float = None):
    """用 MuJoCo 可视化机器人手（弹出交互窗口）。"""
    try:
        from mujoco import renderer
    except ImportError:
        print("[Warning] MuJoCo renderer 不可用，跳过机器人可视化。")
        print("  提示: 升级 mujoco >= 3.0")
        return

    n_frames = len(qpos_sequence)
    if duration_sec:
        fps = n_frames / duration_sec

    rend = renderer.Renderer(retargeter.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    if len(retargeter.body_ids) > 0:
        camera.trackbodyid = retargeter.body_ids[0]
    camera.distance = 0.35
    camera.azimuth = 135
    camera.elevation = -20

    print(f"[Robot Visualize] 播放 {n_frames} 帧 @ {fps:.1f} FPS")
    print("  关闭窗口或按 Ctrl+C 停止")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()
    im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    plt.title("Robot Hand Retargeting Result")
    plt.tight_layout()

    frame_interval = 1.0 / fps

    def update(frame):
        start = time.time()
        for j, jnt_id in enumerate(retargeter.joint_ids):
            qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
            retargeter.data.qpos[qpos_adr] = qpos_sequence[frame, j]
        mujoco.mj_forward(retargeter.model, retargeter.data)
        rend.update_scene(retargeter.data, camera)
        pixels = rend.render()
        im.set_array(pixels)

        elapsed = time.time() - start
        sleep_time = max(0, frame_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=frame_interval*1000, blit=False, repeat=True)
    plt.show()
    return anim


# =============================================================================
# Part 6: 完整 Pipeline（Main）
# =============================================================================

def run_pipeline(args):
    """ freshman 零到一完整 pipeline。"""

    print("=" * 70)
    print(" Freshman Zero-to-One: Human Hand → Robot Hand Retargeting")
    print("=" * 70)

    # ---------- Step 1: 获取人手坐标 ----------
    print("\n[Step 1/6] 获取人手 21 点坐标")

    if args.mode == "builtin":
        if args.gesture == "sequence":
            # 生成手势序列
            gesture_names = args.gesture_sequence.split(',')
            n_frames = args.n_frames
            frames_per = n_frames // len(gesture_names)
            landmarks_seq = []
            labels = []
            for gname in gesture_names:
                lm = HumanHand21.get_gesture(gname.strip())
                for _ in range(frames_per):
                    landmarks_seq.append(lm.copy())
                    labels.append(gname.strip())
            while len(landmarks_seq) < n_frames:
                landmarks_seq.append(landmarks_seq[-1].copy())
                labels.append(labels[-1])
            landmarks_seq = np.array(landmarks_seq)
            print(f"  内置手势序列: {' → '.join(gesture_names)}")
            print(f"  生成 {len(landmarks_seq)} 帧")
        else:
            landmarks_seq = HumanHand21.get_gesture(args.gesture)[np.newaxis, ...]
            labels = [args.gesture]
            print(f"  内置手势: {args.gesture}")
            print(f"  形状: {landmarks_seq.shape}")

    elif args.mode == "file":
        if not args.input:
            print("[错误] --mode file 时需要指定 --input 文件路径")
            sys.exit(1)
        landmarks_seq = load_coordinates_from_file(args.input)
        labels = [f"file_frame_{i}" for i in range(len(landmarks_seq))]

    else:
        print(f"[错误] 未知模式: {args.mode}")
        sys.exit(1)

    # ---------- Step 2: 人手可视化 ----------
    if args.visualize_human:
        print("\n[Step 2/6] 可视化人手")
        if len(landmarks_seq) == 1:
            vis = HumanHandVisualizer()
            vis.plot(landmarks_seq[0], title=f"Human Hand: {labels[0]}")
            if args.save_human:
                vis.save(args.save_human)
            vis.show()
        else:
            animate_hand_sequence(landmarks_seq, labels, interval=200,
                                  save_path=args.save_human)
    else:
        print("\n[Step 2/6] 人手可视化已跳过（加 --visualize-human 开启）")

    # ---------- Step 3: 提取 Fingertip 坐标 ----------
    print("\n[Step 3/6] 提取 Fingertip 坐标")
    # MediaPipe  fingertip 索引: 4, 8, 12, 16, 20
    tip_indices = [4, 8, 12, 16, 20]
    fingertip_positions = landmarks_seq[:, tip_indices, :]  # (n_frames, 5, 3)

    # 根据机器人手调整指数量
    robot_config = get_robot_config(args.model)
    n_robot_tips = robot_config["n_fingertips"]
    if n_robot_tips < fingertip_positions.shape[1]:
        fingertip_positions = fingertip_positions[:, :n_robot_tips, :]
        print(f"  机器人手有 {n_robot_tips} 指，取前 {n_robot_tips} 个 fingertip")
    print(f"  Fingertip 坐标形状: {fingertip_positions.shape}")

    # ---------- Step 4: Workspace 校准 ----------
    print("\n[Step 4/6] Workspace 校准（缩放到机器人手尺寸）")
    print(f"  机器人模型: {args.model.upper()}")

    if not os.path.exists(robot_config["xml"]):
        print(f"[错误] 找不到模型文件: {robot_config['xml']}")
        print("  请确保已下载 URDF 模型。运行以下命令:")
        print("    cd pretrained && python download_models.py")
        sys.exit(1)

    retargeter = DexMVRetargeter(
        robot_config["xml"],
        robot_config["tips"],
        huber_delta=args.huber_delta,
        smoothing_weight=args.smoothing,
    )

    # 获取参考 fingertip 散布，用于缩放
    retargeter.data.qpos[:] = 0.0
    mujoco.mj_fwdPosition(retargeter.model, retargeter.data)
    ref_positions = np.zeros((len(retargeter.body_ids), 3))
    for i, body_id in enumerate(retargeter.body_ids):
        ref_positions[i] = retargeter.data.xpos[body_id].copy()

    ref_spread = np.max(np.linalg.norm(ref_positions - ref_positions.mean(axis=0), axis=1))
    syn_spread = np.max(np.linalg.norm(fingertip_positions[0] - fingertip_positions[0].mean(axis=0), axis=1))
    scale_factor = ref_spread / (syn_spread + 1e-8)

    fingertip_positions = fingertip_positions * scale_factor * 0.8 + ref_positions.mean(axis=0)
    print(f"  缩放因子: {scale_factor:.3f}")
    print(f"  目标位置范围: [{fingertip_positions.min():.3f}, {fingertip_positions.max():.3f}] m")

    # ---------- Step 5: DexMV Retargeting ----------
    print("\n[Step 5/6] DexMV Retargeting (SLSQP + Huber Loss)")
    start_time = time.time()

    if len(landmarks_seq) == 1:
        qpos_result = retargeter.retarget_single(fingertip_positions[0])
        qpos_sequence = qpos_result[np.newaxis, ...]
    else:
        qpos_sequence = retargeter.retarget_sequence(fingertip_positions, verbose=True)

    elapsed = time.time() - start_time
    print(f"\n  重定向耗时: {elapsed:.3f}s ({elapsed / len(landmarks_seq) * 1000:.1f} ms/帧)")

    # ---------- Step 6: 评估 ----------
    print("\n[Step 6/6] 评估重定向精度")
    metrics = evaluate_retargeting(retargeter, qpos_sequence, fingertip_positions)

    print(f"  平均 FPE: {metrics['mean_fpe_mm']:.2f} mm")
    print(f"  最大 FPE: {metrics['max_fpe_mm']:.2f} mm")
    print(f"  每指平均 FPE:")
    for name, fpe in zip(robot_config["finger_names"], metrics["mean_fpe_per_finger_mm"]):
        print(f"    {name:8s}: {fpe:.2f} mm")

    # ---------- 保存结果 ----------
    if args.output:
        np.save(args.output, qpos_sequence)
        print(f"\n[Output] 机器人关节角已保存到 {args.output}，形状: {qpos_sequence.shape}")

    # ---------- 机器人可视化 ----------
    if args.visualize_robot:
        visualize_robot_hand(retargeter, qpos_sequence, fps=args.fps)
    else:
        print("\n[Info] 机器人可视化已跳过（加 --visualize-robot 开启 MuJoCo 窗口）")

    print("\n" + "=" * 70)
    print(" Pipeline 完成！")
    print("=" * 70)
    print(f" 输入:     {len(landmarks_seq)} 帧人手坐标")
    print(f" 模型:     {args.model.upper()} ({retargeter.n_dofs} DOFs)")
    print(f" 平均 FPE: {metrics['mean_fpe_mm']:.2f} mm")
    print(f" 速度:     {elapsed / len(landmarks_seq) * 1000:.1f} ms/帧")
    print("=" * 70)

    return qpos_sequence, metrics


# =============================================================================
# CLI Entry
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Freshman Zero-to-One: Human Hand → Robot Hand Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 内置手势 → Shadow Hand
  python freshman_zero_to_one.py --gesture open --visualize-human --visualize-robot

  # 手势序列 → Allegro Hand
  python freshman_zero_to_one.py --model allegro --gesture sequence --n_frames 30

  # 从文件读取坐标
  python freshman_zero_to_one.py --mode file --input my_hand.json --model shadow

  # 保存结果
  python freshman_zero_to_one.py --gesture fist --output joint_angles.npy
        """
    )

    # 输入模式
    parser.add_argument("--mode", type=str, default="builtin",
                        choices=["builtin", "file"],
                        help="坐标输入模式: builtin(内置手势) / file(从文件读取)")
    parser.add_argument("--input", type=str, default=None,
                        help="坐标文件路径 (.json 或 .npy)")

    # 内置手势
    parser.add_argument("--gesture", type=str, default="open",
                        choices=HumanHand21.get_all_gesture_names() + ["sequence"],
                        help="内置手势名称")
    parser.add_argument("--gesture-sequence", type=str, default="open,fist,pinch",
                        help="手势序列，逗号分隔（仅 --gesture sequence 时有效）")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="序列帧数（仅 --gesture sequence 时有效）")

    # 机器人模型
    parser.add_argument("--model", type=str, default="shadow",
                        choices=["shadow", "allegro", "leap"],
                        help="机器人手模型")

    # 可视化
    parser.add_argument("--visualize-human", action="store_true",
                        help="显示人手 matplotlib 可视化")
    parser.add_argument("--visualize-robot", action="store_true",
                        help="显示机器人手 MuJoCo 可视化")
    parser.add_argument("--save-human", type=str, default=None,
                        help="保存人手可视化图片/GIF路径")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="机器人可视化帧率")

    # 输出
    parser.add_argument("--output", type=str, default=None,
                        help="保存机器人关节角 .npy 文件路径")

    # 算法参数
    parser.add_argument("--huber-delta", type=float, default=0.005,
                        help="Huber Loss delta")
    parser.add_argument("--smoothing", type=float, default=2e-3,
                        help="时序平滑权重")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
