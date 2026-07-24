"""
DexMV-Style High-Precision IK Retargeting
==========================================
Based on DexMV (ECCV 2022) retargeting algorithm, reimplemented with:
- MuJoCo 3.x for forward kinematics and Jacobian computation
- scipy.optimize.minimize (SLSQP) for IK optimization
- Huber Loss (Smooth L1) for robust position matching
- Sequential warm-start for temporal consistency

Reference: Qin et al., "DexMV: Imitation Learning for Dexterous Manipulation
from Human Videos", ECCV 2022.
"""

import os
import numpy as np
import mujoco
from scipy.optimize import minimize
from typing import List, Optional, Tuple


class HuberLoss:
    """Huber (Smooth L1) Loss for robust position matching."""

    def __init__(self, delta: float = 0.01):
        self.delta = delta

    def __call__(self, diff: np.ndarray) -> float:
        """Compute Huber loss for position differences."""
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        return np.sum(0.5 * quadratic ** 2 + self.delta * linear)

    def gradient(self, diff: np.ndarray) -> np.ndarray:
        """Gradient of Huber loss w.r.t. diff."""
        abs_diff = np.abs(diff)
        scale = np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff))
        return scale


class DexMVRetargeter:
    """
    High-precision IK retargeting using MuJoCo 3.x + scipy SLSQP.

    Optimizes robot joint angles to match target fingertip positions
    using Huber loss for robustness.
    """

    def __init__(
        self,
        model_path: str,
        fingertip_body_names: List[str],
        huber_delta: float = 0.01,
        smoothing_weight: float = 2e-3,
    ):
        """
        Args:
            model_path: Path to URDF/MJCF model file.
            fingertip_body_names: List of body names for fingertips.
            huber_delta: Delta parameter for Huber loss.
            smoothing_weight: Weight for temporal smoothing term.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.fingertip_body_names = fingertip_body_names
        self.huber = HuberLoss(delta=huber_delta)
        self.smoothing_weight = smoothing_weight

        # Get body IDs and joint info
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in fingertip_body_names
        ]

        # Determine controllable joint indices (exclude freejoint/world)
        self.joint_ids = []
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.joint_ids.append(i)

        self.n_dofs = len(self.joint_ids)

        # Extract joint limits
        self.joint_limits = np.zeros((self.n_dofs, 2))
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            if self.model.jnt_limited[jnt_id]:
                self.joint_limits[i, 0] = self.model.jnt_range[jnt_id, 0]
                self.joint_limits[i, 1] = self.model.jnt_range[jnt_id, 1]
            else:
                self.joint_limits[i, 0] = -np.pi
                self.joint_limits[i, 1] = np.pi

        print(f"[DexMVRetargeter] Loaded model: {self.n_dofs} DOFs, {len(self.body_ids)} fingertips")
        print(f"  Fingertips: {fingertip_body_names}")
        print(f"  Joint limits: [{self.joint_limits[:,0].min():.2f}, {self.joint_limits[:,1].max():.2f}]")

    def get_fingertip_positions(self) -> np.ndarray:
        """Get current fingertip positions via FK."""
        positions = np.zeros((len(self.body_ids), 3))
        for i, body_id in enumerate(self.body_ids):
            positions[i] = self.data.xpos[body_id].copy()
        return positions

    def compute_jacobian(self) -> np.ndarray:
        """
        Compute position Jacobian for all fingertips w.r.t. all joints.
        Returns: J of shape (n_fingertips * 3, n_dofs)
        """
        J = np.zeros((len(self.body_ids) * 3, self.n_dofs))
        for i, body_id in enumerate(self.body_ids):
            jac_body = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_body, None, body_id)
            # Extract columns for controllable joints
            for j, jnt_id in enumerate(self.joint_ids):
                dof_adr = self.model.jnt_dofadr[jnt_id]
                J[i * 3:(i + 1) * 3, j] = jac_body[:, dof_adr]
        return J

    def _objective(self, qpos: np.ndarray, target_pos: np.ndarray, last_qpos: np.ndarray) -> float:
        """Compute objective: Huber loss + smoothing term."""
        # Set joint angles
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = qpos[i]

        # Forward kinematics
        mujoco.mj_fwdPosition(self.model, self.data)

        # Get fingertip positions
        current_pos = self.get_fingertip_positions()

        # Huber loss on position error
        diff = current_pos - target_pos
        loss = self.huber(diff.flatten())

        # Temporal smoothing term
        if last_qpos is not None:
            loss += self.smoothing_weight * np.sum((qpos - last_qpos) ** 2)

        return loss

    def _gradient(self, qpos: np.ndarray, target_pos: np.ndarray, last_qpos: np.ndarray) -> np.ndarray:
        """Analytical gradient using Jacobian."""
        # Update qpos and FK
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = qpos[i]
        mujoco.mj_fwdPosition(self.model, self.data)

        # Get current positions and Jacobian
        current_pos = self.get_fingertip_positions()
        J = self.compute_jacobian()

        # Gradient of Huber loss
        diff = (current_pos - target_pos).flatten()
        huber_grad = self.huber.gradient(diff)

        # Chain rule: dLoss/dq = dLoss/dpos * dpos/dq = huber_grad^T @ J
        grad = huber_grad @ J

        # Smoothing term gradient
        if last_qpos is not None:
            grad += 2 * self.smoothing_weight * (qpos - last_qpos)

        return grad

    def retarget_single(
        self,
        target_pos: np.ndarray,
        init_qpos: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Retarget a single frame.

        Args:
            target_pos: Target fingertip positions, shape (n_fingertips, 3).
            init_qpos: Initial guess for joint angles.
            verbose: Print optimization info.

        Returns:
            Optimized joint angles, shape (n_dofs,).
        """
        if init_qpos is None:
            init_qpos = self.joint_limits.mean(axis=1)

        # Objective function wrapper for scipy
        def obj_fn(q):
            return self._objective(q, target_pos, None)

        def grad_fn(q):
            return self._gradient(q, target_pos, None)

        # SLSQP optimization with joint limits
        result = minimize(
            obj_fn,
            init_qpos,
            method="SLSQP",
            jac=grad_fn,
            bounds=[(self.joint_limits[i, 0], self.joint_limits[i, 1]) for i in range(self.n_dofs)],
            options={"ftol": 1e-5, "maxiter": 200, "disp": verbose},
        )

        if verbose:
            print(f"  Optimization: success={result.success}, nfev={result.nfev}, loss={result.fun:.6f}")

        return result.x

    def retarget_sequence(
        self,
        target_positions: np.ndarray,
        init_qpos: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Retarget a sequence of frames with warm-start.

        Args:
            target_positions: Sequence of target fingertip positions,
                              shape (n_frames, n_fingertips, 3).
            init_qpos: Initial guess for first frame.
            verbose: Print progress.

        Returns:
            Sequence of joint angles, shape (n_frames, n_dofs).
        """
        n_frames = target_positions.shape[0]
        qpos_sequence = np.zeros((n_frames, self.n_dofs))

        if init_qpos is None:
            last_qpos = self.joint_limits.mean(axis=1)
        else:
            last_qpos = init_qpos.copy()

        for i in range(n_frames):
            # Objective with smoothing term
            def obj_fn(q):
                return self._objective(q, target_positions[i], last_qpos)

            def grad_fn(q):
                return self._gradient(q, target_positions[i], last_qpos)

            result = minimize(
                obj_fn,
                last_qpos,
                method="SLSQP",
                jac=grad_fn,
                bounds=[(self.joint_limits[j, 0], self.joint_limits[j, 1]) for j in range(self.n_dofs)],
                options={"ftol": 1e-5, "maxiter": 200, "disp": False},
            )

            qpos_sequence[i] = result.x
            last_qpos = result.x.copy()

            if verbose and (i == 0 or (i + 1) % 10 == 0):
                print(f"  Frame {i + 1}/{n_frames}: loss={result.fun:.6f}")

        if verbose:
            print(f"[DexMVRetargeter] Retargeted {n_frames} frames")

        return qpos_sequence


class SyntheticHandDataGenerator:
    """Generate synthetic hand landmark data for testing retargeting."""

    def __init__(self, n_fingertips: int = 5):
        self.n_fingertips = n_fingertips

    def generate_open_hand(self, palm_center: np.ndarray = np.array([0.0, 0.0, 0.0])) -> np.ndarray:
        """Generate fingertip positions for an open hand pose."""
        # Relative positions (thumb, index, middle, ring, pinky)
        offsets = np.array([
            [0.02, 0.08, 0.0],   # thumb (outward)
            [0.03, 0.12, 0.0],   # index
            [0.0, 0.13, 0.0],    # middle
            [-0.03, 0.12, 0.0],  # ring
            [-0.05, 0.10, 0.0],  # pinky
        ])[:self.n_fingertips]
        return palm_center + offsets

    def generate_fist(self, palm_center: np.ndarray = np.array([0.0, 0.0, 0.0])) -> np.ndarray:
        """Generate fingertip positions for a fist pose."""
        offsets = np.array([
            [0.02, 0.03, 0.0],   # thumb
            [0.01, 0.04, 0.0],   # index
            [0.0, 0.04, 0.0],    # middle
            [-0.01, 0.04, 0.0],  # ring
            [-0.02, 0.03, 0.0],  # pinky
        ])[:self.n_fingertips]
        return palm_center + offsets

    def generate_pinch(self, palm_center: np.ndarray = np.array([0.0, 0.0, 0.0])) -> np.ndarray:
        """Generate fingertip positions for a pinch gesture."""
        offsets = np.array([
            [0.01, 0.06, 0.0],   # thumb
            [0.01, 0.06, 0.0],   # index (close to thumb)
            [0.0, 0.10, 0.0],    # middle
            [-0.02, 0.09, 0.0],  # ring
            [-0.04, 0.07, 0.0],  # pinky
        ])[:self.n_fingertips]
        return palm_center + offsets

    def generate_sequence(
        self,
        n_frames: int = 60,
        gesture_names: List[str] = ["open", "fist", "pinch"],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate a sequence transitioning between gestures.

        Returns:
            positions: (n_frames, n_fingertips, 3)
            labels: List of gesture labels per frame.
        """
        gestures = {
            "open": self.generate_open_hand(),
            "fist": self.generate_fist(),
            "pinch": self.generate_pinch(),
        }

        frames_per_gesture = n_frames // len(gesture_names)
        positions = np.zeros((n_frames, self.n_fingertips, 3))
        labels = []

        idx = 0
        for g_idx, gesture_name in enumerate(gesture_names):
            target = gestures[gesture_name]
            # Transition from previous or start from open
            if g_idx == 0:
                start = gestures["open"]
            else:
                start = gestures[gesture_names[g_idx - 1]]

            for f in range(frames_per_gesture):
                t = f / frames_per_gesture
                # Smooth transition with ease-in-out
                t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)
                positions[idx] = start + t_smooth * (target - start)
                labels.append(gesture_name)
                idx += 1

        # Fill remaining frames
        while idx < n_frames:
            positions[idx] = gestures[gesture_names[-1]]
            labels.append(gesture_names[-1])
            idx += 1

        return positions, labels


def landmarks_to_fingertip_positions(landmarks_21: np.ndarray) -> np.ndarray:
    """
    Extract fingertip positions from MediaPipe 21-point landmarks.

    Args:
        landmarks_21: (21, 3) array of hand landmarks.

    Returns:
        (5, 3) array of fingertip positions (thumb, index, middle, ring, pinky).
    """
    # MediaPipe fingertip indices: 4, 8, 12, 16, 20
    tip_indices = [4, 8, 12, 16, 20]
    return landmarks_21[tip_indices]


if __name__ == "__main__":
    # Quick test with Shadow Hand
    print("=" * 60)
    print("DexMV-Style Retargeting - Quick Test")
    print("=" * 60)

    # Use Shadow Hand from MuJoCo Menagerie
    shadow_xml = "../../pretrained/urdf/mujoco_menagerie/shadow_hand/right_hand.xml"

    # Shadow Hand fingertip body names
    shadow_tips = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"]

    retargeter = DexMVRetargeter(shadow_xml, shadow_tips, huber_delta=0.005)

    # Generate synthetic data
    gen = SyntheticHandDataGenerator(n_fingertips=5)
    target_pos, labels = gen.generate_sequence(n_frames=30)

    # Retarget
    qpos_seq = retargeter.retarget_sequence(target_pos, verbose=True)

    print(f"\nRetargeting complete!")
    print(f"Output shape: {qpos_seq.shape}")
    print(f"Joint angle range: [{qpos_seq.min():.3f}, {qpos_seq.max():.3f}]")
