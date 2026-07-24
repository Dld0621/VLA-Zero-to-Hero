"""
DexMV-Style Retargeting Pipeline
=================================
Complete pipeline: synthetic data → retargeting → evaluation → visualization.

Steps:
1. Generate synthetic hand landmark sequence (open → fist → pinch)
2. Extract fingertip positions from landmarks
3. Retarget to robot hand joint angles (DexMV-style SLSQP + Huber Loss)
4. Evaluate retargeting accuracy (FPE, JAE)
5. Visualize with MuJoCo

Usage:
    python run_pipeline.py --model shadow --n_frames 60 --visualize
    python run_pipeline.py --model allegro --n_frames 30
"""

import argparse
import numpy as np
import mujoco
import time
import os

from dexmv_retargeting import DexMVRetargeter, SyntheticHandDataGenerator, landmarks_to_fingertip_positions


def evaluate_retargeting(
    retargeter: DexMVRetargeter,
    qpos_sequence: np.ndarray,
    target_positions: np.ndarray,
) -> dict:
    """
    Evaluate retargeting quality.

    Metrics:
        - FPE (Fingertip Position Error): mean fingertip position error (mm)
        - JAE (Joint Angle Error): not applicable without ground truth
        - Max FPE: worst-case fingertip error
        - Mean Loss: mean Huber loss per frame
    """
    n_frames = len(qpos_sequence)
    fpe_list = []
    loss_list = []

    for i in range(n_frames):
        # Set joint angles
        for j, jnt_id in enumerate(retargeter.joint_ids):
            qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
            retargeter.data.qpos[qpos_adr] = qpos_sequence[i, j]

        mujoco.mj_fwdPosition(retargeter.model, retargeter.data)
        current_pos = retargeter.get_fingertip_positions()

        # FPE per fingertip
        errors = np.linalg.norm(current_pos - target_positions[i], axis=1)
        fpe_list.append(errors)

        # Huber loss
        diff = (current_pos - target_positions[i]).flatten()
        loss_list.append(retargeter.huber(diff))

    fpe_array = np.array(fpe_list)

    return {
        "mean_fpe_mm": np.mean(fpe_array) * 1000,  # convert to mm
        "max_fpe_mm": np.max(fpe_array) * 1000,
        "mean_fpe_per_finger_mm": np.mean(fpe_array, axis=0) * 1000,
        "mean_loss": np.mean(loss_list),
        "max_loss": np.max(loss_list),
    }


def get_reference_fingertip_positions(retargeter: DexMVRetargeter, model_name: str = "") -> np.ndarray:
    """Get fingertip positions at a reference (finger-straight) pose."""
    retargeter.data.qpos[:] = 0.0

    # For Allegro Hand, set a reasonable open-hand pose
    if "allegro" in model_name.lower():
        # Thumb: slight abduction and flexion
        # Index/Middle/Ring: slight abduction, no flexion
        for i, jnt_id in enumerate(retargeter.joint_ids):
            qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
            jnt_name = mujoco.mj_id2name(retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name and ("joint_0" in jnt_name or "joint_4" in jnt_name or "joint_8" in jnt_name):
                # MCP abduction: small positive angle
                retargeter.data.qpos[qpos_adr] = 0.1
            elif jnt_name and "joint_12" in jnt_name:
                # Thumb base: small angle
                retargeter.data.qpos[qpos_adr] = 0.2

    mujoco.mj_fwdPosition(retargeter.model, retargeter.data)
    return retargeter.get_fingertip_positions()


def visualize_retargeting(
    retargeter: DexMVRetargeter,
    qpos_sequence: np.ndarray,
    target_positions: np.ndarray,
    fps: float = 30.0,
    record: bool = False,
    output_video: str = "retargeting_output.mp4",
):
    """Visualize retargeting result with MuJoCo renderer."""
    try:
        from mujoco import renderer
    except ImportError:
        print("[Warning] MuJoCo renderer not available. Skipping visualization.")
        return

    n_frames = len(qpos_sequence)
    duration = n_frames / fps

    # Create renderer
    rend = renderer.Renderer(retargeter.model, height=480, width=640)

    # Setup camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = retargeter.body_ids[0]
    camera.distance = 0.4
    camera.azimuth = 135
    camera.elevation = -20

    scene = mujoco.MjvScene(retargeter.model, maxgeom=1000)
    context = mujoco.MjrContext(retargeter.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    print(f"[Visualize] Playing {n_frames} frames at {fps} FPS ({duration:.1f}s)")
    print("  Press Ctrl+C to stop")

    frame_interval = 1.0 / fps
    frames = []

    try:
        for i in range(n_frames):
            start_time = time.time()

            # Set joint angles
            for j, jnt_id in enumerate(retargeter.joint_ids):
                qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
                retargeter.data.qpos[qpos_adr] = qpos_sequence[i, j]

            mujoco.mj_forward(retargeter.model, retargeter.data)

            # Render
            rend.update_scene(retargeter.data, camera)
            pixels = rend.render()

            if record:
                frames.append(pixels)

            # Simple text display in terminal
            if i % 5 == 0:
                print(f"  Frame {i + 1}/{n_frames}", end="\r")

            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\n[Visualize] Stopped by user")

    print(f"\n[Visualize] Done ({len(frames)} frames recorded)" if record else "\n[Visualize] Done")

    # Save video if requested
    if record and frames:
        try:
            import imageio
            imageio.mimsave(output_video, frames, fps=fps)
            print(f"[Visualize] Video saved to {output_video}")
        except ImportError:
            print("[Warning] imageio not installed. Cannot save video.")
            print(f"  Install with: pip install imageio")


def main():
    parser = argparse.ArgumentParser(description="DexMV-Style Retargeting Pipeline")
    parser.add_argument("--model", type=str, default="shadow",
                        choices=["shadow", "allegro", "leap", "franka"],
                        help="Robot hand model to use")
    parser.add_argument("--n_frames", type=int, default=60,
                        help="Number of frames in the sequence")
    parser.add_argument("--gestures", type=str, nargs="+",
                        default=["open", "fist", "pinch"],
                        help="Gesture sequence to generate")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize with MuJoCo renderer")
    parser.add_argument("--record", action="store_true",
                        help="Record video (requires imageio)")
    parser.add_argument("--huber_delta", type=float, default=0.005,
                        help="Huber loss delta parameter")
    parser.add_argument("--smoothing", type=float, default=2e-3,
                        help="Temporal smoothing weight")
    args = parser.parse_args()

    print("=" * 70)
    print("DexMV-Style High-Precision IK Retargeting Pipeline")
    print("=" * 70)

    # --- Step 1: Select Model ---
    base_path = os.path.join("..", "..", "pretrained", "urdf")

    model_configs = {
        "shadow": {
            "xml": os.path.join(base_path, "mujoco_menagerie", "shadow_hand", "scene_right.xml"),
            "tips": ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"],
        },
        "allegro": {
            "xml": os.path.join(base_path, "allegro_hand_right", "allegro_hand_right.urdf"),
            "tips": ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"],
        },
        "leap": {
            "xml": os.path.join(base_path, "leap_hand_sim", "assets", "leap_hand", "robot.urdf"),
            "tips": ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"],
        },
    }

    if args.model not in model_configs:
        print(f"[Error] Model '{args.model}' not supported yet")
        return

    config = model_configs[args.model]
    print(f"\n[Step 1] Model: {args.model.upper()}")
    print(f"  XML: {config['xml']}")
    print(f"  Fingertips: {config['tips']}")

    if not os.path.exists(config["xml"]):
        print(f"[Error] Model file not found: {config['xml']}")
        print(f"  Please ensure URDF models are downloaded (see pretrained/urdf/README.md)")
        return

    # --- Step 2: Initialize Retargeter (to get reference fingertip positions) ---
    print(f"\n[Step 2] Initializing retargeter and calibrating workspace")

    retargeter = DexMVRetargeter(
        config["xml"],
        config["tips"],
        huber_delta=args.huber_delta,
        smoothing_weight=args.smoothing,
    )

    # Get reference fingertip positions (default pose)
    ref_positions = get_reference_fingertip_positions(retargeter, model_name=args.model)
    print(f"  Reference fingertip positions (default pose):")
    for name, pos in zip(config["tips"], ref_positions):
        print(f"    {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # --- Step 3: Generate Synthetic Data (scaled to robot hand workspace) ---
    print(f"\n[Step 3] Generating synthetic hand data (scaled to robot workspace)")
    print(f"  Gestures: {' → '.join(args.gestures)}")
    print(f"  Frames: {args.n_frames}")

    gen = SyntheticHandDataGenerator(n_fingertips=len(config["tips"]))
    target_positions, labels = gen.generate_sequence(
        n_frames=args.n_frames,
        gesture_names=args.gestures,
    )

    # Scale synthetic data to match robot hand workspace
    # Compute scale factor based on reference positions
    ref_spread = np.max(np.linalg.norm(ref_positions - ref_positions.mean(axis=0), axis=1))
    syn_spread = np.max(np.linalg.norm(target_positions[0] - target_positions[0].mean(axis=0), axis=1))
    scale_factor = ref_spread / (syn_spread + 1e-8)

    # Apply scaling and offset to match reference
    target_positions = target_positions * scale_factor * 0.8 + ref_positions.mean(axis=0)

    print(f"  Scale factor: {scale_factor:.3f}")
    print(f"  Target position range: [{target_positions.min():.3f}, {target_positions.max():.3f}] m")

    # --- Step 4: Retargeting ---
    print(f"\n[Step 4] Running retargeting (DexMV-style SLSQP + Huber Loss)")
    print(f"  Huber delta: {args.huber_delta}")
    print(f"  Smoothing weight: {args.smoothing}")
    start_time = time.time()

    qpos_sequence = retargeter.retarget_sequence(target_positions, verbose=True)

    elapsed = time.time() - start_time
    print(f"\n  Retargeting time: {elapsed:.2f}s ({elapsed / args.n_frames * 1000:.1f} ms/frame)")

    # --- Step 5: Evaluation ---
    print(f"\n[Step 5] Evaluating retargeting accuracy")

    metrics = evaluate_retargeting(retargeter, qpos_sequence, target_positions)

    print(f"  Mean FPE: {metrics['mean_fpe_mm']:.3f} mm")
    print(f"  Max FPE:  {metrics['max_fpe_mm']:.3f} mm")
    print(f"  Mean Loss: {metrics['mean_loss']:.6f}")
    print(f"  Max Loss:  {metrics['max_loss']:.6f}")

    if len(config["tips"]) == 5:
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    elif len(config["tips"]) == 4:
        finger_names = ["Thumb", "Index", "Middle", "Ring"]
    else:
        finger_names = [f"Finger {i}" for i in range(len(config["tips"]))]

    print(f"  Per-finger mean FPE:")
    for name, fpe in zip(finger_names, metrics["mean_fpe_per_finger_mm"]):
        print(f"    {name:8s}: {fpe:.3f} mm")

    # --- Step 6: Visualization ---
    if args.visualize:
        print(f"\n[Step 6] Visualizing")
        visualize_retargeting(
            retargeter, qpos_sequence, target_positions,
            fps=30.0, record=args.record,
        )

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("Pipeline Complete!")
    print(f"{'=' * 70}")
    print(f"Model:        {args.model.upper()}")
    print(f"Frames:       {args.n_frames}")
    print(f"Mean FPE:     {metrics['mean_fpe_mm']:.3f} mm")
    print(f"Max FPE:      {metrics['max_fpe_mm']:.3f} mm")
    print(f"Time/frame:   {elapsed / args.n_frames * 1000:.1f} ms")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
