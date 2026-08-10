"""Convert A3 boxing/action pickle trajectories into UFO robot_state_npz motion files.

Pure numpy, no IsaacLab dependency -- runs directly inside UFO's own venv.

Source pkl schema (beyondmimic's data/action, data/0810/* etc.):
    fps: float
    root_pos: [T, 3] float32   (world-frame pelvis position)
    root_rot: [T, 4] float32   (xyzw quaternion)
    dof_pos: [T, 30] or [T, 31] float32  (AGIDATA source column layout)

Target UFO robot_state_npz schema (humanoidverse/utils/motion_data/robot_state_readers.py):
    root_pos: [T, 3] float32
    root_quat: [T, 4] float32  (xyzw, matches configs/robots/a3.yaml root_quat_order)
    dof_pos: [T, N] float32
    joint_names: [N] string    (so the reader reorders/subsets by name, defensive)
    fps: float
    motion_key: string
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Source pkl column layout, 30-dof variant (boxing exports: single head joint).
# Verbatim copy of beyondmimic's scripts/convert_a3_pkl_to_motion_npz.py
# AGIDATA_DOF_JOINT_NAMES_30/31 -- must stay byte-identical to that source of
# truth. Do not re-derive; if beyondmimic's layout changes, resync from there.
AGIDATA_DOF_JOINT_NAMES_30 = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# 31-dof variant (parkour exports, head split into yaw + pitch).
AGIDATA_DOF_JOINT_NAMES_31 = (
    AGIDATA_DOF_JOINT_NAMES_30[:3]
    + ["head_yaw_joint", "head_pitch_joint"]
    + AGIDATA_DOF_JOINT_NAMES_30[4:]
)

SOURCE_LAYOUTS = {30: AGIDATA_DOF_JOINT_NAMES_30, 31: AGIDATA_DOF_JOINT_NAMES_31}

# UFO A3 control_joint_names order (from configs/robots/a3.yaml / a3_deploy_params.JOINT_NAMES),
# i.e. legs -> waist -> arms, 23-dof policy subset. Head/wrist columns present
# in the source pkl are dropped here since A3T2.5's UFO MJCF has no
# independent head/wrist joints (head is absent, wrists are fixed into elbow).
A3_CONTROL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def map_dof_positions(dof_pos: np.ndarray) -> np.ndarray:
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    width = dof_pos.shape[1]
    if width not in SOURCE_LAYOUTS:
        raise ValueError(f"Unsupported source dof_pos width {width}; expected one of {sorted(SOURCE_LAYOUTS)}")
    source_names = SOURCE_LAYOUTS[width]
    src_index = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in A3_CONTROL_JOINT_NAMES if name not in src_index]
    if missing:
        raise ValueError(f"Source pkl dof_pos (width={width}) is missing columns: {missing}")
    return np.stack([dof_pos[:, src_index[name]] for name in A3_CONTROL_JOINT_NAMES], axis=1)


def convert_file(input_path: Path, output_path: Path) -> dict:
    with input_path.open("rb") as f:
        data = pickle.load(f)

    fps = float(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat = np.asarray(data["root_rot"], dtype=np.float32)  # already xyzw
    dof_pos = map_dof_positions(np.asarray(data["dof_pos"], dtype=np.float32))

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{input_path}: root_pos must be [T,3], got {root_pos.shape}")
    if root_quat.ndim != 2 or root_quat.shape[1] != 4:
        raise ValueError(f"{input_path}: root_rot must be [T,4], got {root_quat.shape}")
    t = root_pos.shape[0]
    if root_quat.shape[0] != t or dof_pos.shape[0] != t:
        raise ValueError(f"{input_path}: frame count mismatch root_pos={t} root_rot={root_quat.shape[0]} dof_pos={dof_pos.shape[0]}")

    quat_norm = np.linalg.norm(root_quat, axis=1)
    if np.any(quat_norm <= 0.0) or np.any(~np.isfinite(quat_norm)):
        raise ValueError(f"{input_path}: root_rot contains zero or non-finite quaternions")

    motion_key = input_path.stem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=dof_pos,
        joint_names=np.asarray(A3_CONTROL_JOINT_NAMES),
        fps=np.float32(fps),
        motion_key=motion_key,
    )
    return {"frames": int(t), "fps": fps, "duration_s": float(t / fps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Source .pkl file paths.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write UFO robot_state npz files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for input_str in args.input:
        input_path = Path(input_str)
        output_path = output_dir / f"{input_path.stem}.npz"
        info = convert_file(input_path, output_path)
        print(f"[convert] {input_path.name} -> {output_path.name}  frames={info['frames']} fps={info['fps']} dur={info['duration_s']:.2f}s")


if __name__ == "__main__":
    main()
