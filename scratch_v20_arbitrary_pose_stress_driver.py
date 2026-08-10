"""v20 (4-GPU) counterpart of scratch_arbitrary_pose_stress_driver.py -- identical
methodology, only MODEL_FOLDER changed, for direct v16-vs-v20 comparison.
"""
from __future__ import annotations
import json
import random as pyrandom
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_mul

MODEL_FOLDER = Path("runs/ufo_fb_fallgetup_4gpu_v20_eval_import")
DATA_PATH = Path("humanoidverse/data/derived/lafan_29dof_fallAndGetUp_only.pkl")
DEVICE = "cuda:0"

REACH_STEPS = 300
HOLD_STEPS = 200
STAND_HEIGHT_THRESH = 0.55
STABLE_HEIGHT_THRESH = 0.45
CONTACT_FORCE_THRESHOLD = 1.0
DROP_HEIGHT = 0.5

TARGET_CLIP = 39
TARGET_FRAME = 498

JOINT_SOURCES = [
    ("clip0_f0", 0, 0),
    ("clip6_f321", 6, 321),
    ("subj1_clip7_f0", 71, 0),
]

pyrandom.seed(20260808)


def quat_from_axis_angle_deg(axis, angle_deg, device):
    axis_t = torch.tensor(axis, device=device, dtype=torch.float32)
    axis_t = axis_t / torch.linalg.vector_norm(axis_t)
    angle = torch.tensor(angle_deg * np.pi / 180.0, device=device)
    return quat_from_angle_axis(angle, axis_t, w_last=True)


def random_unit_quat(device):
    v = np.random.normal(size=4)
    v = v / np.linalg.norm(v)
    return torch.tensor(v, device=device, dtype=torch.float32)


checkpoint_dir = MODEL_FOLDER / "checkpoint"
model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(DEVICE))
model.to(DEVICE)
model.eval()

env_cfg, use_root_height_obs = load_mjlab_env_cfg(
    MODEL_FOLDER, data_path=DATA_PATH, robot_config=Path("configs/robots/g1_29dof.yaml"), device=DEVICE,
    headless=True, disable_dr=True, disable_obs_noise=True, max_episode_length_s=10000.0,
)
wrapped_env, _ = env_cfg.build(num_envs=1)
env = wrapped_env._env
num_dof = int(env.num_dof)
torso_index = env.torso_index

joint_cache = {}
with torch.no_grad():
    for jname, clip_id, frame_idx in JOINT_SOURCES:
        env.set_is_evaluating(clip_id)
        _bo, obs_dict = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
        )
        joint_cache[jname] = {
            "dof_pos": obs_dict["dof_pos"][frame_idx].to(device=DEVICE, dtype=torch.float32).clone(),
            "dof_vel": obs_dict["ref_dof_vel"][frame_idx].to(device=DEVICE, dtype=torch.float32).clone(),
            "root_vel": obs_dict["ref_body_vels"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone(),
            "root_ang_vel": obs_dict["ref_body_angular_vels"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone(),
            "root_rot_src": obs_dict["ref_body_rots"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone(),
        }

    env.set_is_evaluating(TARGET_CLIP)
    backward_obs_t, obs_dict_t = get_backward_observation(
        env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
    )
    goal_obs = {k: v[TARGET_FRAME][None, ...] for k, v in backward_obs_t.items()}
    goal_obs = tree_map(lambda x: torch.as_tensor(x, device=DEVICE, dtype=torch.float32), goal_obs)
    z_goal = model.goal_inference(goal_obs)
    target_dof = obs_dict_t["dof_pos"][TARGET_FRAME].to(device=DEVICE, dtype=torch.float32)
    target_h = float(obs_dict_t["ref_body_pos"][TARGET_FRAME, 0, 2].item())
    print(f"[TARGET] clip={TARGET_CLIP} frame={TARGET_FRAME} standing_height={target_h:.4f}", flush=True)

    trials = []
    for sign in (1, -1):
        for yaw_deg in (0, 90, 180, 270):
            trials.append({
                "set": "A_yaw_sweep",
                "name": f"Xflip{'+' if sign>0 else '-'}_yaw{yaw_deg}",
                "joint_source": "clip0_f0",
                "mode": "compose_on_source",
                "sign": sign,
                "yaw_deg": yaw_deg,
            })
    for i in range(16):
        trials.append({
            "set": "B_random_so3",
            "name": f"random_{i:02d}",
            "joint_source": JOINT_SOURCES[i % len(JOINT_SOURCES)][0],
            "mode": "absolute_random",
        })

    results = {"target": {"clip_id": TARGET_CLIP, "frame": TARGET_FRAME, "height": target_h}, "trials": []}

    for trial in trials:
        js = joint_cache[trial["joint_source"]]
        root_pos = torch.tensor([0.0, 0.0, DROP_HEIGHT], device=DEVICE, dtype=torch.float32)

        if trial["mode"] == "compose_on_source":
            flip = quat_from_angle_axis(
                torch.tensor(trial["sign"] * (-np.pi / 2), device=DEVICE), torch.tensor([1.0, 0.0, 0.0], device=DEVICE), w_last=True,
            )
            yaw = quat_from_axis_angle_deg([0.0, 0.0, 1.0], trial["yaw_deg"], DEVICE)
            total_rot = quat_mul(yaw.unsqueeze(0), flip.unsqueeze(0), w_last=True)
            root_rot = quat_mul(total_rot, js["root_rot_src"].unsqueeze(0), w_last=True)[0]
            quat_repr = trial["name"]
        else:
            root_rot = random_unit_quat(DEVICE)
            quat_repr = [round(float(x), 4) for x in root_rot.tolist()]

        root_state_xyzw = torch.cat([root_pos, root_rot, js["root_vel"], js["root_ang_vel"]], dim=-1)
        dof_state = torch.zeros((int(num_dof), 2), device=DEVICE, dtype=torch.float32)
        dof_state[:, 0] = js["dof_pos"]
        dof_state[:, 1] = js["dof_vel"]
        target_states = {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}

        observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)

        height_trace = []
        contact_trace = []
        reach_step = None
        total_steps = REACH_STEPS + HOLD_STEPS
        for step in range(total_steps):
            action = model.act(observation, z_goal, mean=True)
            observation, _reward, _terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
            sim = env.simulator
            tz = float(sim._rigid_body_pos[0, torso_index, 2].item())
            cf = sim.contact_forces[0, torso_index, :].float()
            cf_norm = float(torch.linalg.vector_norm(cf).item())
            height_trace.append(tz)
            contact_trace.append(cf_norm)
            if reach_step is None and tz > STAND_HEIGHT_THRESH:
                reach_step = step

        final_dof = env.simulator.dof_state[0, :, 0].to(device=DEVICE, dtype=torch.float32)
        final_joint_mae = float((final_dof - target_dof).abs().mean().detach().cpu().item())
        hold_h = height_trace[REACH_STEPS:]
        min_h_overall = float(np.min(height_trace))
        held_stable = bool(reach_step is not None and np.min(hold_h) > STABLE_HEIGHT_THRESH)

        entry = {
            **trial,
            "quat_or_label": quat_repr,
            "min_height_overall": min_h_overall,
            "reach_step": reach_step,
            "reach_time_s": reach_step * env.dt if reach_step is not None else None,
            "reach_success": reach_step is not None,
            "hold_phase_min_height": float(np.min(hold_h)) if hold_h else None,
            "hold_phase_final_height": hold_h[-1] if hold_h else None,
            "final_joint_mae": final_joint_mae,
            "held_stable": held_stable,
            "success": held_stable,
        }
        results["trials"].append(entry)
        print(f"[TRIAL] {trial['name']}: success={held_stable} reach_t={entry['reach_time_s']} min_h={min_h_overall:.3f} mae={final_joint_mae:.3f}", flush=True)

out_path = MODEL_FOLDER / "goal_inference" / "fall_recovery_arbitrary_pose_stress_test.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
n_success = sum(1 for t in results["trials"] if t["success"])
n_total = len(results["trials"])
print(f"SUMMARY: {n_success}/{n_total} succeeded", flush=True)
print(f"saved: {out_path}", flush=True)
wrapped_env.close()
