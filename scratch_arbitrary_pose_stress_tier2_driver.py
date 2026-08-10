"""Tier-2 harder stress test: tier-1 (24 trials, yaw-sweep + 16x random SO(3)
from z=0.5m, joints unchanged from source mocap frame) scored 24/24 success --
too easy to characterize any failure boundary. This tier pushes harder along
three axes simultaneously to actually find where recovery breaks:
  1. Larger random-SO(3) sample (40 draws) for a more solid success-rate stat.
  2. Varied drop height: 0.5 / 0.9 / 1.3 / 1.7m (all well beyond the fixed
     0.5m used in training's lie_down_init).
  3. Randomized joint angles: uniform noise added to dof_pos, beyond
     noise_to_initial_level=0 (training never randomizes joints) -- explicit
     "outside training coverage" probe.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.utils.helpers import get_backward_observation

MODEL_FOLDER = Path("runs/ufo_fb_fallgetup_1gpu_v16_eval_import")
DATA_PATH = Path("humanoidverse/data/derived/lafan_29dof_fallAndGetUp_only.pkl")
DEVICE = "cuda:0"

REACH_STEPS = 300
HOLD_STEPS = 200
STAND_HEIGHT_THRESH = 0.55
STABLE_HEIGHT_THRESH = 0.45

TARGET_CLIP = 39
TARGET_FRAME = 498

JOINT_SOURCES = [
    ("clip0_f0", 0, 0),
    ("clip6_f321", 6, 321),
    ("subj1_clip7_f0", 71, 0),
]

DROP_HEIGHTS = [0.5, 0.9, 1.3, 1.7]
N_RANDOM_PER_HEIGHT = 10
JOINT_NOISE_STD_RAD = 0.4

np.random.seed(20260808)


def random_unit_quat():
    v = np.random.normal(size=4)
    v = v / np.linalg.norm(v)
    return v


checkpoint_dir = MODEL_FOLDER / "checkpoint"
model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(DEVICE))
model.to(DEVICE)
model.eval()

env_cfg, use_root_height_obs = load_mjlab_env_cfg(
    MODEL_FOLDER, data_path=DATA_PATH, robot_config=None, device=DEVICE,
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
    idx = 0
    for h in DROP_HEIGHTS:
        for i in range(N_RANDOM_PER_HEIGHT):
            trials.append({
                "name": f"h{h}_r{i:02d}",
                "drop_height": h,
                "joint_source": JOINT_SOURCES[idx % len(JOINT_SOURCES)][0],
                "joint_noise": (idx % 2 == 1),
                "quat": random_unit_quat().tolist(),
            })
            idx += 1

    results = {"target": {"clip_id": TARGET_CLIP, "frame": TARGET_FRAME, "height": target_h}, "trials": []}

    for trial in trials:
        js = joint_cache[trial["joint_source"]]
        root_pos = torch.tensor([0.0, 0.0, trial["drop_height"]], device=DEVICE, dtype=torch.float32)
        root_rot = torch.tensor(trial["quat"], device=DEVICE, dtype=torch.float32)

        dof_pos = js["dof_pos"].clone()
        if trial["joint_noise"]:
            dof_pos = dof_pos + torch.randn_like(dof_pos) * JOINT_NOISE_STD_RAD

        root_state_xyzw = torch.cat([root_pos, root_rot, js["root_vel"], js["root_ang_vel"]], dim=-1)
        dof_state = torch.zeros((int(num_dof), 2), device=DEVICE, dtype=torch.float32)
        dof_state[:, 0] = dof_pos
        dof_state[:, 1] = js["dof_vel"]
        target_states = {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}

        observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)

        height_trace = []
        reach_step = None
        total_steps = REACH_STEPS + HOLD_STEPS
        for step in range(total_steps):
            action = model.act(observation, z_goal, mean=True)
            observation, _reward, _terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
            sim = env.simulator
            tz = float(sim._rigid_body_pos[0, torso_index, 2].item())
            height_trace.append(tz)
            if reach_step is None and tz > STAND_HEIGHT_THRESH:
                reach_step = step

        final_dof = env.simulator.dof_state[0, :, 0].to(device=DEVICE, dtype=torch.float32)
        final_joint_mae = float((final_dof - target_dof).abs().mean().detach().cpu().item())
        hold_h = height_trace[REACH_STEPS:]
        min_h_overall = float(np.min(height_trace))
        held_stable = bool(reach_step is not None and np.min(hold_h) > STABLE_HEIGHT_THRESH)

        entry = {
            **trial,
            "min_height_overall": min_h_overall,
            "reach_step": reach_step,
            "reach_time_s": reach_step * env.dt if reach_step is not None else None,
            "hold_phase_min_height": float(np.min(hold_h)) if hold_h else None,
            "hold_phase_final_height": hold_h[-1] if hold_h else None,
            "final_joint_mae": final_joint_mae,
            "success": held_stable,
        }
        results["trials"].append(entry)
        print(
            f"[TRIAL] {trial['name']} noise={trial['joint_noise']}: success={held_stable} "
            f"reach_t={entry['reach_time_s']} min_h={min_h_overall:.3f} mae={final_joint_mae:.3f}",
            flush=True,
        )

out_path = MODEL_FOLDER / "goal_inference" / "fall_recovery_stress_tier2.json"
out_path.write_text(json.dumps(results, indent=2))
n_success = sum(1 for t in results["trials"] if t["success"])
n_total = len(results["trials"])
print(f"SUMMARY: {n_success}/{n_total} succeeded ({100.0*n_success/n_total:.1f}%)", flush=True)
by_height = {}
for t in results["trials"]:
    by_height.setdefault(t["drop_height"], []).append(t["success"])
for h, vals in sorted(by_height.items()):
    print(f"  height={h}: {sum(vals)}/{len(vals)} success", flush=True)
by_noise = {}
for t in results["trials"]:
    by_noise.setdefault(t["joint_noise"], []).append(t["success"])
for n, vals in sorted(by_noise.items()):
    print(f"  joint_noise={n}: {sum(vals)}/{len(vals)} success", flush=True)
print(f"saved: {out_path}", flush=True)
wrapped_env.close()
