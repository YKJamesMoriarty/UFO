"""Goal-reaching-from-arbitrary-state probe: sample random (motion, frame)
pairs across the full motion library (dance/fight/walk/fallAndGetUp, not just
falls), reset directly to that pose (no drop / no SO(3) randomization -- this
isolates "reach and hold the goal" ability, distinct from the
fall_recovery_* driver scripts which test recovery from adversarial fallen
orientations), then roll out the z_goal-conditioned policy for
REACH_STEPS+HOLD_STEPS and record final joint MAE + torso height stability.
Output format matches v16's existing goal_stability_probe.json for direct
comparison; run identically against v16 and v20 for a fair test.

Usage: python scratch_goal_stability_probe_driver.py <model_folder_name>
"""
from __future__ import annotations
import json
import random as pyrandom
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.utils.helpers import get_backward_observation

MODEL_FOLDER = Path("runs") / sys.argv[1]
DATA_PATH = Path("humanoidverse/data/derived/lafan_29dof_fallAndGetUp_only.pkl")
DEVICE = "cuda:0"

REACH_STEPS = 300
HOLD_STEPS = 200
TARGET_CLIP = 39
TARGET_FRAME = 498
N_SAMPLES = 24

pyrandom.seed(20260809)
np.random.seed(20260809)

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
num_motions = env._motion_lib._num_unique_motions
motion_names = env._motion_lib._motion_data_keys.tolist()

results = {}
with torch.no_grad():
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

    chosen_motions = pyrandom.sample(range(num_motions), min(N_SAMPLES, num_motions))

    for m_id in chosen_motions:
        env.set_is_evaluating(m_id)
        _bo, obs_dict = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
        )
        n_frames = obs_dict["dof_pos"].shape[0]
        frame_idx = pyrandom.randint(0, n_frames - 1)

        dof_pos = obs_dict["dof_pos"][frame_idx].to(device=DEVICE, dtype=torch.float32).clone()
        dof_vel = obs_dict["ref_dof_vel"][frame_idx].to(device=DEVICE, dtype=torch.float32).clone()
        root_pos = obs_dict["ref_body_pos"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone()
        root_rot = obs_dict["ref_body_rots"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone()
        root_vel = obs_dict["ref_body_vels"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone()
        root_ang_vel = obs_dict["ref_body_angular_vels"][frame_idx, 0].to(device=DEVICE, dtype=torch.float32).clone()

        root_state_xyzw = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        dof_state = torch.zeros((num_dof, 2), device=DEVICE, dtype=torch.float32)
        dof_state[:, 0] = dof_pos
        dof_state[:, 1] = dof_vel
        target_states = {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}

        observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)

        height_trace = []
        terminated_step = None
        total_steps = REACH_STEPS + HOLD_STEPS
        for step in range(total_steps):
            action = model.act(observation, z_goal, mean=True)
            observation, _reward, terminated, _truncated, _info = wrapped_env.step(action, to_numpy=False)
            tz = float(env.simulator._rigid_body_pos[0, torso_index, 2].item())
            height_trace.append(tz)
            term_val = bool(terminated[0].item()) if torch.is_tensor(terminated) else bool(terminated)
            if terminated_step is None and term_val:
                terminated_step = step

        final_dof = env.simulator.dof_state[0, :, 0].to(device=DEVICE, dtype=torch.float32)
        final_joint_mae = float((final_dof - target_dof).abs().mean().detach().cpu().item())
        reach_h = height_trace[:REACH_STEPS]
        hold_h = height_trace[REACH_STEPS:]

        key = f"{motion_names[m_id]}_{frame_idx}"
        results[key] = {
            "motion_id": m_id,
            "frame_idx": frame_idx,
            "final_joint_mae_at_200": final_joint_mae,
            "terminated_step": terminated_step,
            "total_steps_executed": len(height_trace),
            "reach_phase_torso_z_mean": float(np.mean(reach_h)),
            "reach_phase_torso_z_final": float(reach_h[-1]),
            "hold_phase_torso_z_mean": float(np.mean(hold_h)) if hold_h else None,
            "hold_phase_torso_z_final": float(hold_h[-1]) if hold_h else None,
        }
        print(
            f"[SAMPLE] {key}: mae={final_joint_mae:.4f} term={terminated_step} "
            f"hold_h_mean={results[key]['hold_phase_torso_z_mean']:.3f}",
            flush=True,
        )

out_dir = MODEL_FOLDER / "goal_inference"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "goal_stability_probe_standalone.json"
out_path.write_text(json.dumps(results, indent=2))
maes = [v["final_joint_mae_at_200"] for v in results.values()]
print(f"SUMMARY: n={len(maes)} mae_mean={np.mean(maes):.4f} mae_median={np.median(maes):.4f} mae_max={np.max(maes):.4f}", flush=True)
print(f"saved: {out_path}", flush=True)
wrapped_env.close()
