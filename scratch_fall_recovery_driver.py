"""Fall-recovery capability test: reset to a genuinely FALLEN pose taken from
various fallAndGetUp training clips, condition on a SINGLE fixed STANDING
target (one clip's settled end-of-motion frame), and check whether the policy
rises to that standing height and holds it stably.
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

REACH_STEPS = 400
HOLD_STEPS = 300
STAND_HEIGHT_THRESH = 0.55
STABLE_HEIGHT_THRESH = 0.45
CONTACT_FORCE_THRESHOLD = 1.0

TARGET_CLIP = 39  # fallAndGetUp1_subject5_clip7: settles to h~0.786, std~0.0006 (cleanest standing plateau)
TARGET_FRAME = 498

TEST_CASES = [
    ("subject4_clip2_f0", 2, 0),
    ("subject4_clip6_f321_deepfall", 6, 321),
    ("subject3_clip1_f0", 17, 0),
    ("subject5_clip7_f0_selfcheck", 39, 0),
    ("subject1_clip7_f0", 71, 0),
    ("subject2_clip1_f5", 49, 5),
    ("subject2_clip4_f5_deepfall", 52, 5),
    ("subject1seq3_clip9_f22", 89, 22),
]

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


def target_states_at_frame(obs_dict, frame_idx, device, num_dof):
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][frame_idx, 0],
            obs_dict["ref_body_rots"][frame_idx, 0],
            obs_dict["ref_body_vels"][frame_idx, 0],
            obs_dict["ref_body_angular_vels"][frame_idx, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][frame_idx].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][frame_idx].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}


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

    results = {"target": {"clip_id": TARGET_CLIP, "frame": TARGET_FRAME, "height": target_h}, "cases": {}}

    for name, clip_id, frame_idx in TEST_CASES:
        env.set_is_evaluating(clip_id)
        backward_obs_s, obs_dict_s = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
        )
        start_h = float(obs_dict_s["ref_body_pos"][frame_idx, 0, 2].item())
        target_states = target_states_at_frame(obs_dict_s, frame_idx, DEVICE, num_dof)
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
        reach_h = height_trace[:REACH_STEPS]
        hold_contact = contact_trace[REACH_STEPS:]

        entry = {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "start_height": start_h,
            "reach_step": reach_step,
            "reach_time_s": reach_step * env.dt if reach_step is not None else None,
            "reach_success": reach_step is not None,
            "reach_phase_max_height": float(np.max(reach_h)),
            "hold_phase_mean_height": float(np.mean(hold_h)),
            "hold_phase_min_height": float(np.min(hold_h)),
            "hold_phase_final_height": hold_h[-1],
            "hold_phase_contact_frac": float(np.mean([1 if c > CONTACT_FORCE_THRESHOLD else 0 for c in hold_contact])),
            "final_joint_mae": final_joint_mae,
            "held_stable": bool(np.min(hold_h) > STABLE_HEIGHT_THRESH) if reach_step is not None else False,
        }
        results["cases"][name] = entry
        print(f"[CASE] {name}: {entry}", flush=True)

out_path = MODEL_FOLDER / "goal_inference" / "fall_recovery_test.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"saved: {out_path}", flush=True)
wrapped_env.close()
