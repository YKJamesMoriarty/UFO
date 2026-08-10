"""Inspect root-height trace (frame0/min/max/last) for every fallAndGetUp
training clip, to identify genuine fallen-start / standing-end clips for the
fall-recovery capability test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg
from humanoidverse.utils.helpers import get_backward_observation

MODEL_FOLDER = Path("runs/ufo_fb_fallgetup_1gpu_v16_eval_import")
DATA_PATH = Path("humanoidverse/data/derived/lafan_29dof_fallAndGetUp_only.pkl")
DEVICE = "cuda:0"

env_cfg, use_root_height_obs = load_mjlab_env_cfg(
    MODEL_FOLDER, data_path=DATA_PATH, robot_config=None, device=DEVICE,
    headless=True, disable_dr=True, disable_obs_noise=True, max_episode_length_s=10000.0,
)
wrapped_env, _ = env_cfg.build(num_envs=1)
env = wrapped_env._env

NUM_MOTIONS = 90  # known from pkl inspection; set_is_evaluating reloads to exactly 1 motion each call

results = {}
with torch.no_grad():
    for motion_id in range(NUM_MOTIONS):
        env.set_is_evaluating(motion_id)
        backward_obs, obs_dict = get_backward_observation(
            env, 0, use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
        )
        root_h = obs_dict["ref_body_pos"][:, 0, 2].detach().cpu().numpy()
        name = None
        try:
            name = env._motion_lib.curr_motion_keys[0]
        except Exception:
            name = None
        argmin = int(root_h.argmin())
        entry = {
            "name": name,
            "num_frames": int(root_h.shape[0]),
            "h_first": float(root_h[0]),
            "h_min": float(root_h.min()),
            "argmin_frame": argmin,
            "h_max": float(root_h.max()),
            "h_last": float(root_h[-1]),
            "h_last10_mean": float(root_h[-10:].mean()),
            "h_last10_std": float(root_h[-10:].std()),
        }
        results[str(motion_id)] = entry
        print(f"[{motion_id}] {entry}", flush=True)

out_path = MODEL_FOLDER / "goal_inference" / "height_profile_fallgetup90.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"saved: {out_path}", flush=True)
wrapped_env.close()
