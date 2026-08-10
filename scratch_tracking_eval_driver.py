"""Standalone tracking-eval driver that bypasses the distributed
train.py/workspace.py resume path (which crashes with a CUDA illegal-memory-
access error inside broadcast_optimizer_state when resuming the v20 4-GPU
checkpoint). HumanoidVerseMjlabTrackingEvaluation.run() only needs a loaded
model + a built env -- no optimizer/distributed state at all -- so we can
call it directly, single-GPU, for both v16 and v20 for a fair, identical-
methodology comparison.

Usage: python scratch_tracking_eval_driver.py <model_folder_name>
  e.g. python scratch_tracking_eval_driver.py ufo_fb_fallgetup_4gpu_v20_eval_import
       python scratch_tracking_eval_driver.py ufo_fb_fallgetup_1gpu_v16_eval_import
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluationConfig,
)
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, load_mjlab_env_cfg

MODEL_FOLDER = Path("runs") / sys.argv[1]
DATA_PATH = Path("humanoidverse/data/derived/lafan_29dof_fallAndGetUp_only.pkl")
DEVICE = "cuda:0"
NUM_ENVS = 256

checkpoint_dir = MODEL_FOLDER / "checkpoint"
model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(DEVICE))
model.to(DEVICE)
model.eval()

train_status_path = checkpoint_dir / "train_status.json"
timestep = json.loads(train_status_path.read_text())["global_time"] if train_status_path.exists() else 0
print(f"[INFO] model_folder={MODEL_FOLDER} timestep={timestep}", flush=True)

env_cfg, use_root_height_obs = load_mjlab_env_cfg(
    MODEL_FOLDER, data_path=DATA_PATH, robot_config=Path("configs/robots/g1_29dof.yaml"), device=DEVICE,
    headless=True, disable_dr=True, disable_obs_noise=True, max_episode_length_s=10000.0,
)
wrapped_env, _ = env_cfg.build(num_envs=NUM_ENVS)

eval_cfg = HumanoidVerseMjlabTrackingEvaluationConfig(num_envs=NUM_ENVS, n_episodes_per_motion=1, disable_tqdm=False)
evaluator = eval_cfg.build()

metrics, wandb_dict = evaluator.run(timestep=timestep, agent_or_model=model, logger=None, env=wrapped_env)

out_dir = MODEL_FOLDER
out_csv = out_dir / "humanoidverse_tracking_eval_standalone.csv"
out_json = out_dir / "humanoidverse_tracking_eval_standalone.json"

rows = []
for motion_file, m in metrics.items():
    row = dict(m)
    row["motion_name"] = motion_file
    row["timestep"] = timestep
    rows.append(row)

fieldnames = sorted({k for r in rows for k in r.keys()})
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

out_json.write_text(json.dumps({"metrics": metrics, "wandb_dict": wandb_dict, "timestep": timestep}, indent=2, default=str))

print(f"[SUMMARY] {out_dir.name}: n_motions={len(rows)}", flush=True)
for k, v in sorted(wandb_dict.items()):
    if not k.endswith("#std"):
        print(f"  {k}: mean={v:.4f}", flush=True)
print(f"saved: {out_csv}", flush=True)
wrapped_env.close()
