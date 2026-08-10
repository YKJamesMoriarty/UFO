"""Standalone ONNX export for v20 (4-GPU) model, mirroring
tracking_inference.py's _export_tracking_onnx() but without building the full
mjlab env (which is unnecessary for export and slower). Produces
FBcprAuxModel.onnx + FBcprAuxModel.meta.json + backward_encoder.onnx under
<model_folder>/exported/.

Usage: python scratch_v20_onnx_export_driver.py <model_folder_name>
"""
from __future__ import annotations
import sys
from pathlib import Path

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.export.backward_encoder import (
    UnsupportedBackwardEncoderExport,
    export_backward_encoder_from_model,
)
from humanoidverse.mjlab_inference_utils import checkpoint_load_device
from humanoidverse.tracking_inference import _export_policy_model
from humanoidverse.utils.robot_spec import load_robot_training_spec, resolve_robot_config_path

MODEL_FOLDER = Path("runs") / sys.argv[1]
ROBOT_CONFIG = Path("configs/robots/g1_29dof.yaml")
DEVICE = "cuda:0"

checkpoint_dir = MODEL_FOLDER / "checkpoint"
model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(DEVICE))
model.to(DEVICE)
model.eval()

robot_config = resolve_robot_config_path(ROBOT_CONFIG)
robot_training = load_robot_training_spec(robot_config)

output_dir = MODEL_FOLDER / "exported"
export_metadata = _export_policy_model(model, output_dir, robot_training)
print(f"[INFO] export_metadata={export_metadata}", flush=True)

try:
    export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx")
    print(f"[INFO] backward_encoder exported to {output_dir / 'backward_encoder.onnx'}", flush=True)
except UnsupportedBackwardEncoderExport as exc:
    print(f"[INFO] Skip backward encoder ONNX export: {exc}", flush=True)

print(f"DONE: exported to {output_dir}", flush=True)
