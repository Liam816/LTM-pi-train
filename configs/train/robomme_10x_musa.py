"""Run with bin/train.sh smoke --cfg configs/train/robomme_10x_musa.py on S5000."""

import os
from pathlib import Path
import runpy

official = runpy.run_path(str(Path(__file__).with_name("robomme_10x_official.py")))
preset = official["preset"]
cuda_visible_devices = [4, 5, 6, 7]

paths = dict(official["paths"])
model = dict(official["model"])
data = dict(official["data"])
optimizer = dict(official["optimizer"])
lr_schedule = dict(official["lr_schedule"])
checkpoint = dict(official["checkpoint"])
runtime = dict(
    official["runtime"],
    device_backend="musa",
    python_bin=os.environ.get("MUSA_PYTHON", "${PYTHON_BIN}"),
    tmux_session="ltm-pi-robomme-10x-musa",
    activation_checkpointing=True,
    env=dict(
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="${WANDB_MODE}",
    ),
)
# One example per card per forward pass; 32 microbatches give effective batch 128.
train = dict(
    official["train"], exp_name="ltm_pi_robomme_fixedlag10_musa", fsdp_devices=4, gradient_accumulation_steps=32
)
# A separate resource root may contain a small, aligned slice for hardware testing.
smoke = dict(steps=2, resource_root=os.environ.get("ROBOMME_SMOKE_RESOURCE_ROOT"))
