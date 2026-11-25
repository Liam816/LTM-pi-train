"""LTM Table 1, original RoboMME, DreamDojo, eight S5000s.

Use bin/train.sh preflight --cfg configs/train/robomme_1x_musa_8gpu.py.
ROBOMME_1X_RESOURCE_ROOT selects a prepared original-data export.
"""
import os
from pathlib import Path

preset = "pi05_robomme_1x_dreamdojo_fixedlag5_h50"
cuda_visible_devices = [0, 1, 2, 3, 4, 5, 6, 7]

root = Path(os.environ.get("ROBOMME_1X_RESOURCE_ROOT", Path(__file__).resolve().parents[3] / "ltm_pi_robomme_1x"))
paths = dict(
    dataset_dir=str(root / "datasets/video"),
    memory_cache_dir=str(root / "artifacts/memory"),
    anchor_cache_dir=str(root / "artifacts/anchors"),
    assets_dir=str(root / "assets"),
    asset_id="lerobot/robomme",
    base_params_dir="${PI05_BASE_PARAMS}",
)

model = dict(
    pi05=True,
    dtype="bfloat16",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    action_horizon=50,
    discrete_state_input=False,
    max_token_len=200,
    active_image_keys=["base_0_rgb", "left_wrist_0_rgb"],
    memory_horizon=1410,
    memory_latent_dim=64,
    memory_stride=1,
    memory_frames_per_token=1,
    memory_pooling_mode="concatenate",
    memory_use_vlm_expert=True,
    memory_prefix_order="language_memory_vision",
    language_causal_attention=True,
    memory_causal_attention=True,
    memory_projector_hidden_dim=512,
    memory_token_dropout_rate=0.0,
    memory_segment_embedding=True,
    memory_segment_embedding_after_projection=True,
    memory_segment_vocab_size=6,
    memory_demo_anchor=True,
    memory_execution_anchor=True,
    memory_demo_anchor_type_id=4,
    memory_execution_anchor_type_id=5,
    memory_demo_lam_boundary_token=True,
    memory_demo_lam_boundary_type_id=2,
    # The launcher reads memory_latent_mean and memory_latent_std from the selected cache manifest.
)

data = dict(
    prompt_from_task=True,
    lowercase_task_prompts=True,
    execution_only=True,
    memory_random_drop_execution_tail_min=5,
    memory_random_drop_execution_tail_max=5,
    action_sequence_start_offset=1,
    hold_last_action_targets=True,
)

train = dict(
    exp_name="ltm_pi_robomme_1x_dreamdojo_lag5_musa_8gpu",
    batch_size=128,
    gradient_accumulation_steps=1,
    fsdp_devices=8,
    num_train_steps=40_000,
    num_workers=2,
    prefetch_factor=2,
    data_loader_in_order=True,
    pin_memory=False,
    memory_bucket_size_multiplier=50,
    memory_pad_to_multiple=64,
    memory_projector_lr_multiplier=4.0,
    ema_decay=0.999,
    seed=0,
    log_interval=20,
    wandb_enabled=True,
)

lr_schedule = dict(
    warmup_steps=1_000,
    peak_lr=2.5e-5,
    decay_steps=40_000,
    decay_lr=2.5e-6,
)

optimizer = dict(clip_gradient_norm=1.0)

checkpoint = dict(
    checkpoint_base_dir=str(root / "checkpoints"),
    save_steps=[10_000, 20_000, 30_000, 40_000],
    save_interval=1_000,
    keep_period=10_000,
    resume=False,
    overwrite=False,
)

runtime = dict(
    device_backend="musa",
    python_bin=os.environ.get("MUSA_PYTHON", "${PYTHON_BIN}"),
    run_base_dir=str(root / "runs"),
    hf_cache_dir="${HF_CACHE}",
    tmux_session="ltm-pi-robomme-1x-musa-8gpu",
    activation_checkpointing=True,
    # Zero-based joint blocks. The other blocks retain their forward activations.
    activation_checkpoint_blocks=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17],
    sample_effective_batch=True,
    defer_gradient_sync=False,
    env=dict(
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", WANDB_MODE="${WANDB_MODE}", WANDB_DIR=str(root / "runs"),
    ),
)

smoke = dict(steps=2, resource_root=os.environ.get("ROBOMME_1X_SMOKE_RESOURCE_ROOT"))
