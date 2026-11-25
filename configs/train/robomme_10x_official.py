"""Official RoboMME 10× training configuration.

Pass this file to bin/train.sh --cfg. Relative paths start at the repository root.
Use ${NAME} to read launcher environment variables and their portable defaults.
"""

preset = "pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default"
cuda_visible_devices = [0, 1, 2, 3, 4, 5, 6, 7]

paths = dict(
    dataset_dir="${ROBOMME_VIDEO_DATASET}",
    memory_cache_dir="${ROBOMME_MEMORY_CACHE}",
    anchor_cache_dir="${ROBOMME_ANCHOR_CACHE}",
    assets_dir="${ROBOMME_ASSETS_ROOT}",
    asset_id="${ROBOMME_ASSET_ID}",
    base_params_dir="${PI05_BASE_PARAMS}",
)

runtime = dict(
    python_bin="${PYTHON_BIN}",
    run_base_dir="${RUN_BASE_DIR}",
    tmux_session="${ROBOMME_SESSION}",
    hf_cache_dir="${HF_CACHE}",
    env=dict(
        XLA_PYTHON_CLIENT_PREALLOCATE="true",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="${WANDB_MODE}",
        # OPENPI_DATA_HOME="relative/path/to/openpi-cache",
    ),
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
    memory_horizon=1800,
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
    execution_only=True,
    memory_random_drop_execution_tail_min=10,
    memory_random_drop_execution_tail_max=10,
    action_sequence_start_offset=1,
    hold_last_action_targets=True,
)

train = dict(
    exp_name="${ROBOMME_EXP}",
    batch_size=128,
    gradient_accumulation_steps=1,
    fsdp_devices=8,
    num_train_steps=120_000,
    num_workers=32,
    prefetch_factor=2,
    data_loader_in_order=False,
    memory_bucket_size_multiplier=50,
    memory_pad_to_multiple=512,
    memory_projector_lr_multiplier=4.0,
    ema_decay=0.999,
    seed=0,
    log_interval=20,
    wandb_enabled=True,
)

lr_schedule = dict(
    warmup_steps=1_000,
    peak_lr=2.5e-5,
    decay_steps=120_000,
    decay_lr=2.5e-6,
)

optimizer = dict(clip_gradient_norm=1.0)

checkpoint = dict(
    checkpoint_base_dir="${CHECKPOINT_BASE_DIR}",
    # A nonempty save_steps list takes priority over save_interval.
    save_steps=list(range(10_000, 120_001, 10_000)),
    save_interval=1_000,
    # Preserve checkpoints at multiples of this value. Use 0 to keep only the latest.
    keep_period=5_000,
    resume=False,
    overwrite=False,
    # Optional independent restore source. New saves go under checkpoint_base_dir.
    # restore_checkpoint_dir="path/to/source/experiment",
    # restore_checkpoint_step=60_000,
)
