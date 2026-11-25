"""MUSA worker for the Python CFG launcher. Each process owns one logical S5000."""

from contextlib import nullcontext
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import jax
import numpy as np
import torch
import torch_musa  # Register the native accelerator and MCCL before using FSDP.
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import CustomPolicy

from openpi.models.model import restore_params
from openpi.models_pytorch.ltm_pi import JointBlock, LTMPolicy, prepare_flow_inputs
from openpi.models_pytorch.ltm_weights import export_inference, load_base
from openpi.training import data_loader
from train_cfg import _build_train_config


def log(**values):
    print(json.dumps(values, sort_keys=True, allow_nan=False), flush=True)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def global_sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def checkpoints(directory):
    return sorted(
        (p for p in directory.glob("[0-9]*") if p.name.isdigit() and (p / "musa.json").is_file()),
        key=lambda p: int(p.name),
    )


def local_params(model):
    return {name.replace("_fsdp_wrapped_module.", ""): p for name, p in model.named_parameters()}


def save_step(root, step_id, completed, rank, world, params, ema, optimizer, cfg, plan, shapes):
    target = root / str(step_id)
    temporary = root / (".saving-" + str(step_id))
    if rank == 0:
        if target.exists():
            raise FileExistsError(target)
        # A previous interrupted write is never selected for restoration.
        if temporary.exists():
            temporary.rename(root / f".interrupted-{step_id}-{time.time_ns()}")
        temporary.mkdir()
    barrier()
    state = dict(
        params={k: p.detach().cpu() for k, p in params.items()},
        ema=None if ema is None else {k: p.cpu() for k, p in ema.items()},
        optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),
        musa_rng=torch.musa.get_rng_state(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )
    torch.save(state, temporary / f"rank-{rank}.pt")
    del state
    barrier()
    if rank == 0:
        export_inference(temporary, world, shapes, cfg)
        asset = Path(plan["paths"]["asset_id"]) / "norm_stats.json"
        destination = temporary / "assets" / asset
        destination.parent.mkdir(parents=True)
        shutil.copyfile(Path(plan["paths"]["assets_dir"]) / asset, destination)
        (temporary / "musa.json").write_text(
            json.dumps(
                dict(
                    format_version=1,
                    step=completed,
                    world=world,
                    preset=cfg.name,
                    batch_size=cfg.batch_size,
                    accumulation=cfg.gradient_accumulation_steps,
                    sample_effective_batch=plan.get("sample_effective_batch", False),
                    model=plan["model"],
                    data=plan["data"],
                    paths=plan["paths"],
                ),
                indent=2,
            )
        )
        temporary.rename(target)
        for prior in checkpoints(root):
            if prior != target and (not cfg.keep_period or int(prior.name) % cfg.keep_period):
                shutil.rmtree(prior)
        log(checkpoint=str(target), completed_steps=completed, ranks=world)
    barrier()
    return target


@torch.no_grad()
def restore_step(path, rank, world, params, ema, optimizer, cfg, plan):
    meta = json.loads((path / "musa.json").read_text())
    if (
        meta["format_version"],
        meta["world"],
        meta["preset"],
        meta["batch_size"],
        meta["accumulation"],
        meta["model"],
        meta["data"],
    ) != (1, world, cfg.name, cfg.batch_size, cfg.gradient_accumulation_steps, plan["model"], plan["data"]):
        raise ValueError("Checkpoint architecture, training batch, data options, or MUSA shard count differ")
    state = torch.load(path / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
    if meta.get("sample_effective_batch", False) != plan.get("sample_effective_batch", False):
        raise ValueError("Checkpoint and current run use different sampling batch sizes")
    if params.keys() != state["params"].keys() or (ema is None) != (state["ema"] is None):
        raise ValueError("Checkpoint parameters or EMA configuration differ")
    for name, param in params.items():
        value = state["params"][name]
        if value.shape != param.shape:
            raise ValueError(f"Checkpoint shard shape mismatch for {name}")
        param.copy_(value)
        if ema is not None:
            ema[name].copy_(state["ema"][name])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    torch.musa.set_rng_state(state["musa_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return meta["step"]


def microbatches(loader, consumed, local_size, device, image_keys, measure):
    """Split an optimizer batch after its sample selection, augmentation and flow RNG draws."""
    loader.set_epoch(consumed // len(loader), batch_offset=consumed % len(loader))
    stream = iter(loader)
    while True:
        tick = time.perf_counter() if measure else 0.0
        obs, actions = next(stream)
        read_seconds = time.perf_counter() - tick if measure else 0.0
        host_mask = obs.memory_mask
        obs = jax.tree.map(lambda t: t.to(device, non_blocking=True), obs)
        actions = actions.to(device, dtype=torch.float32, non_blocking=True)
        if len(actions) == local_size:
            yield (obs, actions, None), host_mask, read_seconds
        else:
            # Image jitter is batch-shared. Drawing it (and flow time/noise) once
            # keeps the same training examples and targets as an unsplit forward.
            obs, flow = prepare_flow_inputs(obs, actions, image_keys, train=True)
            for begin in range(0, len(actions), local_size):
                end = begin + local_size
                values = jax.tree.map(lambda t: t[begin:end], (obs, actions, flow))
                yield values, host_mask[begin:end], read_seconds if begin == 0 else 0.0
        consumed += 1
        if consumed % len(loader) == 0:
            loader.set_epoch(consumed // len(loader))
            stream = iter(loader)


def train(plan, rank, world, device):
    cfg = _build_train_config(plan)
    smoke = plan["smoke"]
    bench = plan.get("benchmark")
    if bench and (smoke or cfg.resume or cfg.restore_checkpoint_dir or not 1 <= cfg.num_train_steps <= 24):
        raise ValueError("Throughput benchmarking requires a fresh run of 1 to 24 complete updates")
    if smoke and not 1 <= cfg.num_train_steps <= 5:
        raise ValueError("MUSA smoke must be limited to at most five updates")
    if cfg.batch_size % (world * cfg.gradient_accumulation_steps):
        raise ValueError("Effective batch must be divisible by devices times gradient accumulation")
    root = cfg.checkpoint_dir
    lock = None
    if rank == 0:
        root.parent.mkdir(parents=True, exist_ok=True)
        lock = (root.parent / f".{root.name}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg.overwrite and root.exists():
            root.rename(root.with_name(f"{root.name}.previous-{time.time_ns()}"))
        if root.exists() and any(root.iterdir()) and not cfg.resume:
            raise FileExistsError(f"Checkpoint directory is in use: {root}")
        root.mkdir(exist_ok=True)
        (root / "config.json").write_text(json.dumps(plan, indent=2))
    barrier()
    source = None
    if cfg.resume:
        found = checkpoints(root)
        if not found:
            raise ValueError(f"No completed MUSA checkpoint in {root}")
        source = found[-1]
    elif cfg.restore_checkpoint_dir:
        source = Path(cfg.restore_checkpoint_dir) / str(cfg.restore_checkpoint_step)
    torch.manual_seed(cfg.seed)
    model = LTMPolicy(cfg.model, activation_checkpointing=plan["activation_checkpointing"],
                      activation_checkpoint_blocks=plan.get("activation_checkpoint_blocks"))
    shapes = {name: p.shape for name, p in model.named_parameters()}
    if source is None and rank == 0:
        load_base(model, plan["paths"]["base_params_dir"], cfg.seed)
    if world > 1:
        mp = MixedPrecision(param_dtype=model.compute_dtype, reduce_dtype=torch.float32)
        vision_layer = type(model.vision.encoder.layers[0])
        token_embedding = model.tokens

        def wrap(layer):
            if isinstance(layer, (JointBlock, vision_layer)) or layer is token_embedding:
                return {"mixed_precision": mp}
            return False

        model = FSDP(
            model,
            device_id=device,
            auto_wrap_policy=CustomPolicy(wrap),
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
        )
    else:
        model.to(device)
    params = local_params(model)
    memory = [p for name, p in params.items() if name.startswith("memory_")]
    base = [p for name, p in params.items() if not name.startswith("memory_")]
    opt = cfg.optimizer
    optimizer = torch.optim.AdamW(
        [dict(params=base), dict(params=memory)],
        lr=cfg.lr_schedule.peak_lr,
        betas=(opt.b1, opt.b2),
        eps=opt.eps,
        weight_decay=opt.weight_decay,
        fused=False,
        foreach=False,
    )
    ema = None if cfg.ema_decay is None else {k: p.detach().clone() for k, p in params.items()}
    physical_batch = cfg.batch_size // cfg.gradient_accumulation_steps
    grouped = plan.get("sample_effective_batch", False)
    loader_batch = cfg.batch_size if grouped else physical_batch
    loader = data_loader.create_data_loader(
        dataclasses.replace(cfg, batch_size=loader_batch), shuffle=True, framework="pytorch"
    )
    torch.manual_seed(cfg.seed + rank)
    torch.musa.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    start = 0 if source is None else restore_step(source, rank, world, params, ema, optimizer, cfg, plan)
    consumed = start * (cfg.batch_size // loader_batch)
    if bench:
        consumed += bench["batch_offset"]
    if bench and bench["batch_offset"] >= len(loader):
        raise ValueError("benchmark batch offset is outside epoch zero")
    stream = microbatches(loader, consumed, physical_batch // world, device, cfg.model.active_image_keys, bool(bench))
    schedule = cfg.lr_schedule.create()
    wandb_run = None
    if rank == 0:
        log(
            device=str(device),
            accelerator=torch.musa.get_device_name(device),
            torch=str(torch.__version__),
            torch_musa=str(getattr(torch_musa, "__version__", "unknown")),
            backend="musa",
            collective="mccl" if world > 1 else None,
            visible=os.environ["MUSA_VISIBLE_DEVICES"],
            world=world,
            batch=cfg.batch_size,
            local_microbatch=physical_batch // world,
            steps=cfg.num_train_steps,
            smoke=smoke,
            checkpoint_root=str(root),
            base_weights=str(plan["paths"]["base_params_dir"]),
        )
        if cfg.wandb_enabled:
            import wandb

            wandb_run = wandb.init(project=cfg.project_name, name=cfg.exp_name, config=plan)
    events = []
    try:
        for index in range(start, cfg.num_train_steps):
            if bench:
                torch.musa.synchronize(device)
                barrier()
                torch.musa.reset_peak_memory_stats(device)
                tick = time.perf_counter()
                local_tokens = local_max = local_pad = local_seen = 0
                load_seconds = 0.0
            lr = float(schedule(index))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg.memory_projector_lr_multiplier
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for micro in range(cfg.gradient_accumulation_steps):
                model_args, host_mask, read_seconds = next(stream)
                if bench:
                    load_seconds += read_seconds
                    lengths = host_mask.sum(1)
                    local_tokens += int(lengths.sum())
                    local_max = max(local_max, int(lengths.max()))
                    local_pad = max(local_pad, host_mask.shape[1])
                    local_seen += host_mask.shape[0]
                defer = plan.get("defer_gradient_sync", False) and world > 1 and micro + 1 < cfg.gradient_accumulation_steps
                with model.no_sync() if defer else nullcontext():
                    loss = model(*model_args) / cfg.gradient_accumulation_steps
                    loss.backward()
                loss_sum += loss.detach()
            norm = (
                model.clip_grad_norm_(opt.clip_gradient_norm)
                if world > 1
                else torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_gradient_norm)
            )
            avg_loss = global_sum(loss_sum) / world
            if not torch.isfinite(torch.stack((avg_loss, norm.float()))).all().item():
                raise FloatingPointError(f"Nonfinite loss or gradient at update {index+1}")
            if smoke:
                probes = [
                    p for k, p in params.items() if k in ("memory_projector_out.weight", "action_out_proj.weight")
                ]
                before = [p.detach().clone() for p in probes]
                memory_grad = sum(
                    (p.grad.detach().float().square().sum() for p in memory if p.grad is not None),
                    start=torch.zeros((), device=device),
                )
                memory_norm = global_sum(memory_grad).sqrt().item()
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for k, p in params.items():
                        ema[k].lerp_(p.detach(), 1 - cfg.ema_decay)
            metric = dict(step=index + 1, loss=avg_loss.item(), grad_norm=norm.item(), lr=lr)
            if bench:
                torch.musa.synchronize(device)
                seconds = time.perf_counter() - tick
                maximum = torch.tensor([seconds, load_seconds, torch.musa.max_memory_allocated(device) / 2**30,
                                        torch.musa.max_memory_reserved(device) / 2**30, local_max, local_pad],
                                       device=device, dtype=torch.float32)
                totals = torch.tensor([local_tokens, local_seen], device=device, dtype=torch.int64)
                if world > 1:
                    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                t, read, allocated, reserved, max_len, padded = maximum.tolist()
                token_count, samples = global_sum(totals).tolist()
                if samples != cfg.batch_size:
                    raise ValueError(f"Expected {cfg.batch_size} samples in update, got {samples}")
                metric.update(seconds=t, samples=samples, loader_s=read, allocated_gib=allocated, reserved_gib=reserved,
                              mean_history=token_count / samples, max_history=int(max_len), padded=int(padded))
            if smoke:
                changes = [
                    global_sum((p.detach() - b).float().abs().sum()).item() for p, b in zip(probes, before, strict=True)
                ]
                if not (
                    memory_norm > 0 and math.isfinite(memory_norm) and all(math.isfinite(v) and v > 0 for v in changes)
                ):
                    raise RuntimeError(f"Gradient/update verification failed: {memory_norm}, {changes}")
                torch.musa.synchronize(device)
                metric.update(
                    memory_grad_norm=memory_norm,
                    verified_parameter_delta_l1=changes,
                    valid_history_tokens=host_mask.sum().item(),
                    action_shape=list(model_args[1].shape),
                )
                events.append(metric)
            if rank == 0 and (index % cfg.log_interval == 0 or index + 1 == cfg.num_train_steps):
                log(**metric)
                if wandb_run is not None:
                    wandb_run.log(metric, step=index + 1)
            step_id = index + 1 if cfg.save_steps else index
            save = (
                step_id in cfg.save_steps
                if cfg.save_steps
                else ((index > start and index % cfg.save_interval == 0) or index + 1 == cfg.num_train_steps)
            )
            if save and not bench:
                target = save_step(root, step_id, index + 1, rank, world, params, ema, optimizer, cfg, plan, shapes)
                if smoke and index + 1 == cfg.num_train_steps:
                    restored = restore_step(target, rank, world, params, ema, optimizer, cfg, plan)
                    if restored != index + 1:
                        raise ValueError("Saved step did not round trip")
                    barrier()
                    if rank == 0:
                        # Exercise exactly the parameter reader and model tree used by the JAX policy.
                        native = restore_params(target / "params", restore_type=np.ndarray)
                        cfg.model.load(native, remove_extra_params=False)
                        result = dict(
                            status="passed",
                            backend="musa",
                            visible=os.environ["MUSA_VISIBLE_DEVICES"],
                            accelerator=torch.musa.get_device_name(device),
                            world=world,
                            completed_steps=restored,
                            restored_checkpoint=str(target),
                            native_inference_weights_verified=True,
                            updates=events,
                        )
                        (Path(cfg.checkpoint_base_dir) / "smoke_result.json").write_text(json.dumps(result, indent=2))
                        log(**result)
    finally:
        loader.close()
        if wandb_run is not None:
            wandb_run.finish()
        if lock is not None:
            lock.close()


def main():
    plan = json.loads(sys.argv[1])
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.musa.is_available() or torch.musa.device_count() != world:
        raise RuntimeError("MUSA visibility must match the native worker count")
    device = torch.device("musa", local_rank)
    torch.musa.set_device(device)
    torch.backends.mudnn.allow_tf32 = False
    torch.backends.musa.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group("mccl", device_id=device)
    try:
        train(plan, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
