"""Launch an existing OpenPI training preset with overrides from a Python configuration."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
from string import Template
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/train/robomme_10x_official.py"
_CONFIG_FIELDS = (
    "preset", "cuda_visible_devices", "runtime", "paths", "train", "checkpoint",
    "model", "data", "optimizer", "lr_schedule", "smoke",
)
_RUN_WITH_LOG = """
set -euo pipefail
mkdir -p -- "$(dirname -- "$1")"
ulimit -n 65536
exec > >(tee -a -- "$1") 2>&1
shift
exec "$@"
"""


def _repo_path(value: str) -> Path:
    return ROOT / Path(value).expanduser()


def _variables() -> dict[str, str]:
    resource = _repo_path(
        os.environ.get("ROBOMME_RESOURCE_ROOT") or str(ROOT.parent / "ltm_pi_robomme-resources")
    )
    defaults = {
        "ROBOMME_RESOURCE_ROOT": str(resource),
        "ROBOMME_VIDEO_DATASET": str(resource / "datasets/video"),
        "ROBOMME_MEMORY_CACHE": str(resource / "artifacts/memory"),
        "ROBOMME_ANCHOR_CACHE": str(resource / "artifacts/anchors"),
        "ROBOMME_ASSETS_ROOT": str(resource / "assets"),
        "ROBOMME_ASSET_ID": "lerobot/robomme_16x1000",
        "PI05_BASE_PARAMS": str(resource / "models/pi05/params"),
        "HF_CACHE": str(ROOT / ".cache/huggingface"),
        "PYTHON_BIN": "python3",
        "RUN_BASE_DIR": str(ROOT / "runs"),
        "CHECKPOINT_BASE_DIR": str(ROOT / "checkpoints"),
        "ROBOMME_EXP": "ltm_pi_robomme_fixedlag10_v1",
        "ROBOMME_SESSION": "ltm-pi-robomme-video-dreamdojo-fixedlag10-h50-b128-120k-v1",
        "WANDB_MODE": "offline",
    }
    return defaults | {key: value for key, value in os.environ.items() if value} | {"REPO_ROOT": str(ROOT)}


def _expand(value, variables):
    if isinstance(value, str):
        return Template(value).substitute(variables)
    if isinstance(value, (list, tuple)):
        return [_expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    return value


def _section(config: dict, name: str, allowed: set[str]) -> dict:
    section = config[name]
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = section.keys() - allowed
    if unknown:
        raise ValueError(f"[{name}] has unknown keys: {', '.join(sorted(unknown))}")
    return section


def _gpu_list(value) -> str:
    if not isinstance(value, list) or not value or any(type(device) is not int or device < 0 for device in value):
        raise ValueError("cuda_visible_devices must be a nonempty list of nonnegative GPU indices")
    if len(set(value)) != len(value):
        raise ValueError("cuda_visible_devices contains duplicate indices")
    return ",".join(map(str, value))


def _read_cfg(cfg_path: Path, *, implicit: bool, smoke: bool = False) -> dict:
    if cfg_path.suffix != ".py":
        raise ValueError(f"--cfg expects a Python (.py) file: {cfg_path}")
    namespace = runpy.run_path(str(cfg_path))
    config = _expand({key: namespace[key] for key in _CONFIG_FIELDS if key in namespace}, _variables())

    runtime = _section(config, "runtime", {
        "python_bin", "run_base_dir", "tmux_session", "hf_cache_dir", "env", "device_backend", "activation_checkpointing",
        "sample_effective_batch", "defer_gradient_sync", "activation_checkpoint_blocks",
    })
    backend = runtime.get("device_backend", "jax")
    if backend not in ("jax", "musa"):
        raise ValueError("runtime.device_backend must be jax or musa")
    for key in ("activation_checkpointing", "sample_effective_batch", "defer_gradient_sync"):
        if key in runtime and type(runtime[key]) is not bool:
            raise ValueError(f"runtime.{key} must be a boolean")
    blocks = runtime.get("activation_checkpoint_blocks")
    if blocks is not None:
        if (not isinstance(blocks, list) or any(type(i) is not int or i < 0 for i in blocks)
                or len(set(blocks)) != len(blocks) or not runtime.get("activation_checkpointing", True)):
            raise ValueError("runtime.activation_checkpoint_blocks requires checkpointing and unique nonnegative indices")
    if smoke:
        if backend != "musa":
            raise ValueError("The bounded smoke action requires a MUSA configuration")
        options = _section(config, "smoke", {"steps", "resource_root"})
        steps = options.get("steps", 2)
        if type(steps) is not int or not 1 <= steps <= 5:
            raise ValueError("smoke.steps must be between 1 and 5")
        world = len(config["cuda_visible_devices"])
        config["train"].update(num_train_steps=steps, batch_size=world, fsdp_devices=world,
                               gradient_accumulation_steps=1, num_workers=0, log_interval=1, wandb_enabled=False)
        config["checkpoint"] = dict(checkpoint_base_dir=runtime["run_base_dir"], save_steps=[steps],
                                    save_interval=steps, keep_period=0, resume=False, overwrite=False)
        if options.get("resource_root"):
            root = _repo_path(options["resource_root"])
            config["paths"].update(dataset_dir=str(root / "datasets/video"), memory_cache_dir=str(root / "artifacts/memory"),
                                   anchor_cache_dir=str(root / "artifacts/anchors"), assets_dir=str(root / "assets"))
    path_keys = ("dataset_dir", "memory_cache_dir", "anchor_cache_dir", "assets_dir", "base_params_dir")
    resource = _section(config, "paths", {*path_keys, "asset_id"})
    paths = {key: str(_repo_path(resource[key])) for key in path_keys}
    asset_id = Path(resource["asset_id"])
    if asset_id.is_absolute() or not asset_id.parts or ".." in asset_id.parts:
        raise ValueError("paths.asset_id must be a relative asset id under paths.assets_dir")
    paths["asset_id"] = str(asset_id)

    checkpoint = _section(config, "checkpoint", {
        "checkpoint_base_dir", "save_steps", "save_interval", "keep_period", "resume", "overwrite",
        "restore_checkpoint_dir", "restore_checkpoint_step",
    })
    for key in ("checkpoint_base_dir", "restore_checkpoint_dir"):
        if key in checkpoint:
            checkpoint[key] = str(_repo_path(checkpoint[key]))
    if "keep_period" in checkpoint:
        keep_period = checkpoint["keep_period"]
        if type(keep_period) is not int or keep_period < 0:
            raise ValueError("checkpoint.keep_period must be a nonnegative integer")
        if keep_period == 0:
            checkpoint["keep_period"] = None

    train = config["train"]
    overlap = train.keys() & checkpoint.keys()
    if overlap:
        raise ValueError(f"Duplicate training options in [train] and [checkpoint]: {', '.join(sorted(overlap))}")
    train = train | checkpoint

    python_bin = runtime["python_bin"]
    if "/" in python_bin:
        python_bin = str(_repo_path(python_bin))
    gpu_ids = _gpu_list(config["cuda_visible_devices"])
    if implicit:
        # Preserve CUDA_VISIBLE_DEVICES overrides for the original entry syntax.
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES") or gpu_ids
    else:
        fsdp_devices = train.get("fsdp_devices")
        if fsdp_devices is not None and (type(fsdp_devices) is not int or not 0 < fsdp_devices <= len(gpu_ids.split(","))):
            raise ValueError("train.fsdp_devices must be positive and no greater than the number of visible GPUs")

    extra_env = runtime.get("env", {})
    if any(not isinstance(value, str) for value in extra_env.values()):
        raise ValueError("[runtime.env] values must be strings")
    environment = extra_env | {
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "HF_HOME": str(_repo_path(runtime["hf_cache_dir"])),
        "PYTHONPATH": str(ROOT / "src"),
    }
    if backend == "musa":
        if train.get("fsdp_devices") != len(gpu_ids.split(",")):
            raise ValueError("On MUSA train.fsdp_devices must match cuda_visible_devices")
        environment.update(MUSA_VISIBLE_DEVICES=gpu_ids, JAX_PLATFORMS="cpu")
    return {
        "device_backend": backend, "activation_checkpointing": runtime.get("activation_checkpointing", True),
        "smoke": smoke,
        "activation_checkpoint_blocks": blocks,
        "sample_effective_batch": runtime.get("sample_effective_batch", False),
        "defer_gradient_sync": runtime.get("defer_gradient_sync", False),
        "preset": config["preset"],
        "paths": paths,
        "train": train,
        "model": config.get("model", {}),
        "data": config.get("data", {}),
        "optimizer": config.get("optimizer", {}),
        "lr_schedule": config.get("lr_schedule", {}),
        "python": python_bin,
        "session": runtime["tmux_session"],
        "log": str(_repo_path(runtime["run_base_dir"]) / train["exp_name"] / "train.log"),
        "environment": environment,
    }


def _check_original_data(info_path, memory, anchors, normalizer, *, bounded):
    info = json.loads(info_path.read_text())
    full = (info["total_episodes"], info["total_frames"]) == (1600, 768897)
    source = info_path.with_name("source.json")
    if not full and (not bounded or not source.is_file()):
        raise ValueError("The Table 1 preset requires the original 1,600-episode RoboMME training split")
    if (memory.get("checkpoint_sha256") != "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
            or memory.get("cached_valid_transitions") != info["total_frames"] - info["total_episodes"]):
        raise ValueError("Expected a complete DreamDojo LAM 400k cache for this original-data export")
    hashes = {name: hashlib.sha256(file.read_bytes()).hexdigest() for name, file in
              (("dataset_info_sha256", info_path), ("dataset_episodes_sha256", info_path.with_name("episodes.jsonl")))}
    for manifest in (memory, anchors):
        if any(manifest.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Memory and anchor caches must match the selected RoboMME dataset metadata")
    stats = json.loads(normalizer.with_name("manifest.json").read_text())
    if (stats.get("total_frames"), stats.get("execution_frames"), stats.get("action_horizon")) != (768897, 476857, 50):
        raise ValueError("Expected state/action normalization over the original RoboMME execution queries")
    expected_info = hashes["dataset_info_sha256"] if full else json.loads(source.read_text())["dataset_info_sha256"]
    if (stats.get("dataset_info_sha256") != expected_info
            or (full and stats.get("dataset_episodes_sha256") != hashes["dataset_episodes_sha256"])):
        raise ValueError("The original-data normalization manifest does not match the source dataset")


def _preflight(plan: dict, *, bounded: bool = False) -> dict:
    if shutil.which(plan["python"]) is None:
        raise ValueError(f"Python not found: {plan['python']}")
    paths = plan["paths"]
    required_dirs = ("dataset_dir", "base_params_dir")
    for key in required_dirs:
        if not Path(paths[key]).is_dir():
            raise ValueError(f"paths.{key} does not exist: {paths[key]}")
    dataset_info = Path(paths["dataset_dir"]) / "meta/info.json"
    memory_manifest = Path(paths["memory_cache_dir"]) / "manifest.json"
    anchor_manifest = Path(paths["anchor_cache_dir"]) / "manifest.json"
    normalizer = Path(paths["assets_dir"]) / paths["asset_id"] / "norm_stats.json"
    for path in (dataset_info, memory_manifest, anchor_manifest, normalizer):
        if not path.is_file():
            raise ValueError(f"Required input file not found: {path}")
    memory = json.loads(memory_manifest.read_text(encoding="utf-8"))
    if plan["preset"] == "pi05_robomme_1x_dreamdojo_fixedlag5_h50":
        _check_original_data(dataset_info, memory, json.loads(anchor_manifest.read_text()), normalizer, bounded=bounded)
    stats = memory["latent_normalization"]
    dimension = plan["model"].get("memory_latent_dim", len(stats["mean"]))
    for key in ("mean", "std"):
        values = stats[key]
        if (not values or len(values) != dimension or not all(math.isfinite(value) for value in values)
                or (key == "std" and any(value <= 0 for value in values))):
            raise ValueError(f"Invalid latent_normalization.{key} in {memory_manifest}")
    return {"memory_latent_mean": stats["mean"], "memory_latent_std": stats["std"]}


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _build_train_config(plan: dict):
    from openpi.training import config as native

    default = native.get_config(plan["preset"])
    paths = plan["paths"]
    model = dataclasses.replace(default.model, **_tuples(plan["model"]))
    assets = dataclasses.replace(default.data.assets, assets_dir=paths["assets_dir"], asset_id=paths["asset_id"])
    data_options = plan["data"] | {
        "memory_cache_dir": paths["memory_cache_dir"],
        "demo_anchor_cache_dir": paths["anchor_cache_dir"],
    }
    base_config = dataclasses.replace(default.data.base_config or native.DataConfig(), **_tuples(data_options))
    data = dataclasses.replace(default.data, repo_id=paths["dataset_dir"], assets=assets, base_config=base_config)
    return dataclasses.replace(
        default,
        **_tuples(plan["train"]),
        model=model,
        data=data,
        weight_loader=dataclasses.replace(default.weight_loader, params_path=paths["base_params_dir"]),
        lr_schedule=dataclasses.replace(default.lr_schedule, **_tuples(plan["lr_schedule"])),
        optimizer=dataclasses.replace(default.optimizer, **_tuples(plan["optimizer"])),
    )


def _launch(plan: dict, cfg_path: Path | None) -> None:
    _preflight(plan)
    command = [str(ROOT / "bin/train.sh"), "run"]
    if cfg_path is not None:
        command.extend(("--cfg", str(cfg_path)))
    subprocess.run(["tmux", "new-session", "-d", "-s", plan["session"], "-n", "train", shlex.join(command)], check=True)
    print(f"started tmux={plan['session']} exp={plan['train']['exp_name']}")


def _status(plan: dict) -> None:
    subprocess.run(
        ["tmux", "list-panes", "-t", plan["session"], "-F", "session=#{session_name} dead=#{pane_dead} pid=#{pane_pid}"],
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if Path(plan["log"]).is_file():
        subprocess.run(["tail", "-n", "80", plan["log"]], check=True)
    if plan["device_backend"] == "musa" and shutil.which("mthreads-gmi"):
        subprocess.run(["mthreads-gmi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    elif shutil.which("nvidia-smi"):
        subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
        ], check=True)


def main(argv: list[str]) -> None:
    if len(argv) == 2 and argv[0] == "--resolved-training-config":
        # This process starts with the CFG's CUDA visibility, before importing JAX.
        import train

        train.main(_build_train_config(json.loads(argv[1])))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="launch", choices=("show", "preflight", "run", "launch", "status", "smoke", "benchmark"))
    parser.add_argument("--cfg", type=Path, default=os.environ.get("ROBOMME_DEFAULT_CFG"),
                        help=f"Python config. Defaults to ROBOMME_DEFAULT_CFG or {DEFAULT_CFG.relative_to(ROOT)}.")
    parser.add_argument("--steps", type=int, help="Total complete optimizer updates for the benchmark action (1–24)")
    parser.add_argument("--batch-offset", type=int, help="Start the benchmark at this loader batch in epoch zero")
    args = parser.parse_args(argv)
    cfg_path = _repo_path(args.cfg).resolve() if args.cfg is not None else None
    try:
        plan = _read_cfg(cfg_path or DEFAULT_CFG, implicit=cfg_path is None, smoke=args.action == "smoke")
        if args.action != "benchmark" and (args.steps is not None or args.batch_offset is not None):
            raise ValueError("--steps and --batch-offset require the benchmark action")
        if args.action == "benchmark":
            if plan["device_backend"] != "musa":
                raise ValueError("The benchmark action requires a MUSA configuration")
            steps = 5 if args.steps is None else args.steps
            offset = 0 if args.batch_offset is None else args.batch_offset
            if not 1 <= steps <= 24 or offset < 0:
                raise ValueError("benchmark requires 1–24 updates and a nonnegative batch offset")
            if plan["train"].get("resume") or plan["train"].get("restore_checkpoint_dir"):
                raise ValueError("benchmark requires a fresh configuration")
            directory = Path(plan["log"]).parents[1] / "benchmark"
            directory.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix=plan["train"]["exp_name"] + "-", dir=directory))
            plan["train"].update(num_train_steps=steps, log_interval=1, save_steps=[steps], overwrite=False,
                                 checkpoint_base_dir=str(run_dir), wandb_enabled=False)
            plan["log"] = str(run_dir / "train.log")
            print(f"MUSA benchmark output: {run_dir}", flush=True)
        if args.action == "show":
            visible_env = {
                key: value for key, value in plan["environment"].items()
                if key in {"CUDA_VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES", "JAX_PLATFORMS", "HF_HOME", "PYTHONPATH", "WANDB_MODE"}
                or key.startswith("XLA_PYTHON_CLIENT_")
            }
            print(json.dumps(plan | {"environment": visible_env}, indent=2))
        elif args.action == "preflight":
            _preflight(plan)
            visibility = "MUSA_VISIBLE_DEVICES" if plan["device_backend"] == "musa" else "CUDA_VISIBLE_DEVICES"
            print(f"Inputs verified for {plan['preset']} on {visibility}={plan['environment'][visibility]}")
        elif args.action == "launch":
            _launch(plan, cfg_path)
        elif args.action == "status":
            _status(plan)
        elif args.action in ("run", "smoke", "benchmark"):
            stats = _preflight(plan, bounded=args.action in ("smoke", "benchmark"))
            training_keys = ("preset", "paths", "train", "model", "data", "optimizer", "lr_schedule")
            training_plan = {key: plan[key] for key in training_keys}
            training_plan["model"] = plan["model"] | stats
            if plan["device_backend"] == "musa":
                if plan["smoke"]:
                    directory = Path(plan["log"]).parents[1] / "smoke"
                    directory.mkdir(parents=True, exist_ok=True)
                    run_dir = Path(tempfile.mkdtemp(prefix="ltm-pi-", dir=directory))
                    training_plan["train"].update(checkpoint_base_dir=str(run_dir), exp_name="smoke")
                    plan["log"] = str(run_dir / "train.log")
                    print(f"MUSA smoke output: {run_dir}", flush=True)
                training_plan.update(smoke=plan["smoke"], activation_checkpointing=plan["activation_checkpointing"])
                if args.action == "benchmark":
                    training_plan["benchmark"] = {"batch_offset": offset}
                for key in ("sample_effective_batch", "defer_gradient_sync", "activation_checkpoint_blocks"):
                    training_plan[key] = plan[key]
                world = len(plan["environment"]["MUSA_VISIBLE_DEVICES"].split(","))
                command = [plan["python"]]
                if world > 1:
                    command.extend(["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"])
                command.extend([str(ROOT / "scripts/train_musa.py"), json.dumps(training_plan, separators=(",", ":"))])
            else:
                command = [plan["python"], str(Path(__file__).resolve()), "--resolved-training-config",
                           json.dumps(training_plan, separators=(",", ":"))]
            os.chdir(ROOT)
            os.execvpe(
                "bash", ["bash", "-c", _RUN_WITH_LOG, "train-cfg", plan["log"], *command],
                os.environ.copy() | plan["environment"],
            )
    except (OSError, KeyError, ValueError, TypeError, SyntaxError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
