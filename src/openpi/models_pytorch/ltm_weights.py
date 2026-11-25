"""Load the released π0.5 parameters into the two PyTorch experts and SigLIP tower."""

from dataclasses import dataclass

import einops
from flax import nnx, traverse_util
import jax
import ml_dtypes
import numpy as np
import torch
import orbax.checkpoint as ocp

from openpi.models.model import restore_params


@dataclass
class Rule:
    path: str
    targets: tuple
    layout: str
    axes: dict


def weight_rules(*, include_memory=False):
    rules = []

    def add(path, targets, layout, **axes):
        if isinstance(targets, str):
            targets = (targets,)
        rules.append(Rule(path, tuple(targets), layout, axes))

    def dense(jax, pt):
        add(jax + "/kernel", pt + ".weight", "i o -> 1 o i")
        add(jax + "/bias", pt + ".bias", "o -> 1 o")

    img = "PaliGemma/img/"
    add(img + "embedding/kernel", "vision.embeddings.patch_embedding.weight", "h w i o -> 1 o i h w")
    add(img + "embedding/bias", "vision.embeddings.patch_embedding.bias", "o -> 1 o")
    add(img + "pos_embedding", "vision.embeddings.position_embedding.weight", "1 s d -> 1 s d")
    for suffix, torch_suffix in (("scale", "weight"), ("bias", "bias")):
        add(img + "Transformer/encoder_norm/" + suffix, "vision.post_layernorm." + torch_suffix, "d -> 1 d")
    dense(img + "head", "projector")
    enc = img + "Transformer/encoderblock/"
    for suffix, pt in (
        ("LayerNorm_0", "layer_norm1"),
        ("LayerNorm_1", "layer_norm2"),
        ("MlpBlock_0/Dense_0", "mlp.fc1"),
        ("MlpBlock_0/Dense_1", "mlp.fc2"),
    ):
        is_norm = suffix.startswith("LayerNorm")
        names = [f"vision.encoder.layers.{i}.{pt}" for i in range(27)]
        add(
            enc + suffix + ("/scale" if is_norm else "/kernel"),
            (n + ".weight" for n in names),
            "l d -> l d" if is_norm else "l i o -> l o i",
        )
        add(enc + suffix + "/bias", (n + ".bias" for n in names), "l o -> l o")
    for key, pt in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("out", "out_proj")):
        path = enc + "MultiHeadDotProductAttention_0/" + key
        names = [f"vision.encoder.layers.{i}.self_attn.{pt}" for i in range(27)]
        output = key == "out"
        add(
            path + "/kernel",
            (n + ".weight" for n in names),
            "l h d o -> l o (h d)" if output else "l i h d -> l (h d) i",
            h=16,
        )
        if output:
            add(path + "/bias", (n + ".bias" for n in names), "l o -> l o")
        else:
            add(path + "/bias", (n + ".bias" for n in names), "l h d -> l (h d)", h=16)

    llm = "PaliGemma/llm/"
    add(llm + "embedder/input_embedding", "tokens.weight", "v d -> 1 v d")
    add(llm + "final_norm/scale", "vlm_norm.weight", "d -> 1 d")
    dense(llm + "final_norm_1/Dense_0", "action_norm.dense")
    layers = llm + "layers/"
    for kind, suffix in (("vlm", ""), ("action", "_1")):
        names = [f"blocks.{i}.{kind}" for i in range(18)]
        att = layers + "attn/"
        attentions = [name + ".self_attn" for name in names]
        add(att + "q_einsum" + suffix + "/w", (n + ".q_proj.weight" for n in attentions), "l h i d -> l (h d) i", h=8)
        add(
            att + "kv_einsum" + suffix + "/w",
            (n + "." + proj + ".weight" for n in attentions for proj in ("k_proj", "v_proj")),
            "l k 1 i d -> (l k) d i",
            k=2,
        )
        add(
            att + "attn_vec_einsum" + suffix + "/w",
            (n + ".o_proj.weight" for n in attentions),
            "l h d o -> l o (h d)",
            h=8,
        )
        add(
            layers + "mlp" + suffix + "/gating_einsum",
            (n + ".mlp." + p + ".weight" for n in names for p in ("gate_proj", "up_proj")),
            "l k i o -> (l k) o i",
            k=2,
        )
        add(layers + "mlp" + suffix + "/linear", (n + ".mlp.down_proj.weight" for n in names), "l i o -> l o i")
        for jax_norm, pt_norm in (
            ("pre_attention_norm", "input_layernorm"),
            ("pre_ffw_norm", "post_attention_layernorm"),
        ):
            path = layers + jax_norm + suffix
            if kind == "vlm":
                add(path + "/scale", (f"{n}.{pt_norm}.weight" for n in names), "l d -> l d")
            else:
                add(path + "/Dense_0/kernel", (f"{n}.{pt_norm}.dense.weight" for n in names), "l i o -> l o i")
                add(path + "/Dense_0/bias", (f"{n}.{pt_norm}.dense.bias" for n in names), "l o -> l o")
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        dense(name, name)
    if include_memory:
        for module in ("memory_projector_in", "memory_projector_out"):
            dense(module, module)
        add("memory_projector_norm/scale", "memory_projector_norm.weight", "d -> 1 d")
        add("memory_projector_norm/bias", "memory_projector_norm.bias", "d -> 1 d")
        add("memory_segment_embed/embedding", "memory_segment_embed.weight", "v d -> 1 v d")
    return rules


@torch.no_grad()
def load_base(model, path, seed):
    """Load every Orbax π0.5 parameter and initialize LMV with the native model seed."""
    flat = traverse_util.flatten_dict(restore_params(path, restore_type=np.ndarray), sep="/")

    # Match scripts/train.py RNG splitting. JAX compiles only these new leaves on CPU.
    @jax.jit
    def adapter(key):
        native = model.cfg.create(key)
        return {
            name: nnx.state(getattr(native, name)).to_pure_dict()
            for name in ("memory_projector_in", "memory_projector_out", "memory_projector_norm", "memory_segment_embed")
        }

    _, init_rng = jax.random.split(jax.random.key(seed))
    _, model_rng = jax.random.split(init_rng)
    flat.update(traverse_util.flatten_dict(jax.device_get(adapter(model_rng)), sep="/"))
    params = dict(model.named_parameters())
    for rule in weight_rules(include_memory=True):
        array = np.asarray(flat.pop(rule.path))
        if array.dtype == ml_dtypes.bfloat16:
            tensor = torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(array)
        values = einops.rearrange(tensor, rule.layout, **rule.axes)
        for target, value in zip(rule.targets, values.unbind(), strict=True):
            parameter = params.pop(target)
            if parameter.shape != value.shape:
                raise ValueError(f"{target}: base shape {value.shape}, model shape {parameter.shape}")
            parameter.copy_(value)
    if flat or params:
        raise ValueError(f"Unmapped base parameters: Orbax {list(flat)}, PyTorch {list(params)}")


def export_inference(checkpoint_dir, world, shapes, cfg):
    """Write the EMA in the Orbax layout consumed by the existing policy server."""
    field = "params" if cfg.ema_decay is None else "ema"
    shards = [
        torch.load(checkpoint_dir / f"rank-{rank}.pt", map_location="cpu", mmap=True, weights_only=False)[field]
        for rank in range(world)
    ]
    if any(shard.keys() != shapes.keys() for shard in shards):
        raise ValueError("Native checkpoint shapes and saved parameter names differ")
    tree = {}
    for rule in weight_rules(include_memory=True):
        parts = []
        for name in rule.targets:
            # FSDP use_orig_params exposes contiguous, potentially empty, rank-local slices.
            slices = [shard.pop(name).reshape(-1) for shard in shards]
            parts.append(torch.cat(slices).reshape(shapes[name]))
        stacked = torch.stack(parts) if len(parts) > 1 else parts[0].unsqueeze(0)
        source, target = rule.layout.split(" -> ")
        native = einops.rearrange(stacked, target + " -> " + source, **rule.axes).contiguous()
        array = (
            native.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            if native.dtype == torch.bfloat16
            else native.numpy()
        )
        tree[rule.path + "/value"] = array
    if any(shards):
        raise ValueError("Unmapped PyTorch parameters during Orbax export")
    for name in ("memory_latent_mean", "memory_latent_std"):
        tree[name + "/value"] = np.asarray(getattr(cfg.model, name), dtype=np.float32)
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(str(checkpoint_dir / "params"), {"params": traverse_util.unflatten_dict(tree, sep="/")})
