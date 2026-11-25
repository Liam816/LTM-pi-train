"""π0.5 flow matching with the released causal language–memory–vision prefix."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma.modeling_gemma import _gated_residual

from openpi.models import gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding, make_att_2d_masks
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch


def _rotate(x, positions):
    width = x.shape[-1]
    freq = 10000 ** (-torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    angle = positions[:, None, :, None] * freq
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * angle.cos() - right * angle.sin(), right * angle.cos() + left * angle.sin()), -1).to(
        x.dtype
    )


def _heads(proj, x, width):
    return proj(x).unflatten(-1, (-1, width)).transpose(1, 2)


class JointBlock(nn.Module):
    """The VLM and action expert have separate parameters and share attention."""

    def __init__(self, vlm, action, head_width):
        super().__init__()
        self.vlm, self.action, self.head_width = vlm, action, head_width

    def forward(self, prefix, suffix, positions, mask, valid, cond):
        queries, keys, values, gates = [], [], [], []
        for stream, layer, conditioning in ((prefix, self.vlm, None), (suffix, self.action, cond)):
            norm, gate = layer.input_layernorm(stream, cond=conditioning)
            queries.append(_heads(layer.self_attn.q_proj, norm, self.head_width))
            keys.append(_heads(layer.self_attn.k_proj, norm, self.head_width))
            values.append(_heads(layer.self_attn.v_proj, norm, self.head_width))
            gates.append(gate)
        q = _rotate(torch.cat(queries, 2), positions)
        k = _rotate(torch.cat(keys, 2), positions).expand(-1, q.shape[1], -1, -1)
        v = torch.cat(values, 2).expand_as(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
        out = out.masked_fill(~valid[:, None, :, None], 0).transpose(1, 2).flatten(2)
        chunks = out.split((prefix.shape[1], suffix.shape[1]), dim=1)
        result = []
        for x, layer, conditioning, chunk, gate in zip(
            (prefix, suffix), (self.vlm, self.action), (None, cond), chunks, gates, strict=True
        ):
            residual = _gated_residual(x, layer.self_attn.o_proj(chunk), gate)
            hidden, gate = layer.post_attention_layernorm(residual, cond=conditioning)
            result.append(_gated_residual(residual, layer.mlp(hidden), gate))
        return tuple(result)


def insert_block(tokens, valid, block, block_mask, index):
    """Insert one contiguous block at a different physical position per example."""
    pos = torch.arange(tokens.shape[1] + block.shape[1], device=tokens.device)[None, :]
    local = pos - index[:, None]
    inside = (local >= 0) & (local < block.shape[1])
    source = (pos - (local >= block.shape[1]).long() * block.shape[1]).clamp(0, tokens.shape[1] - 1)
    added = local.clamp(0, block.shape[1] - 1)
    batch = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
    return (
        torch.where(inside[..., None], block[batch, added], tokens[batch, source]),
        torch.where(inside, block_mask[batch, added], valid[batch, source]),
        local,
    )


class LTMPolicy(nn.Module):
    def __init__(self, cfg, *, activation_checkpointing=True):
        super().__init__()
        if not (
            cfg.pi05
            and cfg.memory_use_vlm_expert
            and cfg.memory_prefix_order == "language_memory_vision"
            and cfg.language_causal_attention
            and cfg.memory_causal_attention
            and cfg.memory_frames_per_token == 1
            and cfg.memory_segment_embedding
            and cfg.memory_segment_embedding_after_projection
            and cfg.memory_demo_anchor
            and cfg.memory_execution_anchor
            and cfg.memory_demo_lam_boundary_token
            and not cfg.memory_demo_anchor_feature_pool
            and not cfg.memory_flow_horizon
            and not cfg.memory_register_count
            and not cfg.memory_demo_direction_generation
            and not cfg.memory_episode_anchor
            and not cfg.memory_anchor_only
            and cfg.memory_fixed_random_latent_seed is None
            and not cfg.action_motion_weighting
            and not cfg.baseline_uncertainty_weighting
            and not cfg.hardness_weighting
            and not cfg.memory_advantage_weighting
        ):
            raise ValueError(
                "This PyTorch backend supports the released framewise, dual-anchor causal LMV architecture"
            )
        vlm, expert = gemma.get_config(cfg.paligemma_variant), gemma.get_config(cfg.action_expert_variant)
        self.cfg, self.compute_dtype = cfg, getattr(torch, cfg.dtype)
        self.activation_checkpointing = activation_checkpointing
        with no_init_weights():
            backbone = PaliGemmaWithExpertModel(vlm, expert, use_adarms=[False, True], precision="float32")
        language, action = backbone.paligemma.language_model, backbone.gemma_expert.model
        self.vision = backbone.paligemma.vision_tower.vision_model
        if self.vision.use_head:
            self.vision.use_head = False
            del self.vision.head
        self.projector = backbone.paligemma.multi_modal_projector.linear
        self.tokens = language.embed_tokens
        self.blocks = nn.ModuleList(
            JointBlock(v, a, vlm.head_dim) for v, a in zip(language.layers, action.layers, strict=True)
        )
        self.vlm_norm, self.action_norm = language.norm, action.norm
        self.action_in_proj = nn.Linear(cfg.action_dim, expert.width)
        self.action_out_proj = nn.Linear(expert.width, cfg.action_dim)
        self.time_mlp_in = nn.Linear(expert.width, expert.width)
        self.time_mlp_out = nn.Linear(expert.width, expert.width)
        self.memory_projector_in = nn.Linear(cfg.memory_latent_dim, cfg.memory_projector_hidden_dim)
        self.memory_projector_out = nn.Linear(cfg.memory_projector_hidden_dim, vlm.width)
        self.memory_projector_norm = nn.LayerNorm(vlm.width, eps=1e-6)
        self.memory_segment_embed = nn.Embedding(cfg.memory_segment_vocab_size, vlm.width)
        self.register_buffer("latent_mean", torch.tensor(cfg.memory_latent_mean, dtype=torch.float32))
        self.register_buffer("latent_std", torch.tensor(cfg.memory_latent_std, dtype=torch.float32))

    def encode_prefix(self, obs):
        cfg = self.cfg
        images = [obs.images[key] for key in cfg.active_image_keys]
        stacked = torch.stack([*images, obs.memory_demo_start_image, obs.memory_execution_start_image], dim=1)
        visual = self.projector(self.vision(pixel_values=stacked.flatten(0, 1)).last_hidden_state)
        visual = visual.unflatten(0, stacked.shape[:2]).to(self.compute_dtype)
        current, demo, execution = visual[:, :-2].flatten(1, 2), visual[:, -2], visual[:, -1]
        patches = demo.shape[1]
        words = (self.tokens(obs.tokenized_prompt) * math.sqrt(self.tokens.embedding_dim)).to(self.compute_dtype)
        seg, valid = obs.memory_segment_ids.long(), obs.memory_mask.bool()
        latents = torch.where(valid[..., None], (obs.memory_latents.float() - self.latent_mean) / self.latent_std, 0.0)
        # The native memory projector consumes normalized FP32 latents and FP32 weights.
        with torch.autocast(seg.device.type, enabled=False):
            tokens = self.memory_projector_out(F.gelu(self.memory_projector_in(latents), approximate="tanh"))
            tokens = self.memory_projector_norm(tokens + self.memory_segment_embed(seg)).to(self.compute_dtype)
        if self.training and cfg.memory_token_dropout_rate:
            valid = valid & (torch.rand(valid.shape, device=valid.device) >= cfg.memory_token_dropout_rate)
        tokens = tokens.masked_fill(~valid[..., None], 0)

        def typed(template, type_id):
            return self.memory_segment_embed(
                torch.full(template.shape[:2], type_id, dtype=torch.long, device=seg.device)
            )

        demo = demo + typed(demo, cfg.memory_demo_anchor_type_id).to(demo.dtype)
        execution = execution + typed(execution, cfg.memory_execution_anchor_type_id).to(execution.dtype)
        sep = self.memory_projector_norm(typed(tokens[:, :1], cfg.memory_demo_lam_boundary_type_id)).to(tokens.dtype)
        dm, em = obs.memory_demo_start_mask.bool()[:, None], obs.memory_execution_start_mask.bool()[:, None]
        tokens, valid, _ = insert_block(tokens, valid, sep, dm, (seg == 0).sum(1))
        tokens, valid, local = insert_block(tokens, valid, execution, em.expand(-1, patches), (seg <= 1).sum(1) + 1)
        current_mask = torch.stack([obs.image_masks[key] for key in cfg.active_image_keys], 1).repeat_interleave(
            patches, dim=1
        )
        masks = (obs.tokenized_prompt_mask.bool(), dm.expand(-1, patches), valid, current_mask)
        demo_ar, current_ar = torch.zeros_like(masks[1]), torch.zeros_like(current_mask)
        demo_ar[:, 0], current_ar[:, 0] = True, True
        boundaries = torch.cat((torch.ones_like(masks[0]), demo_ar, (local <= 0) | (local >= patches), current_ar), 1)
        return torch.cat((words, demo, tokens, current), 1), torch.cat(masks, 1), boundaries

    def forward(self, observation, actions):
        obs = preprocess_observation_pytorch(observation, train=self.training, image_keys=self.cfg.active_image_keys)
        device = actions.device
        # Inverse CDF for Beta(1.5,1), avoiding a device-specific gamma sampler.
        time = torch.rand(actions.shape[0], device=device).pow(1 / 1.5).mul_(0.999).add_(0.001)
        noise = torch.randn_like(actions)
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        time_embedding = create_sinusoidal_pos_embedding(time, self.time_mlp_in.in_features, 4e-3, 4.0, device).float()
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_embedding))))
        suffix = self.action_in_proj(noisy).to(self.compute_dtype)
        with torch.autocast(device.type, dtype=self.compute_dtype, enabled=self.compute_dtype == torch.bfloat16):
            prefix, valid, ar = self.encode_prefix(obs)
            action_mask = torch.ones_like(suffix[..., 0], dtype=torch.bool)
            action_ar = torch.zeros_like(action_mask)
            action_ar[:, 0] = True
            valid = torch.cat((valid, action_mask), 1)
            mask = make_att_2d_masks(valid, torch.cat((ar, action_ar), 1))[:, None]
            # Moore Threads SDPA needs one finite key even for padding queries.
            mask[..., 0] |= ~valid[:, None, :]
            positions = valid.long().cumsum(1) - 1
            for block in self.blocks:
                args = (prefix, suffix, positions, mask, valid, cond)
                if self.training and self.activation_checkpointing:
                    prefix, suffix = checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)
                else:
                    prefix, suffix = block(*args)
            suffix, _ = self.action_norm(suffix, cond=cond)
        return (self.action_out_proj(suffix.float()) - (noise - actions)).square().mean()
