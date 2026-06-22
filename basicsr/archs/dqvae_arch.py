"""
Dynamic Vector Quantization (DQ-VAE) modules integrated into CodeFormer/BasicSR.

This file ports the stage-1 implementation used by DynamicVectorQuantization into
CodeFormer, while keeping BasicSR's registry/checkpoint conventions.  The dual
configuration reproduces dqvae-dual-r-05_imagenet.yml in architecture, VQ,
routing, budget constraint, discriminator and loss formulation.  The triple
configuration extends the candidate downsampling granularities from {8, 16} to
{8, 16, 32}; for 512px faces this corresponds to regular code maps of 64x64,
32x32 and 16x16 respectively.
"""
from __future__ import annotations

import functools
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY


# -----------------------------------------------------------------------------
# DQ-VAE / VQGAN basic blocks. These mirror DynamicVectorQuantization's
# modules/diffusionmodules/model.py.
# -----------------------------------------------------------------------------

def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None,
                 conv_shortcut: bool = False, dropout: float = 0.0,
                 temb_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, **ignore_kwargs) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


# -----------------------------------------------------------------------------
# DQ routing and budget constraints. Training routing is Gumbel-Softmax hard
# sampling; validation/inference routing is deterministic argmax, exactly as in
# DynamicVectorQuantization's feature routers.
# -----------------------------------------------------------------------------

class DualGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels: int, normalization_type: str = "none", gate_type: str = "1layer-fc"):
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate_type = gate_type
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * 2, 2)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * 2, 2),
            )
        else:
            raise NotImplementedError(f"Unsupported gate_type: {gate_type}")

        if normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in normalization_type:
            num_groups = int(normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError(f"Unsupported normalization_type: {normalization_type}")

    def forward(self, h_fine: torch.Tensor, h_coarse: torch.Tensor, entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)
        avg_h_fine = self.gate_pool(h_fine)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0, 2, 3, 1)
        return self.gate(h_logistic)  # B,Hc,Wc,2


class TripleGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels: int, normalization_type: str = "none", gate_type: str = "1layer-fc"):
        super().__init__()
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.gate_fine_pool = nn.AvgPool2d(4, 4)
        self.num_splits = 3
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * self.num_splits, self.num_splits)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * self.num_splits, num_channels * self.num_splits),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        elif gate_type == "2layer-fc-ReLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * self.num_splits, num_channels * self.num_splits),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        else:
            raise NotImplementedError(f"Unsupported gate_type: {gate_type}")

        if normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_median = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in normalization_type:
            num_groups = int(normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_median = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError(f"Unsupported normalization_type: {normalization_type}")

    def forward(self, h_fine: torch.Tensor, h_median: torch.Tensor, h_coarse: torch.Tensor,
                entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_fine = self.feature_norm_fine(h_fine)
        h_median = self.feature_norm_median(h_median)
        h_coarse = self.feature_norm_coarse(h_coarse)
        avg_h_fine = self.gate_fine_pool(h_fine)
        avg_h_median = self.gate_median_pool(h_median)
        h_logistic = torch.cat([h_coarse, avg_h_median, avg_h_fine], dim=1).permute(0, 2, 3, 1)
        return self.gate(h_logistic)  # B,Hc,Wc,3


class BudgetConstraint_RatioMSE_DualGrain(nn.Module):
    """DQ-VAE dual budget loss from dqvae-dual-r-05_imagenet.yml."""
    def __init__(self, target_ratio: float = 0.5, gamma: float = 10.0,
                 min_grain_size: int = 16, max_grain_size: int = 32,
                 calculate_all: bool = True):
        super().__init__()
        self.target_ratio = target_ratio
        self.gamma = gamma
        self.calculate_all = calculate_all
        self.loss = nn.MSELoss()
        self.const = min_grain_size * min_grain_size
        self.max_const = max_grain_size * max_grain_size - self.const

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        # 0: coarse, 1: fine; gate shape B,2,Hc,Wc.
        beta = 1.0 * gate[:, 0, :, :] + 4.0 * gate[:, 1, :, :]
        beta = (beta.sum() / gate.size(0)) - self.const
        budget_ratio = beta / self.max_const
        target_ratio = self.target_ratio * torch.ones_like(budget_ratio, device=gate.device)
        loss_budget = self.gamma * self.loss(budget_ratio, target_ratio)
        if self.calculate_all:
            # Keep the original DQ implementation, including its duplicated term.
            loss_budget_last = self.gamma * self.loss(1 - budget_ratio, 1 - target_ratio)
            return loss_budget_last + loss_budget_last
        return loss_budget


class BudgetConstraint_NormedSeperateRatioMSE_TripleGrain(nn.Module):
    """DQ-VAE triple budget loss for granularities {8,16,32}."""
    def __init__(self, target_fine_ratio: float = 0.3, target_median_ratio: float = 0.3,
                 gamma: float = 1.0, min_grain_size: int = 16,
                 median_grain_size: int = 32, max_grain_size: int = 64):
        super().__init__()
        assert target_fine_ratio + target_median_ratio <= 1.0
        self.target_fine_ratio = target_fine_ratio
        self.target_median_ratio = target_median_ratio
        self.gamma = gamma
        self.loss = nn.MSELoss()
        self.min_const = min_grain_size * min_grain_size
        self.median_const = median_grain_size * median_grain_size - self.min_const
        self.max_const = max_grain_size * max_grain_size - self.min_const

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        # 0: coarse, 1: median, 2: fine; gate shape B,3,Hc,Wc.
        beta_median = 1.0 * gate[:, 0, :, :] + 4.0 * gate[:, 1, :, :] + 1.0 * gate[:, 2, :, :]
        beta_median = (beta_median.sum() / gate.size(0)) - self.min_const
        budget_ratio_median = beta_median / self.median_const
        target_ratio_median = self.target_median_ratio * torch.ones_like(budget_ratio_median, device=gate.device)
        loss_budget_median = self.loss(budget_ratio_median, target_ratio_median)

        beta_fine = 1.0 * gate[:, 0, :, :] + 16.0 * gate[:, 2, :, :] + 1.0 * gate[:, 1, :, :]
        beta_fine = (beta_fine.sum() / gate.size(0)) - self.min_const
        budget_ratio_fine = beta_fine / self.max_const
        target_ratio_fine = self.target_fine_ratio * torch.ones_like(budget_ratio_fine, device=gate.device)
        loss_budget_fine = self.gamma * self.loss(budget_ratio_fine, target_ratio_fine)
        return loss_budget_fine + loss_budget_median


# -----------------------------------------------------------------------------
# EMA vector quantization. This mirrors quantize2_mask.VectorQuantize2.
# -----------------------------------------------------------------------------

class VQEmbedding(nn.Embedding):
    def __init__(self, n_embed: int, embed_dim: int, ema: bool = True, decay: float = 0.99,
                 restart_unused_codes: bool = True, eps: float = 1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        codebook_t = self.weight[:-1, :].t()
        embed_dim, _ = codebook_t.shape
        assert inputs.shape[-1] == embed_dim
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(inputs_norm_sq + codebook_t_norm_sq, inputs_flat, codebook_t, alpha=-2.0)
        return distances.reshape(*inputs.shape[:-1], -1)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        distances = self.compute_distances(inputs)
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x: torch.Tensor, target_n: int) -> torch.Tensor:
        b, embed_dim = x.shape
        n_repeats = (target_n + b - 1) // b
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _sample_restart_vectors(self, vectors: torch.Tensor, weights: Optional[torch.Tensor],
                                target_n: int) -> torch.Tensor:
        """Sample replacement vectors for unused codes.

        When `weights` is provided, sampling follows the same dynamic-token
        weighting as the EMA update.  This prevents repeated coarse/median
        vectors from dominating random restarts merely because they were expanded
        to the finest regular grid.
        """
        n_vectors, embed_dim = vectors.shape
        if weights is None:
            if n_vectors < target_n:
                vectors = self._tile_with_noise(vectors, target_n)
                n_vectors = vectors.shape[0]
            return vectors[torch.randperm(n_vectors, device=vectors.device)][:target_n]

        weights = weights.reshape(-1).to(device=vectors.device, dtype=torch.float32).clamp_min(0)
        valid = torch.isfinite(weights) & (weights > 0)
        if valid.any():
            candidates = vectors[valid]
            candidate_weights = weights[valid]
            weight_sum = candidate_weights.sum()
            if torch.isfinite(weight_sum) and float(weight_sum) > 0.0:
                sample_idxs = torch.multinomial(candidate_weights, target_n, replacement=True)
                sampled = candidates[sample_idxs]
                if candidates.shape[0] < target_n:
                    std = sampled.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
                    sampled = sampled + torch.rand_like(sampled) * std
                return sampled

        # Defensive fallback for degenerate all-zero / non-finite masks.
        if n_vectors < target_n:
            vectors = self._tile_with_noise(vectors, target_n)
            n_vectors = vectors.shape[0]
        return vectors[torch.randperm(n_vectors, device=vectors.device)][:target_n]

    @torch.no_grad()
    def _update_buffers(self, vectors: torch.Tensor, idxs: torch.Tensor,
                        ema_weights: Optional[torch.Tensor] = None) -> None:
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        n_vectors = vectors.shape[0]

        flat_weights: Optional[torch.Tensor]
        if ema_weights is None:
            flat_weights = None
            scatter_weights = vectors.new_ones(1, n_vectors)
        else:
            flat_weights = ema_weights.reshape(-1).to(device=vectors.device, dtype=vectors.dtype)
            if flat_weights.numel() != n_vectors:
                raise ValueError(
                    f"EMA weight shape is incompatible with VQ inputs: "
                    f"got {tuple(ema_weights.shape)} -> {flat_weights.numel()} weights, "
                    f"expected {n_vectors}."
                )
            flat_weights = torch.nan_to_num(flat_weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            scatter_weights = flat_weights.unsqueeze(0)

        one_hot_idxs = vectors.new_zeros(n_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0, index=idxs.unsqueeze(0), src=scatter_weights)
        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors
        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        if self.restart_unused_codes:
            vectors_random = self._sample_restart_vectors(vectors, flat_weights, n_embed)
            if dist.is_initialized():
                dist.broadcast(vectors_random, 0)
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self) -> None:
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        if float(n) <= 0.0:
            return
        normalized_cluster_size = n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs: torch.Tensor,
                ema_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs, ema_weights=ema_weights)
        embeds = self.embed(embed_idxs)
        if self.training and self.ema:
            self._update_embedding()
        return embeds, embed_idxs

    def embed(self, idxs: torch.Tensor) -> torch.Tensor:
        return super().forward(idxs)


class VectorQuantize2(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int, accept_image_fmap: bool = True,
                 commitment_beta: float = 0.25, decay: float = 0.99,
                 restart_unused_codes: bool = True, channel_last: bool = False):
        super().__init__()
        self.accept_image_fmap = accept_image_fmap
        self.beta = commitment_beta
        self.channel_last = channel_last
        self.restart_unused_codes = restart_unused_codes
        self.codebook = VQEmbedding(codebook_size, codebook_dim, decay=decay,
                                    restart_unused_codes=restart_unused_codes)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    @property
    def codebook_size(self) -> int:
        return self.codebook.n_embed

    @property
    def embedding(self) -> VQEmbedding:
        # compatibility with original CodeFormer utilities
        return self.codebook

    def forward(self, x: torch.Tensor, codebook_mask: Optional[torch.Tensor] = None,
                *ignoreargs, **ignorekwargs) -> Tuple[torch.Tensor, torch.Tensor, Tuple[None, None, torch.Tensor]]:
        need_transpose = not self.channel_last and not self.accept_image_fmap
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if need_transpose:
            x = rearrange(x, "b d n -> b n d").contiguous()

        vq_loss_mask: Optional[torch.Tensor] = None
        ema_update_weight: Optional[torch.Tensor] = None
        if codebook_mask is not None:
            if self.accept_image_fmap and codebook_mask.dim() == 4:
                vq_loss_mask = rearrange(codebook_mask, "b c h w -> b (h w) c").contiguous()
            elif (self.accept_image_fmap and codebook_mask.dim() == 3
                  and tuple(codebook_mask.shape[-2:]) == (height, width)):
                vq_loss_mask = rearrange(codebook_mask, "b h w -> b (h w) 1").contiguous()
            else:
                vq_loss_mask = codebook_mask
                if vq_loss_mask.dim() == x.dim() - 1:
                    vq_loss_mask = vq_loss_mask.unsqueeze(-1)
            vq_loss_mask = vq_loss_mask.to(device=x.device, dtype=x.dtype)
            if vq_loss_mask.shape[:-1] != x.shape[:-1]:
                raise ValueError(
                    f"codebook_mask shape is incompatible with VQ inputs: "
                    f"mask={tuple(vq_loss_mask.shape)}, inputs={tuple(x.shape)}."
                )
            if vq_loss_mask.shape[-1] not in (1, x.shape[-1]):
                raise ValueError(
                    f"codebook_mask last dimension must be 1 or channel dim {x.shape[-1]}, "
                    f"but got {vq_loss_mask.shape[-1]}."
                )
            # Same scalar dynamic-token weight is used for EMA cluster/update.
            # For triple DQ-VAE, coarse/median/fine repeated positions contribute
            # 16 * 1/16, 4 * 1/4 and 1 * 1 real token respectively.
            ema_update_weight = vq_loss_mask.squeeze(-1) if vq_loss_mask.shape[-1] == 1 else vq_loss_mask.mean(dim=-1)

        flatten = rearrange(x, "h ... d -> h (...) d").contiguous()
        x_q, x_code = self.codebook(flatten, ema_weights=ema_update_weight)
        if vq_loss_mask is not None:
            loss = self.beta * torch.mean((x_q.detach() - x) ** 2 * vq_loss_mask) + \
                torch.mean((x_q - x.detach()) ** 2 * vq_loss_mask)
        else:
            loss = self.beta * torch.mean((x_q.detach() - x) ** 2) + torch.mean((x_q - x.detach()) ** 2)
        x_q = x + (x_q - x).detach()
        if need_transpose:
            x_q = rearrange(x_q, "b n d -> b d n").contiguous()
        if self.accept_image_fmap:
            x_q = rearrange(x_q, "b (h w) c -> b c h w", h=height, w=width).contiguous()
            x_code = rearrange(x_code, "b (h w) ... -> b h w ...", h=height, w=width).contiguous()
        return x_q, loss, (None, None, x_code)

    @torch.no_grad()
    def get_soft_codes(self, x: torch.Tensor, temp: float = 1.0, stochastic: bool = False):
        distances = self.codebook.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)
        if stochastic:
            soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
            code = torch.multinomial(soft_code_flat, 1).reshape(*soft_code.shape[:-1])
        else:
            code = distances.argmin(dim=-1)
        return soft_code, code

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        # indices can be B,H,W or flattened. Return B,C,H,W if shape is given.
        z_q = self.codebook.embed(indices)
        if shape is not None:
            b, h, w, c = shape
            z_q = z_q.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return z_q

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 3:
            z_q = self.codebook.embed(indices).permute(0, 3, 1, 2).contiguous()
        else:
            z_q = self.codebook.embed(indices)
        return z_q

    def embed_code_with_depth(self, code: torch.Tensor) -> torch.Tensor:
        return self.embed_code(code)


# -----------------------------------------------------------------------------
# Encoders and decoder.
# -----------------------------------------------------------------------------

class DualGrainEncoder(nn.Module):
    def __init__(self, *, ch: int, ch_mult: Tuple[int, ...], num_res_blocks: int,
                 attn_resolutions: List[int], dropout: float, resamp_with_conv: bool,
                 in_channels: int, resolution: int, z_channels: int,
                 router_config: Optional[dict] = None, update_router: bool = True,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.update_router = update_router
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid_coarse = nn.Module()
        self.mid_coarse.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.mid_coarse.attn_1 = AttnBlock(block_in)
        self.mid_coarse.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out_coarse = Normalize(block_in)
        self.conv_out_coarse = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

        block_in_fine = block_in // (ch_mult[-1] // ch_mult[-2])
        self.mid_fine = nn.Module()
        self.mid_fine.block_1 = ResnetBlock(in_channels=block_in_fine, out_channels=block_in_fine,
                                            temb_channels=self.temb_ch, dropout=dropout)
        self.mid_fine.attn_1 = AttnBlock(block_in_fine)
        self.mid_fine.block_2 = ResnetBlock(in_channels=block_in_fine, out_channels=block_in_fine,
                                            temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out_fine = Normalize(block_in_fine)
        self.conv_out_fine = nn.Conv2d(block_in_fine, z_channels, kernel_size=3, stride=1, padding=1)

        rconf = router_config or {}
        self.router = DualGrainFeatureRouter(
            num_channels=rconf.get("num_channels", z_channels),
            normalization_type=rconf.get("normalization_type", "group-32"),
            gate_type=rconf.get("gate_type", "2layer-fc-SiLu"),
        )

    def forward(self, x: torch.Tensor, x_entropy: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        assert x.shape[2] == x.shape[3] == self.resolution, f"{x.shape[2:]}, resolution={self.resolution}"
        temb = None
        features: Dict[str, torch.Tensor] = {}
        hs = [self.conv_in(x)]
        features[str(hs[-1].shape[-1])] = hs[-1]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
                features[str(h.shape[-1])] = h
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                features[str(hs[-1].shape[-1])] = hs[-1]
            if i_level == self.num_resolutions - 2:
                h_fine = h
        h_coarse = hs[-1]
        h_coarse = self.mid_coarse.block_1(h_coarse, temb)
        h_coarse = self.mid_coarse.attn_1(h_coarse)
        h_coarse = self.mid_coarse.block_2(h_coarse, temb)
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = nonlinearity(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        h_fine = self.mid_fine.block_1(h_fine, temb)
        h_fine = self.mid_fine.attn_1(h_fine)
        h_fine = self.mid_fine.block_2(h_fine, temb)
        h_fine = self.norm_out_fine(h_fine)
        h_fine = nonlinearity(h_fine)
        h_fine = self.conv_out_fine(h_fine)
        features[str(h_fine.shape[-1])] = h_fine

        gate_logits = self.router(h_fine=h_fine, h_coarse=h_coarse, entropy=x_entropy)
        if self.update_router and self.training:
            gate = F.gumbel_softmax(gate_logits, dim=-1, hard=True)
        else:
            gate_indices = gate_logits.argmax(dim=-1)
            gate = F.one_hot(gate_indices, num_classes=2).to(dtype=gate_logits.dtype)
        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)
        h_coarse_up = h_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = indices.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).unsqueeze(1)
        h_dual = torch.where(indices_repeat == 0, h_coarse_up, h_fine)
        if self.update_router and self.training:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            h_dual = h_dual * gate_grad
        coarse_mask = 0.25 * torch.ones_like(indices_repeat, device=h_dual.device)
        fine_mask = torch.ones_like(indices_repeat, device=h_dual.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, fine_mask)
        out = {"h_dual": h_dual, "indices": indices, "codebook_mask": codebook_mask, "gate": gate,
               "gate_logits": gate_logits, "h_coarse": h_coarse, "h_fine": h_fine}
        if return_features:
            out["features"] = features
        return out


class TripleGrainEncoder(nn.Module):
    def __init__(self, *, ch: int, ch_mult: Tuple[int, ...], num_res_blocks: int,
                 attn_resolutions: List[int], dropout: float, resamp_with_conv: bool,
                 in_channels: int, resolution: int, z_channels: int,
                 router_config: Optional[dict] = None, update_router: bool = True,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.update_router = update_router
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid_coarse = nn.Module()
        self.mid_coarse.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.mid_coarse.attn_1 = AttnBlock(block_in)
        self.mid_coarse.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out_coarse = Normalize(block_in)
        self.conv_out_coarse = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

        block_in_median = block_in // (ch_mult[-1] // ch_mult[-2])
        self.mid_median = nn.Module()
        self.mid_median.block_1 = ResnetBlock(in_channels=block_in_median, out_channels=block_in_median,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.mid_median.attn_1 = AttnBlock(block_in_median)
        self.mid_median.block_2 = ResnetBlock(in_channels=block_in_median, out_channels=block_in_median,
                                              temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out_median = Normalize(block_in_median)
        self.conv_out_median = nn.Conv2d(block_in_median, z_channels, kernel_size=3, stride=1, padding=1)

        block_in_fine = block_in_median // (ch_mult[-2] // ch_mult[-3])
        self.mid_fine = nn.Module()
        self.mid_fine.block_1 = ResnetBlock(in_channels=block_in_fine, out_channels=block_in_fine,
                                            temb_channels=self.temb_ch, dropout=dropout)
        self.mid_fine.attn_1 = AttnBlock(block_in_fine)
        self.mid_fine.block_2 = ResnetBlock(in_channels=block_in_fine, out_channels=block_in_fine,
                                            temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out_fine = Normalize(block_in_fine)
        self.conv_out_fine = nn.Conv2d(block_in_fine, z_channels, kernel_size=3, stride=1, padding=1)

        rconf = router_config or {}
        self.router = TripleGrainFeatureRouter(
            num_channels=rconf.get("num_channels", z_channels),
            normalization_type=rconf.get("normalization_type", "group-32"),
            gate_type=rconf.get("gate_type", "2layer-fc-SiLu"),
        )

    def forward(self, x: torch.Tensor, x_entropy: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        assert x.shape[2] == x.shape[3] == self.resolution, f"{x.shape[2:]}, resolution={self.resolution}"
        temb = None
        features: Dict[str, torch.Tensor] = {}
        hs = [self.conv_in(x)]
        features[str(hs[-1].shape[-1])] = hs[-1]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
                features[str(h.shape[-1])] = h
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                features[str(hs[-1].shape[-1])] = hs[-1]
            if i_level == self.num_resolutions - 2:
                h_median = h
            elif i_level == self.num_resolutions - 3:
                h_fine = h
        h_coarse = hs[-1]

        h_coarse = self.mid_coarse.block_1(h_coarse, temb)
        h_coarse = self.mid_coarse.attn_1(h_coarse)
        h_coarse = self.mid_coarse.block_2(h_coarse, temb)
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = nonlinearity(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        h_median = self.mid_median.block_1(h_median, temb)
        h_median = self.mid_median.attn_1(h_median)
        h_median = self.mid_median.block_2(h_median, temb)
        h_median = self.norm_out_median(h_median)
        h_median = nonlinearity(h_median)
        h_median = self.conv_out_median(h_median)

        h_fine = self.mid_fine.block_1(h_fine, temb)
        h_fine = self.mid_fine.attn_1(h_fine)
        h_fine = self.mid_fine.block_2(h_fine, temb)
        h_fine = self.norm_out_fine(h_fine)
        h_fine = nonlinearity(h_fine)
        h_fine = self.conv_out_fine(h_fine)
        features[str(h_fine.shape[-1])] = h_fine
        features[str(h_median.shape[-1])] = h_median
        features[str(h_coarse.shape[-1])] = h_coarse

        gate_logits = self.router(h_fine=h_fine, h_median=h_median, h_coarse=h_coarse, entropy=x_entropy)

        # Use the straight-through hard Gumbel-Softmax gate directly in the
        # feature composition path. The old implementation converted the gate
        # to indices with argmax and selected features with torch.where, then
        # multiplied the selected feature by gate.max(). That weakens the
        # reconstruction gradient to the router and makes routing overly driven
        # by the budget loss. Here, hard=True keeps the forward route discrete,
        # while the backward pass follows the soft Gumbel-Softmax probabilities.
        if self.update_router and self.training:
            gate = F.gumbel_softmax(gate_logits, tau=1.0, dim=-1, hard=True)
        else:
            # Evaluation, or training with a frozen router, should still use a
            # deterministic hard route and must not treat raw logits as weights.
            gate_indices = gate_logits.argmax(dim=-1)
            gate = F.one_hot(gate_indices, num_classes=3).to(dtype=gate_logits.dtype)

        gate = gate.permute(0, 3, 1, 2).contiguous()  # B,3,Hc,Wc
        indices = gate.argmax(dim=1)

        h_coarse_up = h_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        h_median_up = h_median.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = indices.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2).unsqueeze(1)

        # Expand each coarse-cell routing decision to the full fine latent grid.
        # Because gate is hard one-hot in the forward pass, this is numerically
        # equivalent to discrete branch selection, but gradients from the
        # reconstruction loss can flow back to gate_logits through the ST path.
        gate_fine_grid = gate.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        h_triple = (
            gate_fine_grid[:, 0:1, :, :] * h_coarse_up
            + gate_fine_grid[:, 1:2, :, :] * h_median_up
            + gate_fine_grid[:, 2:3, :, :] * h_fine
        )

        coarse_mask = 0.0625 * torch.ones_like(indices_repeat, device=x.device)
        median_mask = 0.25 * torch.ones_like(indices_repeat, device=x.device)
        fine_mask = torch.ones_like(indices_repeat, device=x.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, median_mask)
        codebook_mask = torch.where(indices_repeat == 1, median_mask, codebook_mask)
        codebook_mask = torch.where(indices_repeat == 2, fine_mask, codebook_mask)
        out = {"h_triple": h_triple, "indices": indices, "codebook_mask": codebook_mask, "gate": gate,
               "gate_logits": gate_logits, "h_coarse": h_coarse, "h_median": h_median, "h_fine": h_fine}
        if return_features:
            out["features"] = features
        return out


def convert_to_coord_format(b: int, h: int, w: int, device="cpu", integer_values: bool = False) -> torch.Tensor:
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1)


class ConLinear(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, is_first: bool = False, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.ffm(x))


class FourierPositionEmbedding(nn.Module):
    # Exact DQ-VAE FourierPositionEmbedding.
    def __init__(self, coord_size: int, hidden_size: int, integer_values: bool = False):
        super().__init__()
        self.coord = convert_to_coord_format(1, coord_size, coord_size, "cpu", integer_values)
        self.lff = LFF(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coord = self.coord.to(x.device)
        fourier_features = self.lff(coord)
        return x + fourier_features


class PositionEmbedding2DLearned(nn.Module):
    def __init__(self, n_row: int, feats_dim: int, n_col: Optional[int] = None):
        super().__init__()
        n_col = n_col if n_col is not None else n_row
        self.row_embed = nn.Embedding(n_row, feats_dim)
        self.col_embed = nn.Embedding(n_col, feats_dim)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos


def make_grain_map(grain_indices: Optional[torch.Tensor], gate: Optional[torch.Tensor],
                   num_grains: int, size: Tuple[int, int], dtype: torch.dtype,
                   device: torch.device) -> Optional[torch.Tensor]:
    """Return B,num_grains,H,W grain condition from a soft gate or hard indices."""
    if gate is not None:
        grain_map = gate
        if grain_map.dim() != 4:
            raise ValueError(f"gate must be 4D, but got shape {tuple(grain_map.shape)}.")
        if grain_map.shape[1] != num_grains and grain_map.shape[-1] == num_grains:
            grain_map = grain_map.permute(0, 3, 1, 2).contiguous()
        if grain_map.shape[1] != num_grains:
            raise ValueError(
                f"gate channel dim must be {num_grains}, but got shape {tuple(grain_map.shape)}."
            )
        return F.interpolate(grain_map.to(device=device, dtype=dtype), size=size, mode="nearest")

    if grain_indices is None:
        return None
    if grain_indices.dim() == 4:
        if grain_indices.shape[1] == 1:
            grain_indices = grain_indices.squeeze(1)
        elif grain_indices.shape[1] == num_grains:
            return F.interpolate(grain_indices.to(device=device, dtype=dtype), size=size, mode="nearest")
        elif grain_indices.shape[-1] == num_grains:
            grain_map = grain_indices.permute(0, 3, 1, 2).contiguous()
            return F.interpolate(grain_map.to(device=device, dtype=dtype), size=size, mode="nearest")
        else:
            raise ValueError(
                f"grain_indices 4D tensor is incompatible with {num_grains} grains: "
                f"{tuple(grain_indices.shape)}."
            )
    if grain_indices.dim() != 3:
        raise ValueError(f"grain_indices must be B,H,W or B,1,H,W, but got {tuple(grain_indices.shape)}.")
    grain_indices = grain_indices.to(device=device).long().clamp(0, num_grains - 1)
    grain_map = F.one_hot(grain_indices, num_classes=num_grains).permute(0, 3, 1, 2).contiguous()
    return F.interpolate(grain_map.to(dtype=dtype), size=size, mode="nearest")


class GrainFiLMBlock(nn.Module):
    """Grain-conditioned feature modulation without encoder skip features."""
    def __init__(self, num_grains: int, channels: int):
        super().__init__()
        hidden_channels = min(128, max(32, channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(num_grains, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, grain_map: Optional[torch.Tensor]) -> torch.Tensor:
        if grain_map is None:
            return h
        grain_map = F.interpolate(grain_map.to(dtype=h.dtype, device=h.device), size=h.shape[-2:], mode="nearest")
        gamma, beta = self.net(grain_map).chunk(2, dim=1)
        return h * (1 + gamma) + beta


class DQDecoder(nn.Module):
    def __init__(self, ch: int, in_ch: int, out_ch: int, ch_mult: Tuple[int, ...],
                 num_res_blocks: int, resolution: int, attn_resolutions: List[int],
                 dropout: float = 0.0, resamp_with_conv: bool = True,
                 give_pre_end: bool = False, latent_size: int = 64,
                 window_size: int = 2, position_type: str = "fourier+learned",
                 num_grains: int = 3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.temb_ch = 0
        self.ch = ch
        self.give_pre_end = give_pre_end
        self.num_grains = num_grains
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, in_ch, curr_res, curr_res)
        self.conv_in = nn.Conv2d(in_ch, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        self.grain_film = nn.ModuleDict()
        self.decoder_channels: Dict[str, int] = {}
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            self.decoder_channels[str(curr_res)] = block_in
            self.grain_film[str(curr_res)] = GrainFiLMBlock(num_grains, block_in)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.position_type = position_type
        if position_type == "learned":
            self.position_bias = PositionEmbedding2DLearned(n_row=latent_size, feats_dim=in_ch)
        elif position_type == "fourier":
            self.position_bias = FourierPositionEmbedding(coord_size=latent_size, hidden_size=in_ch)
        elif position_type == "fourier+learned":
            self.position_bias_fourier = FourierPositionEmbedding(coord_size=latent_size, hidden_size=in_ch)
            self.position_bias_learned = PositionEmbedding2DLearned(n_row=latent_size, feats_dim=in_ch)
        else:
            raise NotImplementedError(f"Unsupported position_type: {position_type}")

    def add_position(self, h: torch.Tensor) -> torch.Tensor:
        if self.position_type in ("full", "fourier"):
            h = self.position_bias(h)
        elif self.position_type == "fourier+learned":
            h = self.position_bias_fourier(h)
            h = self.position_bias_learned(h)
        else:
            h = self.position_bias(h)
        return h

    def forward(self, h: torch.Tensor, grain_indices: Optional[torch.Tensor] = None,
                gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        grain_map = make_grain_map(grain_indices, gate, self.num_grains, h.shape[-2:], h.dtype, h.device)
        h = self.add_position(h)
        temb = None
        h = self.conv_in(h)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            h = self.grain_film[str(h.shape[-1])](h, grain_map)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


@ARCH_REGISTRY.register()
class DQDynamicVQVAE(nn.Module):
    """BasicSR-compatible DQ-VAE for CodeFormer stage-1 replacement."""
    def __init__(self, img_size: int = 512, grain_type: str = "triple", ch: int = 128,
                 ch_mult: Optional[List[int]] = None, num_res_blocks: int = 2,
                 attn_resolutions: Optional[List[int]] = None, dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int = 3,
                 z_channels: int = 256, codebook_size: int = 1024,
                 codebook_dim: int = 256, commitment_beta: float = 0.25,
                 decay: float = 0.99, restart_unused_codes: bool = True,
                 quant_before_dim: int = 256, quant_after_dim: int = 256,
                 quant_sample_temperature: float = 0.0,
                 router_config: Optional[dict] = None,
                 decoder_ch_mult: Optional[List[int]] = None,
                 decoder_attn_resolutions: Optional[List[int]] = None,
                 latent_size: Optional[int] = None,
                 decoder_position_type: str = "fourier+learned",
                 model_path: Optional[str] = None):
        super().__init__()
        assert grain_type in ("dual", "triple")
        self.img_size = img_size
        self.ch = ch
        self.grain_type = grain_type
        self.num_grains = 2 if grain_type == "dual" else 3
        self.codebook_size = codebook_size
        self.embed_dim = codebook_dim
        self.quant_sample_temperature = quant_sample_temperature
        if grain_type == "dual":
            ch_mult = ch_mult or [1, 1, 2, 2, 4]
            attn_resolutions = attn_resolutions or [img_size // 16, img_size // 8]
            router_config = router_config or {"num_channels": z_channels, "normalization_type": "group-32", "gate_type": "2layer-fc-SiLu"}
            self.encoder = DualGrainEncoder(ch=ch, ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
                                            attn_resolutions=attn_resolutions, dropout=dropout,
                                            resamp_with_conv=resamp_with_conv, in_channels=in_channels,
                                            resolution=img_size, z_channels=z_channels,
                                            router_config=router_config)
        else:
            ch_mult = ch_mult or [1, 1, 2, 2, 4, 4]
            attn_resolutions = attn_resolutions or [img_size // 32, img_size // 16, img_size // 8]
            router_config = router_config or {"num_channels": z_channels, "normalization_type": "group-32", "gate_type": "2layer-fc-SiLu"}
            self.encoder = TripleGrainEncoder(ch=ch, ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
                                              attn_resolutions=attn_resolutions, dropout=dropout,
                                              resamp_with_conv=resamp_with_conv, in_channels=in_channels,
                                              resolution=img_size, z_channels=z_channels,
                                              router_config=router_config)
        decoder_ch_mult = decoder_ch_mult or [1, 1, 2, 2]
        decoder_attn_resolutions = decoder_attn_resolutions or [img_size // 16]
        if latent_size is None:
            latent_size = img_size // (2 ** (len(decoder_ch_mult) - 1))
        self.latent_size = latent_size
        self.decoder = DQDecoder(ch=ch, in_ch=quant_before_dim, out_ch=3,
                                 ch_mult=tuple(decoder_ch_mult), num_res_blocks=num_res_blocks,
                                 resolution=img_size, attn_resolutions=decoder_attn_resolutions,
                                 dropout=dropout, resamp_with_conv=resamp_with_conv,
                                 latent_size=latent_size, window_size=2,
                                 position_type=decoder_position_type,
                                 num_grains=self.num_grains)
        self.quantize = VectorQuantize2(codebook_size=codebook_size, codebook_dim=codebook_dim,
                                        channel_last=False, accept_image_fmap=True,
                                        commitment_beta=commitment_beta, decay=decay,
                                        restart_unused_codes=restart_unused_codes)
        self.quant_conv = nn.Conv2d(quant_before_dim, quant_after_dim, 1)
        self.post_quant_conv = nn.Conv2d(quant_after_dim, quant_before_dim, 1)
        self.grain_latent_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(quant_before_dim, quant_before_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(quant_before_dim, quant_before_dim, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(self.num_grains)
        ])
        for adapter in self.grain_latent_adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        if model_path is not None:
            self.load_pretrained(model_path)

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        chkpt = torch.load(path, map_location="cpu")
        if "params_ema" in chkpt:
            sd = chkpt["params_ema"]
        elif "params" in chkpt:
            sd = chkpt["params"]
        elif "state_dict" in chkpt:
            sd = chkpt["state_dict"]
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        else:
            sd = chkpt
        self.load_state_dict(sd, strict=strict)

    def encode(self, x: torch.Tensor, return_features: bool = False):
        h_dict = self.encoder(x, None, return_features=return_features)
        key = "h_dual" if self.grain_type == "dual" else "h_triple"
        h = self.quant_conv(h_dict[key])
        quant, emb_loss, info = self.quantize(x=h, temp=self.quant_sample_temperature,
                                              codebook_mask=h_dict["codebook_mask"])
        return quant, emb_loss, info, h_dict["indices"], h_dict["gate"], h_dict

    def _apply_grain_latent_adapters(self, quant: torch.Tensor,
                                     grain_indices: Optional[torch.Tensor] = None,
                                     gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        grain_map = make_grain_map(grain_indices, gate, self.num_grains, quant.shape[-2:], quant.dtype, quant.device)
        if grain_map is None:
            return quant
        delta = torch.zeros_like(quant)
        for idx, adapter in enumerate(self.grain_latent_adapters):
            delta = delta + grain_map[:, idx:idx + 1, :, :] * adapter(quant)
        return quant + delta

    def decode(self, quant: torch.Tensor, grain_indices: Optional[torch.Tensor] = None,
               gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        quant = self.post_quant_conv(quant)
        quant = self._apply_grain_latent_adapters(quant, grain_indices=grain_indices, gate=gate)
        return self.decoder(quant, grain_indices=grain_indices, gate=gate)

    def decode_code(self, code_b: torch.Tensor, grain_indices: Optional[torch.Tensor] = None,
                    gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        quant_b = self.get_codebook_feat(code_b)
        return self.decode(quant_b, grain_indices=grain_indices, gate=gate)

    def get_codebook_feat(self, indices: torch.Tensor, shape: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        if shape is None and indices.dim() == 3:
            return self.quantize.embed_code(indices)
        if shape is not None:
            return self.quantize.get_codebook_entry(indices, shape=shape)
        return self.quantize.embed_code(indices)

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor):
        quant, emb_loss, info, grain_indices, gate, h_dict = self.encode(x, return_features=False)
        indices = info[2]
        if indices.dim() == 4 and indices.shape[-1] == 1:
            indices = indices.squeeze(-1)
        return indices.long(), grain_indices.long(), quant

    def forward(self, x: torch.Tensor):
        quant, diff, info, grain_indices, gate, _ = self.encode(x)
        dec = self.decode(quant, grain_indices=grain_indices, gate=gate)
        return dec, diff, {"min_encoding_indices": info[2], "grain_indices": grain_indices, "gate": gate}


# -----------------------------------------------------------------------------
# DQ discriminator and exact LPIPS/GAN loss support.
# -----------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@ARCH_REGISTRY.register()
class DQNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3,
                 use_actnorm: bool = False, model_path: Optional[str] = None):
        super().__init__()
        if use_actnorm:
            raise NotImplementedError("ActNorm is not needed for dqvae-dual-r-05_imagenet.yml (use_actnorm=false).")
        norm_layer = nn.BatchNorm2d
        use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence: List[nn.Module] = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                      padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)
        if model_path is not None:
            chkpt = torch.load(model_path, map_location="cpu")
            self.load_state_dict(chkpt.get("params", chkpt), strict=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        # self.conv3_out = nn.Conv2d(num_feat * 8, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        # out = torch.cat([out, F.interpolate(out_conv3, size=out.shape[2:], mode='nearest')], dim=1)

        return out


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers: List[nn.Module] = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class VGG16Slices(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x); h_relu1_2 = h
        h = self.slice2(h); h_relu2_2 = h
        h = self.slice3(h); h_relu3_3 = h
        h = self.slice4(h); h_relu4_3 = h
        h = self.slice5(h); h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class DQLPIPS(nn.Module):
    def __init__(self, lpips_weight_path: str = "weights/lpips/vgg.pth", use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16Slices(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        if os.path.isfile(lpips_weight_path):
            self.load_state_dict(torch.load(lpips_weight_path, map_location="cpu"), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        res = []
        for kk in range(len(self.chns)):
            feats0, feats1 = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs = (feats0 - feats1) ** 2
            res.append(spatial_average(lins[kk].model(diffs), keepdim=True))
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(logits_fake)


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    if global_step < threshold:
        weight = value
    return weight
