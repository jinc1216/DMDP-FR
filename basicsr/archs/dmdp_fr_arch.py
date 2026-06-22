# """Multi-granularity CodeFormer built on DQ-VAE stage-1.

# Stage-2 keeps CodeFormer's core idea (LQ encoder + global Transformer + code
# classification) but predicts DQ-VAE's variable-length code representation:
# coarse/median/fine content logits plus a supervised granularity route.  Stage-3
# adds a small grain-aware CFT/SFT fusion block to improve fidelity in information-
# dense regions while protecting quality in coarse regions.
# """
# from __future__ import annotations

# import math
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from basicsr.archs.dqvae_arch import DQDynamicVQVAE, ResnetBlock, nonlinearity
# from basicsr.utils.registry import ARCH_REGISTRY


# def calc_mean_std(feat: Tensor, eps: float = 1e-5):
#     size = feat.size()
#     assert len(size) == 4
#     b, c = size[:2]
#     feat_var = feat.view(b, c, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(b, c, 1, 1)
#     feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
#     return feat_mean, feat_std


# def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#     normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# def _get_activation_fn(activation: str):
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# class TransformerSALayer(nn.Module):
#     def __init__(self, embed_dim: int, nhead: int = 8, dim_mlp: int = 2048,
#                  dropout: float = 0.0, activation: str = "gelu"):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(embed_dim, dim_mlp)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_mlp, embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = _get_activation_fn(activation)

#     @staticmethod
#     def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None) -> Tensor:
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout2(tgt2)
#         return tgt


# class GrainAwareFuseSFTBlock(nn.Module):
#     """Minimal grain-aware CFT block for stage-3.

#     It is CodeFormer's SFT residual with one extra scalar conditioning map.  The
#     scalar map is high for fine-grained regions and low for coarse regions, so
#     LQ features are injected more strongly where stage-1 decided that more codes
#     are needed.
#     """
#     def __init__(self, enc_ch: int, dec_ch: int):
#         super().__init__()
#         self.encode = nn.Sequential(
#             nn.Conv2d(enc_ch + dec_ch + 1, dec_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.scale = nn.Sequential(
#             nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
#         )
#         self.shift = nn.Sequential(
#             nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
#         )

#     def forward(self, enc_feat: Tensor, dec_feat: Tensor, grain_scalar: Tensor, w: float = 1.0) -> Tensor:
#         if enc_feat.shape[-2:] != dec_feat.shape[-2:]:
#             enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
#         if grain_scalar.shape[-2:] != dec_feat.shape[-2:]:
#             grain_scalar = F.interpolate(grain_scalar, size=dec_feat.shape[-2:], mode="nearest")
#         fused_cond = self.encode(torch.cat([enc_feat, dec_feat, grain_scalar], dim=1))
#         scale = self.scale(fused_cond)
#         shift = self.shift(fused_cond)
#         # coarse≈0.25, median≈0.625, fine≈1.0 residual strength
#         strength = 0.25 + 0.75 * grain_scalar
#         residual = w * strength * (dec_feat * scale + shift)
#         return dec_feat + residual


# @ARCH_REGISTRY.register()
# class DMDPFR(DQDynamicVQVAE):
#     def __init__(self, dim_embd: int = 512, n_head: int = 8, n_layers: int = 9,
#                  codebook_size: int = 1024, connect_list: Optional[List[str]] = None,
#                  fix_modules: Optional[List[str]] = None, stage1_model_path: Optional[str] = None,
#                  vqgan_path: Optional[str] = None, max_position_tokens: int = 4096,
#                  **dqvae_kwargs):
#         dqvae_kwargs.setdefault("grain_type", "triple")
#         dqvae_kwargs.setdefault("codebook_size", codebook_size)
#         codebook_size = dqvae_kwargs["codebook_size"]
#         super().__init__(**dqvae_kwargs)
#         load_path = stage1_model_path or vqgan_path or dqvae_kwargs.get("model_path", None)
#         if load_path is not None:
#             # strict=False allows loading a pure DQ-VAE checkpoint into DMDPFR.
#             chkpt = torch.load(load_path, map_location="cpu")
#             if "params_ema" in chkpt:
#                 sd = chkpt["params_ema"]
#             elif "params" in chkpt:
#                 sd = chkpt["params"]
#             elif "state_dict" in chkpt:
#                 sd = {k.replace("model.", "", 1): v for k, v in chkpt["state_dict"].items()}
#             else:
#                 sd = chkpt
#             self.load_state_dict(sd, strict=False)

#         if fix_modules is None:
#             fix_modules = ["quantize", "decoder", "post_quant_conv"]
#         for module in fix_modules:
#             if hasattr(self, module):
#                 for p in getattr(self, module).parameters():
#                     p.requires_grad = False

#         self.connect_list = connect_list or [str(self.img_size // 8), str(self.img_size // 4), str(self.img_size // 2)]
#         self.n_layers = n_layers
#         self.dim_embd = dim_embd
#         self.dim_mlp = dim_embd * 2
#         self.max_position_tokens = max_position_tokens
#         self.position_emb = nn.Parameter(torch.zeros(max_position_tokens, dim_embd))
#         nn.init.trunc_normal_(self.position_emb, std=0.02)

#         self.feat_emb = nn.Linear(self.embed_dim, dim_embd)
#         self.ft_layers = nn.Sequential(*[
#             TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
#             for _ in range(n_layers)
#         ])
#         self.median_ft_layers = nn.Sequential(*[
#             TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
#             for _ in range(n_layers)
#         ])
#         self.fine_ft_layers = nn.Sequential(*[
#             TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
#             for _ in range(n_layers)
#         ])
#         gate_in_ch = dim_embd * (3 if self.grain_type == "triple" else 2)
#         self.gate_pred = nn.Sequential(
#             nn.Conv2d(gate_in_ch, dim_embd, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(dim_embd, 3 if self.grain_type == "triple" else 2, 1),
#         )
#         self.idx_pred_coarse = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))
#         self.median_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
#         self.fine_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
#         self.coarse_code_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
#         self.median_code_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
#         self.median_refine = nn.Sequential(nn.Conv2d(dim_embd, dim_embd, 3, padding=1), nn.GELU())
#         self.fine_refine = nn.Sequential(nn.Conv2d(dim_embd, dim_embd, 3, padding=1), nn.GELU())
#         self.idx_pred_median = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))
#         self.idx_pred_fine = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))

#         self.encoder_channels = self._infer_encoder_feature_channels()
#         self.fuse_convs_dict = nn.ModuleDict()
#         for f_size in self.connect_list:
#             dec_ch = self.decoder.decoder_channels.get(str(f_size), None)
#             enc_ch = self.encoder_channels.get(str(f_size), dec_ch)
#             if dec_ch is not None and enc_ch is not None:
#                 self.fuse_convs_dict[str(f_size)] = GrainAwareFuseSFTBlock(enc_ch, dec_ch)

#     def _infer_encoder_feature_channels(self) -> Dict[str, int]:
#         ch = self.ch
#         # Prefer actual defaults stored in modules if available; fall back to common DQ settings.
#         ch_mult = [1, 1, 2, 2, 4, 4] if self.grain_type == "triple" else [1, 1, 2, 2, 4]
#         channels: Dict[str, int] = {}
#         curr_res = self.img_size
#         for mult in ch_mult:
#             out_ch = ch * mult
#             channels[str(curr_res)] = out_ch
#             curr_res //= 2
#         # branch outputs overwrite lower-level backbone channels
#         channels[str(self.img_size // 8)] = self.embed_dim
#         channels[str(self.img_size // 16)] = self.embed_dim
#         if self.grain_type == "triple":
#             channels[str(self.img_size // 32)] = self.embed_dim
#         return channels

#     def _encode_lq(self, x: Tensor):
#         h_dict = self.encoder(x, None, return_features=True)
#         h_coarse = self.quant_conv(h_dict["h_coarse"])
#         h_fine = self.quant_conv(h_dict["h_fine"])
#         h_median = self.quant_conv(h_dict["h_median"]) if self.grain_type == "triple" else None
#         key = "h_triple" if self.grain_type == "triple" else "h_dual"
#         h_dynamic = self.quant_conv(h_dict[key])
#         return h_dict, h_coarse, h_median, h_fine, h_dynamic

#     def _transform_tokens(self, query_emb: Tensor, b: int, h: int, w: int,
#                           layers: nn.Module, level_name: str) -> Tensor:
#         token_num = h * w
#         if token_num > self.max_position_tokens:
#             raise ValueError(f"{level_name} token length {token_num} > max_position_tokens {self.max_position_tokens}")
#         pos_emb = self.position_emb[:token_num].to(
#             device=query_emb.device, dtype=query_emb.dtype).unsqueeze(1).repeat(1, b, 1)
#         for layer in layers:
#             query_emb = layer(query_emb, query_pos=pos_emb)
#         return query_emb.permute(1, 2, 0).view(b, self.dim_embd, h, w).contiguous()

#     def _transform_coarse(self, h_coarse: Tensor) -> Tensor:
#         b, c, h, w = h_coarse.shape
#         feat_emb = self.feat_emb(h_coarse.flatten(2).permute(2, 0, 1))
#         return self._transform_tokens(feat_emb, b, h, w, self.ft_layers, "coarse")

#     def _transform_context(self, context: Tensor, layers: nn.Module, level_name: str) -> Tensor:
#         b, c, h, w = context.shape
#         if c != self.dim_embd:
#             raise ValueError(f"{level_name} context has {c} channels, expected {self.dim_embd}")
#         query_emb = context.flatten(2).permute(2, 0, 1)
#         return self._transform_tokens(query_emb, b, h, w, layers, level_name)

#     def _soft_codebook_feature(self, logits: Tensor) -> Tensor:
#         probs = F.softmax(logits, dim=1)
#         codebook = self.quantize.embedding.weight[:self.codebook_size]
#         codebook = codebook.to(device=logits.device, dtype=probs.dtype)
#         return torch.einsum("b n h w, n c -> b c h w", probs, codebook)

#     @staticmethod
#     def _pool_to_context(feat: Tensor, context: Tensor) -> Tensor:
#         if feat.shape[-2:] == context.shape[-2:]:
#             return feat
#         return F.adaptive_avg_pool2d(feat, context.shape[-2:])

#     def _predict_gate(self, coarse_ctx: Tensor, fine_ctx: Tensor,
#                       median_ctx: Optional[Tensor] = None) -> Tensor:
#         if self.grain_type == "triple":
#             if median_ctx is None:
#                 raise ValueError("median_ctx is required for triple-grain route prediction.")
#             gate_ctx = torch.cat([
#                 coarse_ctx,
#                 self._pool_to_context(median_ctx, coarse_ctx),
#                 self._pool_to_context(fine_ctx, coarse_ctx),
#             ], dim=1)
#         else:
#             gate_ctx = torch.cat([
#                 coarse_ctx,
#                 self._pool_to_context(fine_ctx, coarse_ctx),
#             ], dim=1)
#         return self.gate_pred(gate_ctx)

#     def _predict_multigrain_logits(self, h_coarse: Tensor, h_median: Optional[Tensor], h_fine: Tensor):
#         coarse_ctx = self._transform_coarse(h_coarse)
#         logits_coarse = self.idx_pred_coarse(coarse_ctx)
#         coarse_code_ctx = self.coarse_code_proj(self._soft_codebook_feature(logits_coarse))
#         if self.grain_type == "dual":
#             fine_seed = self.fine_refine(
#                 self.fine_proj(h_fine)
#                 + F.interpolate(coarse_ctx, size=h_fine.shape[-2:], mode="nearest")
#                 + F.interpolate(coarse_code_ctx, size=h_fine.shape[-2:], mode="nearest")
#             )
#             fine_ctx = self._transform_context(fine_seed, self.fine_ft_layers, "fine")
#             logits_fine = self.idx_pred_fine(fine_ctx)
#             gate_logits = self._predict_gate(coarse_ctx, fine_ctx)
#             return {"gate": gate_logits, "coarse": logits_coarse, "fine": logits_fine,
#                     "coarse_ctx": coarse_ctx, "fine_ctx": fine_ctx}
#         median_seed = self.median_refine(
#             self.median_proj(h_median)
#             + F.interpolate(coarse_ctx, size=h_median.shape[-2:], mode="nearest")
#             + F.interpolate(coarse_code_ctx, size=h_median.shape[-2:], mode="nearest")
#         )
#         median_ctx = self._transform_context(median_seed, self.median_ft_layers, "median")
#         logits_median = self.idx_pred_median(median_ctx)
#         median_code_ctx = self.median_code_proj(self._soft_codebook_feature(logits_median))
#         fine_seed = self.fine_refine(
#             self.fine_proj(h_fine)
#             + F.interpolate(coarse_ctx, size=h_fine.shape[-2:], mode="nearest")
#             + F.interpolate(coarse_code_ctx, size=h_fine.shape[-2:], mode="nearest")
#             + F.interpolate(median_ctx, size=h_fine.shape[-2:], mode="nearest")
#             + F.interpolate(median_code_ctx, size=h_fine.shape[-2:], mode="nearest")
#         )
#         fine_ctx = self._transform_context(fine_seed, self.fine_ft_layers, "fine")
#         logits_fine = self.idx_pred_fine(fine_ctx)
#         gate_logits = self._predict_gate(coarse_ctx, fine_ctx, median_ctx)
#         return {"gate": gate_logits, "coarse": logits_coarse, "median": logits_median, "fine": logits_fine,
#                 "coarse_ctx": coarse_ctx, "median_ctx": median_ctx, "fine_ctx": fine_ctx,
#                 "coarse_code_ctx": coarse_code_ctx, "median_code_ctx": median_code_ctx}

#     @staticmethod
#     def assemble_triple_code_map(code_coarse: Tensor, code_median: Tensor, code_fine: Tensor, grain_idx: Tensor) -> Tensor:
#         code_full = code_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
#         median_full = code_median.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
#         median_mask = (grain_idx == 1).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
#         fine_mask = (grain_idx == 2).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
#         code_full = torch.where(median_mask, median_full, code_full)
#         code_full = torch.where(fine_mask, code_fine, code_full)
#         return code_full

#     @staticmethod
#     def assemble_dual_code_map(code_coarse: Tensor, code_fine: Tensor, grain_idx: Tensor) -> Tensor:
#         code_full = code_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
#         fine_mask = (grain_idx == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
#         code_full = torch.where(fine_mask, code_fine, code_full)
#         return code_full

#     def logits_to_code_map(self, pred: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
#         gate_prob = F.softmax(pred["gate"], dim=1)
#         grain_idx = gate_prob.argmax(dim=1)
#         code_coarse = pred["coarse"].argmax(dim=1)
#         code_fine = pred["fine"].argmax(dim=1)
#         if self.grain_type == "dual":
#             code_full = self.assemble_dual_code_map(code_coarse, code_fine, grain_idx)
#         else:
#             code_median = pred["median"].argmax(dim=1)
#             code_full = self.assemble_triple_code_map(code_coarse, code_median, code_fine, grain_idx)
#         return code_full.long(), grain_idx.long(), gate_prob

#     def _grain_scalar_from_prob(self, gate_prob: Tensor) -> Tensor:
#         if self.grain_type == "dual":
#             return gate_prob[:, 1:2]
#         # 0 coarse, 0.5 median, 1 fine
#         return 0.5 * gate_prob[:, 1:2] + gate_prob[:, 2:3]

#     def decode_with_fusion(self, quant: Tensor, enc_feat_dict: Optional[Dict[str, Tensor]] = None,
#                            gate_prob: Optional[Tensor] = None, w: float = 0.0) -> Tensor:
#         h = self.post_quant_conv(quant)
#         h = self.decoder.add_position(h)
#         temb = None
#         h = self.decoder.conv_in(h)
#         h = self.decoder.mid.block_1(h, temb)
#         h = self.decoder.mid.attn_1(h)
#         h = self.decoder.mid.block_2(h, temb)
#         grain_scalar = self._grain_scalar_from_prob(gate_prob) if gate_prob is not None else None
#         for i_level in reversed(range(self.decoder.num_resolutions)):
#             for i_block in range(self.decoder.num_res_blocks + 1):
#                 h = self.decoder.up[i_level].block[i_block](h, temb)
#                 if len(self.decoder.up[i_level].attn) > 0:
#                     h = self.decoder.up[i_level].attn[i_block](h)
#             f_size = str(h.shape[-1])
#             if w > 0 and enc_feat_dict is not None and grain_scalar is not None and f_size in self.fuse_convs_dict and f_size in enc_feat_dict:
#                 h = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), h, grain_scalar, w=w)
#             if i_level != 0:
#                 h = self.decoder.up[i_level].upsample(h)
#         h = self.decoder.norm_out(h)
#         h = nonlinearity(h)
#         h = self.decoder.conv_out(h)
#         return h

#     @torch.no_grad()
#     def encode_to_indices(self, x: Tensor):
#         # For frozen GT extraction, use the inherited DQ-VAE encoder/quantizer.
#         return super().encode_to_indices(x)

#     def forward(self, x: Tensor, w: float = 0.0, detach_quant: bool = True,
#                 code_only: bool = False, adain: bool = False):
#         h_dict, h_coarse, h_median, h_fine, h_dynamic = self._encode_lq(x)
#         pred = self._predict_multigrain_logits(h_coarse, h_median, h_fine)
#         if code_only:
#             return pred, h_dynamic
#         code_full, grain_idx, gate_prob = self.logits_to_code_map(pred)
#         quant_feat = self.get_codebook_feat(code_full)
#         if detach_quant:
#             quant_feat = quant_feat.detach()
#         if adain:
#             quant_feat = adaptive_instance_normalization(quant_feat, h_dynamic)
#         out = self.decode_with_fusion(quant_feat, enc_feat_dict=h_dict.get("features", None), gate_prob=gate_prob, w=w)
#         return out, pred, h_dynamic

#     def get_last_fusion_layer(self):
#         if len(self.fuse_convs_dict) == 0:
#             return self.decoder.conv_out.weight
#         # choose the largest spatial fusion block, matching CodeFormer's stage-3 adaptive weight practice.
#         key = sorted(self.fuse_convs_dict.keys(), key=lambda x: int(x))[-1]
#         return self.fuse_convs_dict[key].shift[-1].weight


# # -----------------------------------------------------------------------------
# # Target construction helpers used by DMDPFRModel.
# # -----------------------------------------------------------------------------

# def make_triple_targets(full_indices: Tensor, grain_indices: Tensor) -> Dict[str, Tensor]:
#     """Build selected-scale code targets from a full fine code map.

#     full_indices: B,Hf,Wf where Hf/Wf correspond to f=8.
#     grain_indices: B,Hc,Wc where values are 0(coarse f=32), 1(median f=16), 2(fine f=8).
#     """
#     coarse_t = full_indices[:, ::4, ::4].contiguous()
#     median_t = full_indices[:, ::2, ::2].contiguous()
#     fine_t = full_indices.contiguous()
#     mask_coarse = (grain_indices == 0)
#     mask_median = (grain_indices == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
#     mask_fine = (grain_indices == 2).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
#     return {"coarse": coarse_t.long(), "median": median_t.long(), "fine": fine_t.long(),
#             "mask_coarse": mask_coarse, "mask_median": mask_median, "mask_fine": mask_fine}


# def make_dual_targets(full_indices: Tensor, grain_indices: Tensor) -> Dict[str, Tensor]:
#     coarse_t = full_indices[:, ::2, ::2].contiguous()
#     fine_t = full_indices.contiguous()
#     mask_coarse = (grain_indices == 0)
#     mask_fine = (grain_indices == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
#     return {"coarse": coarse_t.long(), "fine": fine_t.long(),
#             "mask_coarse": mask_coarse, "mask_fine": mask_fine}




"""Multi-granularity CodeFormer built on DQ-VAE stage-1.

Stage-2 keeps CodeFormer's core idea (LQ encoder + global Transformer + code
classification) but predicts DQ-VAE's variable-length code representation:
coarse/median/fine content logits plus a supervised granularity route. Stage-3
uses CodeFormer's SFT feature fusion style on top of the DQ-VAE decoder prior.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from basicsr.archs.dqvae_arch import DQDynamicVQVAE, make_grain_map, nonlinearity
from basicsr.archs.vqgan_arch import ResBlock
from basicsr.utils.registry import ARCH_REGISTRY


def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    size = feat.size()
    assert len(size) == 4
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int = 8, dim_mlp: int = 2048,
                 dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # The attention weights are never used; disabling them reduces memory
        # traffic and enables optimized attention kernels in recent PyTorch.
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class GranularityAwareFuseSFTBlock(nn.Module):
    """Granularity-aware multi-scale fusion (GAMF) block for Stage-III.

    This is the DMDP-FR counterpart of CodeFormer's residual SFT block,
    but follows the DMDP-FR Stage-III equations:

        (alpha_l, beta_l) = P_l([Gamma_l * F_e_l || F_d_l])
        F_ed_l = F_d_l + w * Gamma_l * (alpha_l * F_d_l + beta_l)

    ``Gamma_l`` is a scalar spatial map produced from the predicted routing
    distribution.  It is not a learnable parameter and therefore adds no new
    trainable capacity beyond the original SFT heads.  The module keeps the
    historical attribute names ``condition``, ``scale_head`` and ``shift_head``
    so older DMDP-FR checkpoints can still be loaded with ``strict=False``.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.condition = ResBlock(in_ch + out_ch, out_ch)
        self.scale_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        # Fixed, non-learnable bounds for the Stage-III residual branch.
        # V2 used residual_scale=0.25 as a very conservative safety bound.  In
        # practice this can make the effective GAMF residual too weak because
        # it is multiplied by both w and Gamma.  We keep the residual bounded
        # with tanh, but raise the bound and add a small Gamma floor so coarse
        # regions are still allowed to receive a controlled fidelity correction.
        # No trainable parameters or required configuration items are added.
        self.residual_scale = 0.5
        self.gamma_floor = 0.25
        self._zero_init_last_affine_layers()

    def _zero_init_last_affine_layers(self) -> None:
        # Identity start: with zero alpha/beta the frozen DQ prior is unchanged
        # at the beginning of Stage-III. This avoids a randomly initialized SFT
        # branch disturbing the Stage-II predicted prior.
        for head in (self.scale_head, self.shift_head):
            last = head[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def reset_parameters(self) -> None:
        # Used by DMDP-FR.reset_fuse_sft_parameters() when starting a fresh
        # Stage-III run from an older Stage-II checkpoint.
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self._zero_init_last_affine_layers()

    @staticmethod
    def _resize_like(feat: Tensor, ref: Tensor, mode: str) -> Tensor:
        if feat.shape[-2:] == ref.shape[-2:]:
            return feat
        if mode == "nearest":
            return F.interpolate(feat, size=ref.shape[-2:], mode="nearest")
        return F.interpolate(feat, size=ref.shape[-2:], mode=mode, align_corners=False)

    def forward(self, enc_feat: Tensor, dec_feat: Tensor, gamma: Optional[Tensor] = None,
                w: float = 1.0) -> Tensor:
        if enc_feat.shape[-2:] != dec_feat.shape[-2:]:
            enc_feat = self._resize_like(enc_feat, dec_feat, mode="bilinear")
        if enc_feat.dtype != dec_feat.dtype or enc_feat.device != dec_feat.device:
            enc_feat = enc_feat.to(device=dec_feat.device, dtype=dec_feat.dtype)

        if gamma is None:
            gamma_l = torch.ones(dec_feat.shape[0], 1, *dec_feat.shape[-2:],
                                 device=dec_feat.device, dtype=dec_feat.dtype)
        else:
            gamma_l = gamma
            if gamma_l.dim() == 3:
                gamma_l = gamma_l.unsqueeze(1)
            if gamma_l.shape[1] != 1:
                # Keep Eq. (15)'s conditioning map scalar-valued even if a caller
                # accidentally provides a multi-channel map.
                gamma_l = gamma_l.mean(dim=1, keepdim=True)
            gamma_l = self._resize_like(gamma_l, dec_feat, mode="nearest")
            gamma_l = gamma_l.to(device=dec_feat.device, dtype=dec_feat.dtype).clamp(0.0, 1.0)

        # Keep the relative routing order from Eq. (15) while avoiding an almost
        # closed fusion gate in coarse/mixed regions.  This makes GAMF act as a
        # controlled detail-restoration path rather than a nearly disabled SFT
        # branch whenever the predicted route is not confidently fine.
        gamma_eff = self.gamma_floor + (1.0 - self.gamma_floor) * gamma_l

        cond = self.condition(torch.cat([gamma_eff * enc_feat, dec_feat], dim=1))

        # Bounded affine residual.  The original paper defines alpha_l and beta_l
        # as outputs of P_l; using tanh keeps the same mathematical form while
        # preventing unbounded over-sharpening, ringing, or washed-out smoothing
        # from the newly trained Stage-III fusion branch.  The larger fixed bound
        # keeps the residual strong enough to inject visible route-guided details.
        alpha = torch.tanh(self.scale_head(cond)) * self.residual_scale
        beta = torch.tanh(self.shift_head(cond)) * self.residual_scale
        residual = alpha * dec_feat + beta
        return dec_feat + float(w) * gamma_eff * residual


# Backwards-compatible name used by older code/checkpoints.
CodeFormerFuseSFTBlock = GranularityAwareFuseSFTBlock

@ARCH_REGISTRY.register()
class DMDPFR(DQDynamicVQVAE):
    def __init__(self, dim_embd: int = 512, n_head: int = 8, n_layers: int = 9,
                 codebook_size: int = 1024, connect_list: Optional[List[str]] = None,
                 fix_modules: Optional[List[str]] = None, stage1_model_path: Optional[str] = None,
                 vqgan_path: Optional[str] = None, max_position_tokens: int = 4096,
                 **dqvae_kwargs):
        dqvae_kwargs.setdefault("grain_type", "triple")
        dqvae_kwargs.setdefault("codebook_size", codebook_size)
        codebook_size = dqvae_kwargs["codebook_size"]
        encoder_ch_mult = dqvae_kwargs.get("ch_mult", None)
        super().__init__(**dqvae_kwargs)
        if encoder_ch_mult is None:
            encoder_ch_mult = [1, 1, 2, 2, 4, 4] if self.grain_type == "triple" else [1, 1, 2, 2, 4]
        self.encoder_ch_mult = tuple(encoder_ch_mult)
        load_path = stage1_model_path or vqgan_path or dqvae_kwargs.get("model_path", None)
        if load_path is not None:
            # strict=False allows loading a pure DQ-VAE checkpoint into DMDPFR.
            chkpt = torch.load(load_path, map_location="cpu")
            if "params_ema" in chkpt:
                sd = chkpt["params_ema"]
            elif "params" in chkpt:
                sd = chkpt["params"]
            elif "state_dict" in chkpt:
                sd = {k.replace("model.", "", 1): v for k, v in chkpt["state_dict"].items()}
            else:
                sd = chkpt
            self.load_state_dict(sd, strict=False)

        if fix_modules is None:
            fix_modules = ["quantize", "decoder", "post_quant_conv"]
        for module in fix_modules:
            if hasattr(self, module):
                for p in getattr(self, module).parameters():
                    p.requires_grad = False

        if connect_list is None:
            # For DMDP-FR, the latent-resolution decoder scale (32x32 for
            # the triple 512px setup) is the assembled multi-grain code/route
            # prior rather than a normal intermediate feature.  Keep that core
            # prior clean and start SFT fusion from the next decoder scale.
            latent_res = int(getattr(self, "latent_size", 0) or 0)
            self.connect_list = [
                k for k in sorted(self.decoder.decoder_channels.keys(), key=lambda x: int(x))
                if latent_res < int(k) < self.img_size
            ]
        else:
            self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2
        self.max_position_tokens = max_position_tokens
        self.position_emb = nn.Parameter(torch.zeros(max_position_tokens, dim_embd))
        nn.init.trunc_normal_(self.position_emb, std=0.02)

        self.feat_emb = nn.Linear(self.embed_dim, dim_embd)
        self.ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.median_ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.fine_ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(n_layers)
        ])
        gate_in_ch = dim_embd * (3 if self.grain_type == "triple" else 2)
        self.gate_pred = nn.Sequential(
            nn.Conv2d(gate_in_ch, dim_embd, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_embd, 3 if self.grain_type == "triple" else 2, 1),
        )
        self.idx_pred_coarse = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))
        self.median_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
        self.fine_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
        self.coarse_code_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
        self.median_code_proj = nn.Conv2d(self.embed_dim, dim_embd, 1)
        self.median_refine = nn.Sequential(nn.Conv2d(dim_embd, dim_embd, 3, padding=1), nn.GELU())
        self.fine_refine = nn.Sequential(nn.Conv2d(dim_embd, dim_embd, 3, padding=1), nn.GELU())
        self.idx_pred_median = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))
        self.idx_pred_fine = nn.Sequential(nn.GroupNorm(32, dim_embd), nn.Conv2d(dim_embd, codebook_size, 1))

        self.encoder_channels = self._infer_encoder_feature_channels()
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            dec_ch = self.decoder.decoder_channels.get(str(f_size), None)
            enc_ch = self.encoder_channels.get(str(f_size), dec_ch)
            if dec_ch is not None and enc_ch is not None:
                self.fuse_convs_dict[str(f_size)] = GranularityAwareFuseSFTBlock(enc_ch, dec_ch)

    def _infer_encoder_feature_channels(self) -> Dict[str, int]:
        """Infer feature channels produced by the DQ encoder at each resolution.

        The original helper hard-coded a six-level encoder and marked 64/32/16
        as fine/median/coarse branch outputs.  The provided triple stage-1/2
        configs actually use seven encoder levels, giving a full latent grid of
        32x32 and a route grid of 8x8.  This routine derives the resolutions from
        the configured encoder depth so stage-3 fusion stays aligned with the
        stage-1/stage-2 checkpoint.
        """
        channels: Dict[str, int] = {}
        curr_res = self.img_size
        for i_level, mult in enumerate(self.encoder_ch_mult):
            channels[str(curr_res)] = self.ch * mult
            if i_level != len(self.encoder_ch_mult) - 1:
                curr_res //= 2

        num_res = len(self.encoder_ch_mult)
        coarse_res = self.img_size // (2 ** (num_res - 1))
        channels[str(coarse_res)] = self.embed_dim
        if self.grain_type == "triple":
            median_res = self.img_size // (2 ** (num_res - 2))
            fine_res = self.img_size // (2 ** (num_res - 3))
            channels[str(median_res)] = self.embed_dim
            channels[str(fine_res)] = self.embed_dim
        else:
            fine_res = self.img_size // (2 ** (num_res - 2))
            channels[str(fine_res)] = self.embed_dim
        return channels

    def _encode_lq(self, x: Tensor):
        h_dict = self.encoder(x, None, return_features=True)
        h_coarse = self.quant_conv(h_dict["h_coarse"])
        h_fine = self.quant_conv(h_dict["h_fine"])
        h_median = self.quant_conv(h_dict["h_median"]) if self.grain_type == "triple" else None
        key = "h_triple" if self.grain_type == "triple" else "h_dual"
        h_dynamic = self.quant_conv(h_dict[key])
        return h_dict, h_coarse, h_median, h_fine, h_dynamic

    def _transform_tokens(self, query_emb: Tensor, b: int, h: int, w: int,
                          layers: nn.Module, level_name: str) -> Tensor:
        token_num = h * w
        if token_num > self.max_position_tokens:
            raise ValueError(f"{level_name} token length {token_num} > max_position_tokens {self.max_position_tokens}")
        pos_emb = self.position_emb[:token_num].to(
            device=query_emb.device, dtype=query_emb.dtype).unsqueeze(1).expand(-1, b, -1)
        for layer in layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        return query_emb.permute(1, 2, 0).view(b, self.dim_embd, h, w).contiguous()

    def _transform_coarse(self, h_coarse: Tensor) -> Tensor:
        b, c, h, w = h_coarse.shape
        feat_emb = self.feat_emb(h_coarse.flatten(2).permute(2, 0, 1))
        return self._transform_tokens(feat_emb, b, h, w, self.ft_layers, "coarse")

    def _transform_context(self, context: Tensor, layers: nn.Module, level_name: str) -> Tensor:
        b, c, h, w = context.shape
        if c != self.dim_embd:
            raise ValueError(f"{level_name} context has {c} channels, expected {self.dim_embd}")
        query_emb = context.flatten(2).permute(2, 0, 1)
        return self._transform_tokens(query_emb, b, h, w, layers, level_name)

    def _soft_codebook_feature(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=1)
        codebook = self.quantize.embedding.weight[:self.codebook_size]
        codebook = codebook.to(device=logits.device, dtype=probs.dtype)
        return torch.einsum("b n h w, n c -> b c h w", probs, codebook)

    @staticmethod
    def _pool_to_context(feat: Tensor, context: Tensor) -> Tensor:
        if feat.shape[-2:] == context.shape[-2:]:
            return feat
        return F.adaptive_avg_pool2d(feat, context.shape[-2:])

    def _predict_gate(self, coarse_ctx: Tensor, fine_ctx: Tensor,
                      median_ctx: Optional[Tensor] = None) -> Tensor:
        if self.grain_type == "triple":
            if median_ctx is None:
                raise ValueError("median_ctx is required for triple-grain route prediction.")
            gate_ctx = torch.cat([
                coarse_ctx,
                self._pool_to_context(median_ctx, coarse_ctx),
                self._pool_to_context(fine_ctx, coarse_ctx),
            ], dim=1)
        else:
            gate_ctx = torch.cat([
                coarse_ctx,
                self._pool_to_context(fine_ctx, coarse_ctx),
            ], dim=1)
        return self.gate_pred(gate_ctx)

    def _predict_multigrain_logits(self, h_coarse: Tensor, h_median: Optional[Tensor], h_fine: Tensor):
        coarse_ctx = self._transform_coarse(h_coarse)
        logits_coarse = self.idx_pred_coarse(coarse_ctx)
        coarse_code_ctx = self.coarse_code_proj(self._soft_codebook_feature(logits_coarse))
        if self.grain_type == "dual":
            fine_seed = self.fine_refine(
                self.fine_proj(h_fine)
                + F.interpolate(coarse_ctx, size=h_fine.shape[-2:], mode="nearest")
                + F.interpolate(coarse_code_ctx, size=h_fine.shape[-2:], mode="nearest")
            )
            fine_ctx = self._transform_context(fine_seed, self.fine_ft_layers, "fine")
            logits_fine = self.idx_pred_fine(fine_ctx)
            gate_logits = self._predict_gate(coarse_ctx, fine_ctx)
            return {"gate": gate_logits, "coarse": logits_coarse, "fine": logits_fine,
                    "coarse_ctx": coarse_ctx, "fine_ctx": fine_ctx}
        median_seed = self.median_refine(
            self.median_proj(h_median)
            + F.interpolate(coarse_ctx, size=h_median.shape[-2:], mode="nearest")
            + F.interpolate(coarse_code_ctx, size=h_median.shape[-2:], mode="nearest")
        )
        median_ctx = self._transform_context(median_seed, self.median_ft_layers, "median")
        logits_median = self.idx_pred_median(median_ctx)
        median_code_ctx = self.median_code_proj(self._soft_codebook_feature(logits_median))
        fine_seed = self.fine_refine(
            self.fine_proj(h_fine)
            + F.interpolate(coarse_ctx, size=h_fine.shape[-2:], mode="nearest")
            + F.interpolate(coarse_code_ctx, size=h_fine.shape[-2:], mode="nearest")
            + F.interpolate(median_ctx, size=h_fine.shape[-2:], mode="nearest")
            + F.interpolate(median_code_ctx, size=h_fine.shape[-2:], mode="nearest")
        )
        fine_ctx = self._transform_context(fine_seed, self.fine_ft_layers, "fine")
        logits_fine = self.idx_pred_fine(fine_ctx)
        gate_logits = self._predict_gate(coarse_ctx, fine_ctx, median_ctx)
        return {"gate": gate_logits, "coarse": logits_coarse, "median": logits_median, "fine": logits_fine,
                "coarse_ctx": coarse_ctx, "median_ctx": median_ctx, "fine_ctx": fine_ctx,
                "coarse_code_ctx": coarse_code_ctx, "median_code_ctx": median_code_ctx}

    @staticmethod
    def assemble_triple_code_map(code_coarse: Tensor, code_median: Tensor, code_fine: Tensor, grain_idx: Tensor) -> Tensor:
        code_full = code_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        median_full = code_median.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        median_mask = (grain_idx == 1).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        fine_mask = (grain_idx == 2).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
        code_full = torch.where(median_mask, median_full, code_full)
        code_full = torch.where(fine_mask, code_fine, code_full)
        return code_full

    @staticmethod
    def assemble_dual_code_map(code_coarse: Tensor, code_fine: Tensor, grain_idx: Tensor) -> Tensor:
        code_full = code_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        fine_mask = (grain_idx == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        code_full = torch.where(fine_mask, code_fine, code_full)
        return code_full

    def logits_to_code_map(self, pred: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        gate_prob = F.softmax(pred["gate"], dim=1)
        grain_idx = gate_prob.argmax(dim=1)
        code_coarse = pred["coarse"].argmax(dim=1)
        code_fine = pred["fine"].argmax(dim=1)
        if self.grain_type == "dual":
            code_full = self.assemble_dual_code_map(code_coarse, code_fine, grain_idx)
        else:
            code_median = pred["median"].argmax(dim=1)
            code_full = self.assemble_triple_code_map(code_coarse, code_median, code_fine, grain_idx)
        return code_full.long(), grain_idx.long(), gate_prob

    @staticmethod
    def _resize_nearest(feat: Tensor, size: Tuple[int, int]) -> Tensor:
        if tuple(feat.shape[-2:]) == tuple(size):
            return feat
        return F.interpolate(feat, size=size, mode="nearest")

    def _straight_through_gate(self, gate_prob: Tensor, grain_idx: Tensor) -> Tensor:
        gate_hard = F.one_hot(grain_idx.long(), num_classes=self.num_grains).permute(0, 3, 1, 2).contiguous()
        gate_hard = gate_hard.to(device=gate_prob.device, dtype=gate_prob.dtype)
        return gate_prob + (gate_hard - gate_prob).detach()

    def _ensure_gate_bchw(self, gate: Tensor) -> Tensor:
        """Return gate logits/probabilities as B,num_grains,H,W."""
        if gate.dim() != 4:
            raise ValueError(f"gate must be 4D, but got shape {tuple(gate.shape)}.")
        if gate.shape[1] == self.num_grains:
            return gate
        if gate.shape[-1] == self.num_grains:
            return gate.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"gate must have {self.num_grains} channels, but got shape {tuple(gate.shape)}."
        )

    def _granularity_eta(self, pred: Optional[Dict[str, Tensor]], device: torch.device,
                         dtype: torch.dtype) -> Tensor:
        """Compute eta_m=(f_1/f_m)^2 in this implementation's route order.

        DMDP-FR predicts routes in coarse->fine order for dual and
        coarse->median->fine order for triple.  Since H_m = H / f_m, the paper's
        eta_m=(f_1/f_m)^2 is equal to the area ratio between the m-th code grid
        and the finest code grid:

            eta_m = (H_m / H_1) * (W_m / W_1).

        The values are inferred from existing Stage-I/II code-logit resolutions,
        so no new architectural/configuration parameter is introduced.
        """
        if pred is not None:
            keys = ["coarse", "fine"] if self.grain_type == "dual" else ["coarse", "median", "fine"]
            if all(k in pred for k in keys):
                fine_h, fine_w = pred[keys[-1]].shape[-2:]
                eta = []
                for key in keys:
                    h, w = pred[key].shape[-2:]
                    eta.append((float(h) / float(fine_h)) * (float(w) / float(fine_w)))
                return torch.tensor(eta, device=device, dtype=dtype)

        eta = [0.25, 1.0] if self.grain_type == "dual" else [1.0 / 16.0, 0.25, 1.0]
        return torch.tensor(eta, device=device, dtype=dtype)

    def build_gamf_gamma(self, pred: Dict[str, Tensor], gate_prob: Optional[Tensor] = None) -> Tensor:
        """Build the Stage-III granularity-aware conditioning map Gamma.

        Args:
            pred: prediction dictionary containing routing logits ``pred['gate']``
                and per-granularity code logits.
            gate_prob: optional routing probabilities in B,G,H,W order.  When it
                is not supplied, ``softmax(pred['gate'])`` is used.

        Returns:
            Tensor: B,1,H_route,W_route scalar Gamma map. For the standard
            triple layout {coarse, median, fine}, eta is {1/16, 1/4, 1}.
        """
        if gate_prob is None:
            gate_prob = F.softmax(self._ensure_gate_bchw(pred["gate"]), dim=1)
        else:
            gate_prob = self._ensure_gate_bchw(gate_prob)
        eta = self._granularity_eta(pred, device=gate_prob.device, dtype=gate_prob.dtype)
        gamma = (gate_prob * eta.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        return gamma.clamp(0.0, 1.0)

    def _assemble_soft_quant_feat(self, pred: Dict[str, Tensor], gate_prob: Tensor) -> Tensor:
        """Assemble a differentiable soft DQ latent map at the full fine grid.

        Forward restoration still uses the hard top-1 code via a straight-through
        estimator in ``logits_to_quant_feat``.  The soft assembly exists only for
        gradients from pixel/perceptual/GAN losses to reach the stage-2 code and
        route predictors during stage-3.
        """
        full_size = pred["fine"].shape[-2:]
        gate_full = self._resize_nearest(gate_prob, full_size)
        coarse_feat = self._resize_nearest(self._soft_codebook_feature(pred["coarse"]), full_size)
        fine_feat = self._soft_codebook_feature(pred["fine"])
        if self.grain_type == "dual":
            return gate_full[:, 0:1] * coarse_feat + gate_full[:, 1:2] * fine_feat
        median_feat = self._resize_nearest(self._soft_codebook_feature(pred["median"]), full_size)
        return (gate_full[:, 0:1] * coarse_feat
                + gate_full[:, 1:2] * median_feat
                + gate_full[:, 2:3] * fine_feat)

    def logits_to_quant_feat(self, pred: Dict[str, Tensor], detach_quant: bool = True) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Convert predicted multi-grain logits to a DQ latent feature map.

        Returns:
            quant_feat: full-resolution latent map consumed by the DQ decoder.
            grain_idx: hard route indices on the coarse route grid.
            gate_cond: optional straight-through gate for differentiable stage-3
                image-level training; ``None`` means use ``grain_idx``.
        """
        code_full, grain_idx, gate_prob = self.logits_to_code_map(pred)
        quant_hard = self.get_codebook_feat(code_full)
        if detach_quant:
            return quant_hard.detach(), grain_idx, None
        quant_soft = self._assemble_soft_quant_feat(pred, gate_prob)
        quant_hard = quant_hard.to(dtype=quant_soft.dtype, device=quant_soft.device)
        quant_feat = quant_soft + (quant_hard - quant_soft).detach()
        gate_cond = self._straight_through_gate(gate_prob, grain_idx)
        return quant_feat, grain_idx, gate_cond

    def decode_with_fusion(self, quant: Tensor, enc_feat_dict: Optional[Dict[str, Tensor]] = None,
                           gate_prob: Optional[Tensor] = None, grain_idx: Optional[Tensor] = None,
                           gamma_map: Optional[Tensor] = None, w: float = 0.0) -> Tensor:
        h = self.post_quant_conv(quant)
        # Preserve the Stage-I DQ-VAE grain-conditioned latent prior before
        # inserting CodeFormer-style Stage-III SFT corrections.
        if grain_idx is not None:
            h = self._apply_grain_latent_adapters(h, grain_indices=grain_idx, gate=None)
            grain_map = make_grain_map(grain_idx, None, self.num_grains, h.shape[-2:], h.dtype, h.device)
        elif gate_prob is not None:
            h = self._apply_grain_latent_adapters(h, grain_indices=None, gate=gate_prob)
            grain_map = make_grain_map(None, gate_prob, self.num_grains, h.shape[-2:], h.dtype, h.device)
        else:
            grain_map = None
        h = self.decoder.add_position(h)
        temb = None
        h = self.decoder.conv_in(h)
        h = self.decoder.mid.block_1(h, temb)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h, temb)
        for i_level in reversed(range(self.decoder.num_resolutions)):
            for i_block in range(self.decoder.num_res_blocks + 1):
                h = self.decoder.up[i_level].block[i_block](h, temb)
                if len(self.decoder.up[i_level].attn) > 0:
                    h = self.decoder.up[i_level].attn[i_block](h)
            f_size = str(h.shape[-1])
            if grain_map is not None:
                h = self.decoder.grain_film[f_size](h, grain_map)
            if w > 0 and enc_feat_dict is not None and f_size in self.fuse_convs_dict and f_size in enc_feat_dict:
                # CodeFormer Stage-III keeps the frozen prior/code path stable and
                # trains image-level corrections mainly through the fusion branch.
                # The LQ encoder feature is detached as in CodeFormer; Gamma is
                # still computed from the predicted routing distribution and used
                # exactly as the spatial modulation factor in GAMF.
                h = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), h, gamma=gamma_map, w=w)
            if i_level != 0:
                h = self.decoder.up[i_level].upsample(h)
        h = self.decoder.norm_out(h)
        h = nonlinearity(h)
        h = self.decoder.conv_out(h)
        return h

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor):
        # For frozen GT extraction, use the inherited DQ-VAE encoder/quantizer.
        return super().encode_to_indices(x)

    def forward(self, x: Tensor, w: float = 0.0, detach_quant: bool = True,
                code_only: bool = False, adain: bool = False):
        h_dict, h_coarse, h_median, h_fine, h_dynamic = self._encode_lq(x)
        pred = self._predict_multigrain_logits(h_coarse, h_median, h_fine)
        encoder_gate_logits = h_dict.get("gate_logits", None)
        if encoder_gate_logits is not None:
            if encoder_gate_logits.dim() == 4 and encoder_gate_logits.shape[-1] in (2, 3):
                encoder_gate_logits = encoder_gate_logits.permute(0, 3, 1, 2).contiguous()
            pred["encoder_gate"] = encoder_gate_logits
        if code_only:
            return pred, h_dynamic
        quant_feat, grain_idx, gate_cond = self.logits_to_quant_feat(pred, detach_quant=detach_quant)
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, h_dynamic)
        # ``gate_cond`` is only populated when callers explicitly request a
        # differentiable quant/gate path via detach_quant=False. Stage-3 normally
        # keeps detach_quant=True to mirror CodeFormer and decodes from a detached
        # hard code map. GAMF, however, always uses the soft routing distribution
        # pi=softmax(G_lq) to build Gamma, as required by Eq. (15).
        gamma_map = self.build_gamf_gamma(pred)
        if detach_quant:
            # Stage-III image-level losses should train the GAMF residual branch
            # without rewriting the route predictor through Gamma.  Routing remains
            # supervised by L_routing in L_stage-II, while Eq. (15)'s forward Gamma
            # definition is unchanged.
            gamma_map = gamma_map.detach()
        out = self.decode_with_fusion(
            quant_feat, enc_feat_dict=h_dict.get("features", None), gate_prob=gate_cond,
            grain_idx=None if gate_cond is not None else grain_idx, gamma_map=gamma_map, w=w)
        return out, pred, h_dynamic

    def reset_fuse_sft_parameters(self) -> None:
        """Reset only the stage-3 SFT fusion modules.

        This optional reset follows normal module initialization, matching the
        behavior of constructing CodeFormer's SFT blocks.  It must not be called
        when resuming an interrupted stage-3 run because that would erase the
        learned image-level fusion branch.
        """
        for block in self.fuse_convs_dict.values():
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
            else:
                for module in block.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

    def get_last_fusion_layer(self):
        if len(self.fuse_convs_dict) == 0:
            return self.decoder.conv_out.weight
        # choose the largest spatial fusion block, matching CodeFormer's stage-3 adaptive weight practice.
        key = sorted(self.fuse_convs_dict.keys(), key=lambda x: int(x))[-1]
        block = self.fuse_convs_dict[key]
        if hasattr(block, "shift_head"):
            return block.shift_head[-1].weight
        return block.shift[-1].weight


# -----------------------------------------------------------------------------
# Target construction helpers used by DMDPFRModel.
# -----------------------------------------------------------------------------

def make_triple_targets(full_indices: Tensor, grain_indices: Tensor) -> Dict[str, Tensor]:
    """Build selected-scale code targets from a full fine code map.

    full_indices: B,Hf,Wf full latent map from DQ-VAE.
    grain_indices: B,Hc,Wc where values are 0(coarse), 1(median), 2(fine).
    In the provided 512px triple config, Hf=32 and Hc=8; the 4/2/1
    repeats below map coarse/median/fine codes onto the full latent grid.
    """
    coarse_t = full_indices[:, ::4, ::4].contiguous()
    median_t = full_indices[:, ::2, ::2].contiguous()
    fine_t = full_indices.contiguous()
    mask_coarse = (grain_indices == 0)
    mask_median = (grain_indices == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    mask_fine = (grain_indices == 2).repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
    return {"coarse": coarse_t.long(), "median": median_t.long(), "fine": fine_t.long(),
            "mask_coarse": mask_coarse, "mask_median": mask_median, "mask_fine": mask_fine}


def make_dual_targets(full_indices: Tensor, grain_indices: Tensor) -> Dict[str, Tensor]:
    coarse_t = full_indices[:, ::2, ::2].contiguous()
    fine_t = full_indices.contiguous()
    mask_coarse = (grain_indices == 0)
    mask_fine = (grain_indices == 1).repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    return {"coarse": coarse_t.long(), "fine": fine_t.long(),
            "mask_coarse": mask_coarse, "mask_fine": mask_fine}