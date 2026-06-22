# """Training model for multi-granularity DMDP-FR stage-2 and stage-3."""
# from __future__ import annotations

# from collections import OrderedDict
# from os import path as osp
# from typing import Dict

# import torch
# import torch.nn.functional as F
# from tqdm import tqdm

# from basicsr.archs import build_network
# from basicsr.archs.dmdp_fr_arch import make_dual_targets, make_triple_targets
# from basicsr.losses import build_loss
# from basicsr.metrics import calculate_metric
# from basicsr.utils import get_root_logger, imwrite, tensor2img
# from basicsr.utils.registry import MODEL_REGISTRY
# from .sr_model import SRModel


# @MODEL_REGISTRY.register()
# class DMDPFRModel(SRModel):
#     """Stage-2/3 model.

#     Stage-2: code-only training on CodeFormer degradations with GT codes/routes
#     from the frozen DQ-VAE prior.  Stage-3: enables grain-aware fusion and
#     image-level losses while keeping the same code/route supervision.
#     """

#     def feed_data(self, data):
#         self.gt = data["gt"].to(self.device)
#         input_key = "in" if "in" in data else "lq"
#         self.input = data[input_key].to(self.device)
#         self.input_large_de = data["in_large_de"].to(self.device) if "in_large_de" in data else None
#         self.b = self.gt.shape[0]
#         if "latent_gt" in data:
#             latent = data["latent_gt"]
#             if isinstance(latent, dict):
#                 self.idx_gt = torch.as_tensor(latent["indices"], device=self.device).long()
#                 self.grain_gt = torch.as_tensor(latent["grain_indices"], device=self.device).long()
#             else:
#                 # Backwards compatibility is kept, but multi-grain training should use network_vqgan
#                 # or a dict latent with both indices and grain_indices.
#                 self.idx_gt = torch.as_tensor(latent, device=self.device).long()
#                 self.grain_gt = None
#         else:
#             self.idx_gt = None
#             self.grain_gt = None

#     def init_training_settings(self):
#         logger = get_root_logger()
#         train_opt = self.opt["train"]
#         self.ema_decay = train_opt.get("ema_decay", 0)
#         if self.ema_decay > 0:
#             logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
#             self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
#             load_path = self.opt["path"].get("pretrain_network_g", None)
#             if load_path is not None:
#                 self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True),
#                                   self.opt["path"].get("param_key_g", "params_ema"))
#             else:
#                 self.model_ema(0)
#             self.net_g_ema.eval()

#         train_latent_gt_path = self.opt.get("datasets", {}).get("train", {}).get("latent_gt_path", None)
#         if train_latent_gt_path is not None:
#             self.generate_idx_gt = False
#         elif self.opt.get("network_vqgan", None) is not None:
#             self.hq_vqgan_fix = build_network(self.opt["network_vqgan"]).to(self.device)
#             self.hq_vqgan_fix.eval()
#             for p in self.hq_vqgan_fix.parameters():
#                 p.requires_grad = False
#             self.generate_idx_gt = True
#         else:
#             raise NotImplementedError("DMDP-FR requires network_vqgan or precomputed multi-grain latent_gt.")
#         logger.info(f"Need to generate multi-grain latent GT code: {self.generate_idx_gt}")

#         self.hq_feat_loss = train_opt.get("use_hq_feat_loss", True)
#         self.feat_loss_weight = train_opt.get("feat_loss_weight", 1.0)
#         self.code_loss_weight = train_opt.get("code_loss_weight", 1.0)
#         self.gate_loss_weight = train_opt.get("gate_loss_weight", 1.0)
#         self.supervise_all_codes = train_opt.get("supervise_all_codes", False)
#         self.fidelity_weight = train_opt.get("fidelity_weight", 0.0)
#         self.scale_adaptive_gan_weight = train_opt.get("scale_adaptive_gan_weight", 0.1)
#         self.fix_generator = train_opt.get("fix_generator", True)
#         self.use_large_de_train = train_opt.get("use_large_de_train", False)
#         self.small_de_w1_until_iter = int(train_opt.get("small_de_w1_until_iter", 40000))
#         self.large_de_start_iter = int(train_opt.get("large_de_start_iter", 80000))
#         self.large_de_only_until_iter = int(train_opt.get("large_de_only_until_iter", 120000))
#         self.large_de_mixed_interval = max(1, int(train_opt.get("large_de_mixed_interval", 15)))
#         if self.use_large_de_train and not (
#             self.small_de_w1_until_iter <= self.large_de_start_iter <= self.large_de_only_until_iter
#         ):
#             raise ValueError(
#                 "Expected small_de_w1_until_iter <= large_de_start_iter <= large_de_only_until_iter, "
#                 f"but got {self.small_de_w1_until_iter}, {self.large_de_start_iter}, "
#                 f"{self.large_de_only_until_iter}."
#             )
#         self.net_g_start_iter = train_opt.get("net_g_start_iter", 0)
#         self.net_d_iters = train_opt.get("net_d_iters", 1)
#         self.net_d_start_iter = train_opt.get("net_d_start_iter", 0)
#         self.setup_amp(train_opt, logger)
#         self.setup_gradient_accumulation(train_opt, logger)

#         self.net_g.train()
#         self.use_gan = self.fidelity_weight > 0 and self.opt.get("network_d", None) is not None
#         if self.use_gan:
#             self.net_d = build_network(self.opt["network_d"])
#             self.net_d = self.model_to_device(self.net_d)
#             self.print_network(self.net_d)
#             load_path = self.opt["path"].get("pretrain_network_d", None)
#             if load_path is not None:
#                 self.load_network(self.net_d, load_path, self.opt["path"].get("strict_load_d", True))
#             self.net_d.train()

#         if train_opt.get("pixel_opt"):
#             self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
#         else:
#             self.cri_pix = None
#         if train_opt.get("perceptual_opt"):
#             self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)
#         else:
#             self.cri_perceptual = None
#         if train_opt.get("gan_opt"):
#             self.cri_gan = build_loss(train_opt["gan_opt"]).to(self.device)
#         else:
#             self.cri_gan = None

#         self.setup_optimizers()
#         self.setup_schedulers()

#     @staticmethod
#     def _unwrap(model):
#         return model.module if hasattr(model, "module") else model

#     def setup_optimizers(self):
#         train_opt = self.opt["train"]
#         optim_params_g = []
#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#                 optim_params_g.append(v)
#             else:
#                 get_root_logger().warning(f"Params {k} will not be optimized.")
#         optim_type = train_opt["optim_g"].pop("type")
#         self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt["optim_g"])
#         self.optimizers.append(self.optimizer_g)
#         if self.use_gan:
#             optim_type = train_opt["optim_d"].pop("type")
#             self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt["optim_d"])
#             self.optimizers.append(self.optimizer_d)

#     def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max=1.0):
#         recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
#         g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
#         return torch.clamp(d_weight, 0.0, disc_weight_max).detach()

#     @staticmethod
#     def masked_cross_entropy(logits, target, mask=None):
#         # logits B,N,H,W; target B,H,W; mask B,H,W bool.
#         loss = F.cross_entropy(logits, target.long(), reduction="none")
#         if mask is None:
#             return loss.mean()
#         mask = mask.to(loss.device).float()
#         denom = mask.sum().clamp_min(1.0)
#         return (loss * mask).sum() / denom

#     def _make_targets(self, full_indices, grain_indices, grain_type):
#         return make_triple_targets(full_indices, grain_indices) if grain_type == "triple" else make_dual_targets(full_indices, grain_indices)

#     def _generate_gt(self):
#         if self.generate_idx_gt:
#             with torch.no_grad():
#                 idx_gt, grain_gt, quant_feat_gt = self.hq_vqgan_fix.encode_to_indices(self.gt)
#             self.idx_gt = idx_gt.long()
#             self.grain_gt = grain_gt.long()
#             self.quant_feat_gt = quant_feat_gt.detach()
#         else:
#             if self.idx_gt is None or self.grain_gt is None:
#                 raise ValueError("Precomputed DQ latent_gt must contain both 'indices' and 'grain_indices'.")
#             net_g = self._unwrap(self.net_g)
#             with torch.no_grad():
#                 self.quant_feat_gt = net_g.get_codebook_feat(self.idx_gt).detach()

#     def _code_and_route_losses(self, pred, lq_feat, loss_dict):
#         net_g = self._unwrap(self.net_g)
#         grain_type = net_g.grain_type
#         targets = self._make_targets(self.idx_gt, self.grain_gt, grain_type)

#         l_total = 0
#         l_gate = F.cross_entropy(pred["gate"], self.grain_gt.long()) * self.gate_loss_weight
#         l_total += l_gate
#         loss_dict["l_gate_ce"] = l_gate

#         if self.supervise_all_codes:
#             mask_coarse = mask_median = mask_fine = None
#         else:
#             mask_coarse = targets["mask_coarse"]
#             mask_median = targets.get("mask_median", None)
#             mask_fine = targets["mask_fine"]

#         l_code_coarse = self.masked_cross_entropy(pred["coarse"], targets["coarse"], mask_coarse)
#         l_code = l_code_coarse
#         loss_dict["l_code_coarse"] = l_code_coarse
#         if grain_type == "triple":
#             l_code_median = self.masked_cross_entropy(pred["median"], targets["median"], mask_median)
#             l_code = l_code + l_code_median
#             loss_dict["l_code_median"] = l_code_median
#         l_code_fine = self.masked_cross_entropy(pred["fine"], targets["fine"], mask_fine)
#         l_code = (l_code + l_code_fine) * self.code_loss_weight
#         loss_dict["l_code_fine"] = l_code_fine
#         loss_dict["l_code_total"] = l_code
#         l_total += l_code

#         if self.hq_feat_loss:
#             # LQ dynamic feature approaches the quantized HQ feature from the frozen DQ-VAE.
#             l_feat_encoder = torch.mean((self.quant_feat_gt.detach() - lq_feat) ** 2) * self.feat_loss_weight
#             l_total += l_feat_encoder
#             loss_dict["l_feat_encoder"] = l_feat_encoder
#         return l_total

#     def _use_large_degradation(self, current_iter):
#         if not self.use_large_de_train or self.input_large_de is None:
#             return False
#         if current_iter <= self.large_de_start_iter:
#             return False
#         if current_iter <= self.large_de_only_until_iter:
#             return True
#         return current_iter % self.large_de_mixed_interval != 0

#     def _degradation_and_w(self, current_iter):
#         """Match CodeFormer stage-3's degradation and fidelity-weight schedule."""
#         if self.fidelity_weight <= 0:
#             return self._use_large_degradation(current_iter), 0.0
#         if not self.use_large_de_train or self.input_large_de is None:
#             return False, self.fidelity_weight
#         if current_iter <= self.small_de_w1_until_iter:
#             return False, 1.0
#         if current_iter <= self.large_de_start_iter:
#             return False, 1.3
#         if current_iter <= self.large_de_only_until_iter:
#             return True, 0.0
#         if current_iter % self.large_de_mixed_interval == 0:
#             return False, 1.3
#         return True, 0.0

#     def optimize_parameters(self, current_iter):
#         loss_dict = OrderedDict()
#         self._generate_gt()
#         use_large_de, train_w = self._degradation_and_w(current_iter)
#         train_input = self.input_large_de if use_large_de else self.input
#         use_image_losses = train_w > 0 and not use_large_de
#         loss_dict["large_de"] = self.gt.new_tensor(float(use_large_de))
#         loss_dict["fidelity_w"] = self.gt.new_tensor(float(train_w))

#         if self.use_gan:
#             for p in self.net_d.parameters():
#                 p.requires_grad = False
#         if self.is_accumulation_start(current_iter):
#             self.optimizer_g.zero_grad()
#         did_optimizer_step = False
#         did_g_optimizer_step = False
#         do_g_step = False
#         with self.amp_autocast():
#             if use_image_losses:
#                 self.output, pred, lq_feat = self.net_g(train_input, w=train_w, detach_quant=True)
#             else:
#                 pred, lq_feat = self.net_g(train_input, w=0, code_only=True)
#                 self.output = None

#             l_g_total = 0
#             do_g_step = current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter
#             if do_g_step:
#                 l_g_total = l_g_total + self._code_and_route_losses(pred, lq_feat, loss_dict)

#                 if use_image_losses:
#                     recon_loss = 0
#                     if self.cri_pix:
#                         l_g_pix = self.cri_pix(self.output, self.gt)
#                         l_g_total += l_g_pix
#                         recon_loss = recon_loss + l_g_pix
#                         loss_dict["l_g_pix"] = l_g_pix
#                     if self.cri_perceptual:
#                         l_g_percep = self.cri_perceptual(self.output, self.gt)
#                         l_g_total += l_g_percep
#                         recon_loss = recon_loss + l_g_percep
#                         loss_dict["l_g_percep"] = l_g_percep
#                     if self.use_gan and current_iter > self.net_d_start_iter:
#                         fake_g_pred = self.net_d(self.output)
#                         l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
#                         net_g = self._unwrap(self.net_g)
#                         if self.fix_generator:
#                             last_layer = net_g.get_last_fusion_layer()
#                         else:
#                             last_layer = net_g.decoder.conv_out.weight
#                         d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
#                         d_weight *= self.scale_adaptive_gan_weight
#                         l_g_total += d_weight * l_g_gan
#                         loss_dict["d_weight"] = d_weight
#                         loss_dict["l_g_gan"] = d_weight * l_g_gan
#                 l_g_total = l_g_total / self.get_accumulation_loss_scale(current_iter)

#         if do_g_step:
#             self.amp_scaler.scale(l_g_total).backward()
#             if self.is_accumulation_update(current_iter):
#                 self.amp_scaler.step(self.optimizer_g)
#                 self.optimizer_g.zero_grad()
#                 did_optimizer_step = True
#                 did_g_optimizer_step = True

#         if self.ema_decay > 0 and did_g_optimizer_step:
#             self.model_ema(decay=self.ema_decay)

#         if self.use_gan and current_iter > self.net_d_start_iter and use_image_losses:
#             for p in self.net_d.parameters():
#                 p.requires_grad = True
#             if self.is_accumulation_start(current_iter):
#                 self.optimizer_d.zero_grad()
#             with self.amp_autocast():
#                 real_d_pred = self.net_d(self.gt)
#                 l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
#                 l_d_real_backward = l_d_real / self.get_accumulation_loss_scale(current_iter)
#             loss_dict["l_d_real"] = l_d_real
#             loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
#             self.amp_scaler.scale(l_d_real_backward).backward()
#             with self.amp_autocast():
#                 fake_d_pred = self.net_d(self.output.detach())
#                 l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
#                 l_d_fake_backward = l_d_fake / self.get_accumulation_loss_scale(current_iter)
#             loss_dict["l_d_fake"] = l_d_fake
#             loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
#             self.amp_scaler.scale(l_d_fake_backward).backward()
#             if self.is_accumulation_update(current_iter):
#                 self.amp_scaler.step(self.optimizer_d)
#                 self.optimizer_d.zero_grad()
#                 did_optimizer_step = True

#         if did_optimizer_step:
#             self.amp_scaler.update()

#         self.log_dict = self.reduce_loss_dict(loss_dict)

#     def test(self):
#         with torch.no_grad():
#             if hasattr(self, "net_g_ema"):
#                 self.net_g_ema.eval()
#                 self.output, _, _ = self.net_g_ema(self.input, w=self.fidelity_weight)
#             else:
#                 self.net_g.eval()
#                 self.output, _, _ = self.net_g(self.input, w=self.fidelity_weight)
#                 self.net_g.train()

#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
#         self.rank0_validation(current_iter, self.nondist_validation, dataloader, current_iter, tb_logger, save_img)

#     def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
#         dataset_name = dataloader.dataset.opt["name"]
#         with_metrics = self.opt["val"].get("metrics") is not None
#         if with_metrics:
#             self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
#         pbar = tqdm(total=len(dataloader), unit="image")
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data.get("lq_path", [f"{idx:08d}"])[0]))[0]
#             self.feed_data(val_data)
#             self.test()
#             visuals = self.get_current_visuals()
#             sr_img = tensor2img([visuals["result"]])
#             gt_img = tensor2img([visuals["gt"]])
#             if save_img:
#                 if self.opt["is_train"]:
#                     save_img_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
#                 else:
#                     suffix = self.opt["val"].get("suffix", self.opt["name"])
#                     save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}_{suffix}.png")
#                 imwrite(sr_img, save_img_path)
#             if with_metrics:
#                 for name, opt_ in self.opt["val"]["metrics"].items():
#                     metric_data = dict(img1=sr_img, img2=gt_img)
#                     self.metric_results[name] += calculate_metric(metric_data, opt_)
#             pbar.update(1)
#             pbar.set_description(f"Test {img_name}")
#         pbar.close()
#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#             self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

#     def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
#         log_str = f"Validation {dataset_name}\n"
#         for metric, value in self.metric_results.items():
#             log_str += f"\t # {metric}: {value:.4f}\n"
#         logger = get_root_logger()
#         logger.info(log_str)
#         if tb_logger:
#             for metric, value in self.metric_results.items():
#                 tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict["gt"] = self.gt.detach().cpu()
#         out_dict["input"] = self.input.detach().cpu()
#         out_dict["result"] = self.output.detach().cpu()
#         return out_dict

#     def save(self, epoch, current_iter):
#         if self.ema_decay > 0:
#             self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
#         else:
#             self.save_network(self.net_g, "net_g", current_iter)
#         if self.use_gan:
#             self.save_network(self.net_d, "net_d", current_iter)
#         self.save_training_state(epoch, current_iter)



"""Training model for multi-granularity DMDP-FR stage-2 and stage-3."""
from __future__ import annotations

from collections import OrderedDict
from os import path as osp
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.archs.dmdp_fr_arch import make_dual_targets, make_triple_targets
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class DMDPFRModel(SRModel):
    """Stage-2/3 model.

    Stage-2: code-only training on CodeFormer degradations with GT codes/routes
    from the frozen DQ-VAE prior.  Stage-3: enables grain-aware fusion and
    image-level losses while keeping the same code/route supervision.
    """

    def feed_data(self, data):
        self.gt = data["gt"].to(self.device)
        input_key = "in" if "in" in data else "lq"
        self.input = data[input_key].to(self.device)
        self.input_large_de = data["in_large_de"].to(self.device) if "in_large_de" in data else None
        self.b = self.gt.shape[0]
        if "latent_gt" in data:
            latent = data["latent_gt"]
            if isinstance(latent, dict):
                self.idx_gt = torch.as_tensor(latent["indices"], device=self.device).long()
                self.grain_gt = torch.as_tensor(latent["grain_indices"], device=self.device).long()
            else:
                # Backwards compatibility is kept, but multi-grain training should use network_vqgan
                # or a dict latent with both indices and grain_indices.
                self.idx_gt = torch.as_tensor(latent, device=self.device).long()
                self.grain_gt = None
        else:
            self.idx_gt = None
            self.grain_gt = None

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt["train"]
        # Match CodeFormer stage-3: the SFT fusion state is loaded with net_g
        # from the stage-2 checkpoint and is not reset automatically here.
        reset_stage3_fusion = bool(train_opt.get("reset_stage3_fusion", False))
        should_reset_stage3_fusion = (
            reset_stage3_fusion
            and self.opt["path"].get("resume_state", None) is None
        )
        if should_reset_stage3_fusion:
            net_g = self._unwrap(self.net_g)
            if hasattr(net_g, "reset_fuse_sft_parameters"):
                logger.info("Reset stage-3 SFT fusion modules before image-level training.")
                net_g.reset_fuse_sft_parameters()
            else:
                logger.warning("reset_stage3_fusion is enabled, but net_g has no reset_fuse_sft_parameters().")

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True),
                                  self.opt["path"].get("param_key_g", "params_ema"))
            else:
                self.model_ema(0)
            if should_reset_stage3_fusion:
                # Keep EMA exactly identical to the trainable generator at iter 0,
                # including the freshly-reset identity SFT branch.
                self.model_ema(0)
            self.net_g_ema.eval()

        train_latent_gt_path = self.opt.get("datasets", {}).get("train", {}).get("latent_gt_path", None)
        if train_latent_gt_path is not None:
            self.generate_idx_gt = False
        elif self.opt.get("network_vqgan", None) is not None:
            self.hq_vqgan_fix = build_network(self.opt["network_vqgan"]).to(self.device)
            self.hq_vqgan_fix.eval()
            for p in self.hq_vqgan_fix.parameters():
                p.requires_grad = False
            self.generate_idx_gt = True
        else:
            raise NotImplementedError("DMDP-FR requires network_vqgan or precomputed multi-grain latent_gt.")
        logger.info(f"Need to generate multi-grain latent GT code: {self.generate_idx_gt}")

        self.hq_feat_loss = train_opt.get("use_hq_feat_loss", True)
        self.feat_loss_weight = train_opt.get("feat_loss_weight", 1.0)
        self.code_loss_weight = train_opt.get("code_loss_weight", 1.0)
        self.code_loss_weight_coarse = train_opt.get("code_loss_weight_coarse", self.code_loss_weight)
        self.code_loss_weight_median = train_opt.get("code_loss_weight_median", self.code_loss_weight)
        self.code_loss_weight_fine = train_opt.get("code_loss_weight_fine", self.code_loss_weight)
        self.gate_loss_weight = train_opt.get("gate_loss_weight", 1.0)
        self.encoder_gate_loss_weight = train_opt.get("encoder_gate_loss_weight", 0.0)
        self.supervise_all_codes = train_opt.get("supervise_all_codes", False)
        self.fidelity_weight = train_opt.get("fidelity_weight", 0.0)
        self.scale_adaptive_gan_weight = train_opt.get("scale_adaptive_gan_weight", 0.1)
        self.fix_generator = train_opt.get("fix_generator", True)
        self.use_large_de_train = train_opt.get("use_large_de_train", False)
        self.small_de_w1_until_iter = int(train_opt.get("small_de_w1_until_iter", 40000))
        self.large_de_start_iter = int(train_opt.get("large_de_start_iter", 80000))
        self.large_de_only_until_iter = int(train_opt.get("large_de_only_until_iter", 120000))
        self.large_de_mixed_interval = max(1, int(train_opt.get("large_de_mixed_interval", 15)))
        if self.use_large_de_train and not (
            self.small_de_w1_until_iter <= self.large_de_start_iter <= self.large_de_only_until_iter
        ):
            raise ValueError(
                "Expected small_de_w1_until_iter <= large_de_start_iter <= large_de_only_until_iter, "
                f"but got {self.small_de_w1_until_iter}, {self.large_de_start_iter}, "
                f"{self.large_de_only_until_iter}."
            )
        self.net_g_start_iter = train_opt.get("net_g_start_iter", 0)
        self.net_d_iters = train_opt.get("net_d_iters", 1)
        self.net_d_start_iter = train_opt.get("net_d_start_iter", 0)
        self.setup_amp(train_opt, logger)
        self.setup_gradient_accumulation(train_opt, logger)
        # D is updated only on small-degradation image iterations, so it needs
        # its own accumulation counter rather than the global iteration number.
        self.d_accum_counter = 0

        self.net_g.train()
        self.use_gan = self.fidelity_weight > 0 and self.opt.get("network_d", None) is not None
        if self.use_gan:
            self.net_d = build_network(self.opt["network_d"])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            load_path = self.opt["path"].get("pretrain_network_d", None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt["path"].get("strict_load_d", True))
            self.net_d.train()

        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get("gan_opt"):
            self.cri_gan = build_loss(train_opt["gan_opt"]).to(self.device)
        else:
            self.cri_gan = None

        self.setup_optimizers()
        self.setup_schedulers()

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                get_root_logger().warning(f"Params {k} will not be optimized.")
        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer_g)
        if self.use_gan:
            optim_type = train_opt["optim_d"].pop("type")
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt["optim_d"])
            self.optimizers.append(self.optimizer_d)

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max=1.0):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(d_weight, 0.0, disc_weight_max).detach()

    @staticmethod
    def masked_cross_entropy(logits, target, mask=None):
        # logits B,N,H,W; target B,H,W; mask B,H,W bool.
        loss = F.cross_entropy(logits, target.long(), reduction="none")
        if mask is None:
            return loss.mean()
        mask = mask.to(loss.device).float()
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom

    @staticmethod
    def masked_accuracy(logits, target, mask=None):
        pred = logits.detach().argmax(dim=1)
        correct = (pred == target.long()).float()
        if mask is None:
            return correct.mean()
        mask = mask.to(correct.device).float()
        denom = mask.sum().clamp_min(1.0)
        return (correct * mask).sum() / denom

    def _make_targets(self, full_indices, grain_indices, grain_type):
        return make_triple_targets(full_indices, grain_indices) if grain_type == "triple" else make_dual_targets(full_indices, grain_indices)

    def _generate_gt(self):
        if self.generate_idx_gt:
            with torch.no_grad():
                idx_gt, grain_gt, quant_feat_gt = self.hq_vqgan_fix.encode_to_indices(self.gt)
            self.idx_gt = idx_gt.long()
            self.grain_gt = grain_gt.long()
            self.quant_feat_gt = quant_feat_gt.detach()
        else:
            if self.idx_gt is None or self.grain_gt is None:
                raise ValueError("Precomputed DQ latent_gt must contain both 'indices' and 'grain_indices'.")
            net_g = self._unwrap(self.net_g)
            with torch.no_grad():
                self.quant_feat_gt = net_g.get_codebook_feat(self.idx_gt).detach()

    def _code_and_route_losses(self, pred, lq_feat, loss_dict):
        net_g = self._unwrap(self.net_g)
        grain_type = net_g.grain_type
        targets = self._make_targets(self.idx_gt, self.grain_gt, grain_type)

        l_total = 0
        l_gate = F.cross_entropy(pred["gate"], self.grain_gt.long()) * self.gate_loss_weight
        l_total += l_gate
        loss_dict["l_gate_ce"] = l_gate
        loss_dict["acc_gate"] = (pred["gate"].detach().argmax(dim=1) == self.grain_gt).float().mean()

        encoder_gate = pred.get("encoder_gate", None)
        if self.encoder_gate_loss_weight > 0:
            if encoder_gate is None:
                raise ValueError("encoder_gate_loss_weight > 0 requires DMDP-FR to return encoder_gate logits.")
            if tuple(encoder_gate.shape[-2:]) != tuple(self.grain_gt.shape[-2:]):
                raise ValueError(
                    f"Encoder gate logits shape {tuple(encoder_gate.shape)} is incompatible with "
                    f"grain_gt shape {tuple(self.grain_gt.shape)}."
                )
            l_encoder_gate = F.cross_entropy(encoder_gate, self.grain_gt.long()) * self.encoder_gate_loss_weight
            l_total += l_encoder_gate
            loss_dict["l_encoder_gate_ce"] = l_encoder_gate
            loss_dict["acc_encoder_gate"] = (
                encoder_gate.detach().argmax(dim=1) == self.grain_gt).float().mean()

        if self.supervise_all_codes:
            mask_coarse = mask_median = mask_fine = None
        else:
            mask_coarse = targets["mask_coarse"]
            mask_median = targets.get("mask_median", None)
            mask_fine = targets["mask_fine"]

        l_code_coarse_raw = self.masked_cross_entropy(pred["coarse"], targets["coarse"], mask_coarse)
        l_code_coarse = l_code_coarse_raw * self.code_loss_weight_coarse
        l_code = l_code_coarse
        loss_dict["l_code_coarse"] = l_code_coarse_raw
        loss_dict["l_code_coarse_w"] = l_code_coarse
        loss_dict["acc_code_coarse"] = self.masked_accuracy(pred["coarse"], targets["coarse"], mask_coarse)
        if grain_type == "triple":
            l_code_median_raw = self.masked_cross_entropy(pred["median"], targets["median"], mask_median)
            l_code_median = l_code_median_raw * self.code_loss_weight_median
            l_code = l_code + l_code_median
            loss_dict["l_code_median"] = l_code_median_raw
            loss_dict["l_code_median_w"] = l_code_median
            loss_dict["acc_code_median"] = self.masked_accuracy(pred["median"], targets["median"], mask_median)
        l_code_fine_raw = self.masked_cross_entropy(pred["fine"], targets["fine"], mask_fine)
        l_code_fine = l_code_fine_raw * self.code_loss_weight_fine
        l_code = l_code + l_code_fine
        loss_dict["l_code_fine"] = l_code_fine_raw
        loss_dict["l_code_fine_w"] = l_code_fine
        loss_dict["acc_code_fine"] = self.masked_accuracy(pred["fine"], targets["fine"], mask_fine)
        loss_dict["l_code_total"] = l_code
        l_total += l_code

        if self.hq_feat_loss:
            # LQ dynamic feature approaches the quantized HQ feature from the frozen DQ-VAE.
            l_feat_encoder = torch.mean((self.quant_feat_gt.detach() - lq_feat) ** 2) * self.feat_loss_weight
            l_total += l_feat_encoder
            loss_dict["l_feat_encoder"] = l_feat_encoder
        return l_total

    def _use_large_degradation(self, current_iter):
        if not self.use_large_de_train or self.input_large_de is None:
            return False
        if current_iter <= self.large_de_start_iter:
            return False
        if current_iter <= self.large_de_only_until_iter:
            return True
        return current_iter % self.large_de_mixed_interval != 0

    def _degradation_and_w(self, current_iter):
        """Match CodeFormer stage-3's degradation and fidelity-weight schedule."""
        if self.fidelity_weight <= 0:
            return self._use_large_degradation(current_iter), 0.0
        if not self.use_large_de_train or self.input_large_de is None:
            return False, self.fidelity_weight
        if current_iter <= self.small_de_w1_until_iter:
            return False, 1.0
        if current_iter <= self.large_de_start_iter:
            return False, 1.3
        if current_iter <= self.large_de_only_until_iter:
            return True, 0.0
        if current_iter % self.large_de_mixed_interval == 0:
            return False, 1.3
        return True, 0.0

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        self._generate_gt()
        use_large_de, train_w = self._degradation_and_w(current_iter)
        train_input = self.input_large_de if use_large_de else self.input
        use_image_losses = train_w > 0 and not use_large_de
        loss_dict["large_de"] = self.gt.new_tensor(float(use_large_de))
        loss_dict["fidelity_w"] = self.gt.new_tensor(float(train_w))

        if self.use_gan:
            for p in self.net_d.parameters():
                p.requires_grad = False
        if self.is_accumulation_start(current_iter):
            self.optimizer_g.zero_grad()
        did_optimizer_step = False
        did_g_optimizer_step = False
        do_g_step = False
        with self.amp_autocast():
            if use_image_losses:
                # Match CodeFormer stage-3: image-level losses decode from a
                # detached hard code map, so they optimize the SFT fusion branch
                # without back-propagating into code/route prediction.
                self.output, pred, lq_feat = self.net_g(train_input, w=train_w, detach_quant=True)
            else:
                pred, lq_feat = self.net_g(train_input, w=0, code_only=True)
                self.output = None

            l_g_total = 0
            do_g_step = current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter
            if do_g_step:
                l_g_total = l_g_total + self._code_and_route_losses(pred, lq_feat, loss_dict)

                if use_image_losses:
                    recon_loss = 0
                    if self.cri_pix:
                        l_g_pix = self.cri_pix(self.output, self.gt)
                        l_g_total += l_g_pix
                        recon_loss = recon_loss + l_g_pix
                        loss_dict["l_g_pix"] = l_g_pix
                    if self.cri_perceptual:
                        l_g_percep = self.cri_perceptual(self.output, self.gt)
                        l_g_total += l_g_percep
                        recon_loss = recon_loss + l_g_percep
                        loss_dict["l_g_percep"] = l_g_percep
                    if self.use_gan and current_iter > self.net_d_start_iter:
                        fake_g_pred = self.net_d(self.output)
                        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                        net_g = self._unwrap(self.net_g)
                        if self.fix_generator:
                            last_layer = net_g.get_last_fusion_layer()
                        else:
                            last_layer = net_g.decoder.conv_out.weight
                        d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                        d_weight *= self.scale_adaptive_gan_weight
                        l_g_total += d_weight * l_g_gan
                        loss_dict["d_weight"] = d_weight
                        loss_dict["l_g_gan"] = d_weight * l_g_gan
                l_g_total = l_g_total / self.get_accumulation_loss_scale(current_iter)

        if do_g_step:
            self.amp_scaler.scale(l_g_total).backward()
            if self.is_accumulation_update(current_iter):
                self.amp_scaler.step(self.optimizer_g)
                self.optimizer_g.zero_grad()
                did_optimizer_step = True
                did_g_optimizer_step = True

        if self.ema_decay > 0 and did_g_optimizer_step:
            self.model_ema(decay=self.ema_decay)

        if self.use_gan and current_iter > self.net_d_start_iter and use_image_losses:
            for p in self.net_d.parameters():
                p.requires_grad = True
            if self.d_accum_counter % self.grad_accum_steps == 0:
                self.optimizer_d.zero_grad()
            d_loss_scale = float(self.grad_accum_steps)
            with self.amp_autocast():
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                l_d_real_backward = l_d_real / d_loss_scale
            loss_dict["l_d_real"] = l_d_real
            loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
            self.amp_scaler.scale(l_d_real_backward).backward()
            with self.amp_autocast():
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                l_d_fake_backward = l_d_fake / d_loss_scale
            loss_dict["l_d_fake"] = l_d_fake
            loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
            self.amp_scaler.scale(l_d_fake_backward).backward()
            self.d_accum_counter += 1
            is_final_iter = current_iter >= int(self.opt.get("train", {}).get("total_iter", current_iter + 1))
            if self.d_accum_counter % self.grad_accum_steps == 0 or is_final_iter:
                self.amp_scaler.step(self.optimizer_d)
                self.optimizer_d.zero_grad()
                did_optimizer_step = True

        if did_optimizer_step:
            self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                self.output, _, _ = self.net_g_ema(self.input, w=self.fidelity_weight)
            else:
                self.net_g.eval()
                self.output, _, _ = self.net_g(self.input, w=self.fidelity_weight)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.rank0_validation(current_iter, self.nondist_validation, dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
        pbar = tqdm(total=len(dataloader), unit="image")
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data.get("lq_path", [f"{idx:08d}"])[0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            gt_img = tensor2img([visuals["gt"]])
            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
                else:
                    suffix = self.opt["val"].get("suffix", self.opt["name"])
                    save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}_{suffix}.png")
                imwrite(sr_img, save_img_path)
            if with_metrics:
                for name, opt_ in self.opt["val"]["metrics"].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f"Test {img_name}")
        pbar.close()
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["gt"] = self.gt.detach().cpu()
        out_dict["input"] = self.input.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        if self.use_gan:
            self.save_network(self.net_d, "net_d", current_iter)
        self.save_training_state(epoch, current_iter)
