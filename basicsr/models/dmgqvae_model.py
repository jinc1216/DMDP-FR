from __future__ import annotations

from collections import OrderedDict
from os import path as osp

import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.archs.dmgqvae_arch import (
    BudgetConstraint_NormedSeperateRatioMSE_TripleGrain,
    BudgetConstraint_RatioMSE_DualGrain,
    DMGQLPIPS,
)
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class DMGQVAEModel(SRModel):
    """Stage-1 HQ prior learning with Dynamic Vector Quantization."""

    def feed_data(self, data):
        self.gt = data["gt"].to(self.device)
        self.b = self.gt.shape[0]

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt["train"]
        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt["path"].get("strict_load_g", True), "params_ema")
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self.net_d = build_network(self.opt["network_d"])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        load_path = self.opt["path"].get("pretrain_network_d", None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt["path"].get("strict_load_d", True))

        self.net_g.train()
        self.net_d.train()

        # DMGQ-VAE stage-1 loss hyperparameters; defaults match dmgqvae-dual-r-05_imagenet.yml.
        self.codebook_weight = train_opt.get("codebook_weight", 1.0)
        self.pixelloss_weight = train_opt.get("pixelloss_weight", 1.0)
        self.disc_weight = train_opt.get("disc_weight", 0.8)
        self.perceptual_weight = train_opt.get("perceptual_weight", 1.0)
        self.net_d_start_iter = train_opt.get("net_d_start_iter", 0)
        self.net_g_start_iter = train_opt.get("net_g_start_iter", 0)
        self.net_d_iters = train_opt.get("net_d_iters", 1)
        self.cri_gan = build_loss(train_opt["gan_opt"]).to(self.device) if train_opt.get("gan_opt") else None
        self.setup_amp(train_opt, logger)
        self.setup_gradient_accumulation(train_opt, logger)
        self.cri_lpips = DMGQLPIPS(train_opt.get("lpips_weight_path", "weights/lpips/vgg.pth")).to(self.device).eval()

        grain_type = self.opt["network_g"].get("grain_type", "triple")
        budget_opt = train_opt.get("budget_opt", None)
        if budget_opt is None:
            self.budget_loss = None
        elif grain_type == "dual":
            self.budget_loss = BudgetConstraint_RatioMSE_DualGrain(**budget_opt).to(self.device)
        elif grain_type == "triple":
            self.budget_loss = BudgetConstraint_NormedSeperateRatioMSE_TripleGrain(**budget_opt).to(self.device)
        else:
            raise ValueError(f"Unknown grain_type {grain_type}")

        self.setup_optimizers()
        self.setup_schedulers()

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
        optim_type = train_opt["optim_d"].pop("type")
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt["optim_d"])
        self.optimizers.append(self.optimizer_d)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(d_weight, 0.0, 1.0).detach()

    def _gan_is_active(self, current_iter):
        return current_iter > self.net_d_start_iter

    def _generator_loss(self, x, xrec, qloss, gate, current_iter, loss_dict):
        rec_loss = torch.abs(x.contiguous() - xrec.contiguous()) * self.pixelloss_weight
        if self.perceptual_weight > 0:
            p_loss = self.cri_lpips(x.contiguous(), xrec.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0, device=x.device)
        nll_loss = torch.mean(rec_loss)
        gan_active = self._gan_is_active(current_iter)
        if gan_active:
            if self.cri_gan is None:
                raise ValueError("DMGQVAEModel requires train.gan_opt when net_d is active.")
            logits_fake = self.net_d(xrec.contiguous())
            g_loss = self.cri_gan(logits_fake, True, is_disc=False)
            try:
                last_layer = self._unwrap(self.net_g).decoder.conv_out.weight
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                d_weight = torch.tensor(0.0, device=x.device)
            d_weight = d_weight * self.disc_weight
        else:
            g_loss = torch.tensor(0.0, device=x.device)
            d_weight = torch.tensor(0.0, device=x.device)
        loss = nll_loss + d_weight * g_loss + self.codebook_weight * qloss.mean()
        if self.budget_loss is not None:
            budget = self.budget_loss(gate=gate)
            loss = loss + budget
            loss_dict["l_budget"] = budget.detach().mean()
        loss_dict["l_rec"] = rec_loss.detach().mean()
        loss_dict["l_nll"] = nll_loss.detach().mean()
        loss_dict["l_lpips"] = p_loss.detach().mean() if torch.is_tensor(p_loss) else torch.tensor(p_loss)
        loss_dict["l_codebook"] = qloss.detach().mean()
        loss_dict["l_g_gan"] = (d_weight * g_loss).detach().mean()
        loss_dict["d_weight"] = d_weight.detach()
        return loss

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        for p in self.net_d.parameters():
            p.requires_grad = False

        if self.is_accumulation_start(current_iter):
            self.optimizer_g.zero_grad()
        did_optimizer_step = False
        did_g_optimizer_step = False
        do_g_step = False
        with self.amp_autocast():
            self.output, qloss, quant_stats = self.net_g(self.gt)
            gate = quant_stats["gate"]
            grain_indices = quant_stats["grain_indices"]
            do_g_step = current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter
            if do_g_step:
                l_g_total = self._generator_loss(self.gt, self.output, qloss, gate, current_iter, loss_dict)
                l_g_total = l_g_total / self.get_accumulation_loss_scale(current_iter)

        if do_g_step:
            self.amp_scaler.scale(l_g_total).backward()
            if self.is_accumulation_update(current_iter):
                self.amp_scaler.step(self.optimizer_g)
                self.optimizer_g.zero_grad()
                did_optimizer_step = True
                did_g_optimizer_step = True

        if self._gan_is_active(current_iter):
            if self.cri_gan is None:
                raise ValueError("DMGQVAEModel requires train.gan_opt when net_d is active.")
            for p in self.net_d.parameters():
                p.requires_grad = True
            if self.is_accumulation_start(current_iter):
                self.optimizer_d.zero_grad()
            with self.amp_autocast():
                real_d_pred = self.net_d(self.gt.contiguous().detach())
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                l_d_real_backward = l_d_real / self.get_accumulation_loss_scale(current_iter)
            self.amp_scaler.scale(l_d_real_backward).backward()

            with self.amp_autocast():
                fake_d_pred = self.net_d(self.output.contiguous().detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                l_d_fake_backward = l_d_fake / self.get_accumulation_loss_scale(current_iter)
            self.amp_scaler.scale(l_d_fake_backward).backward()

            l_d = l_d_real + l_d_fake
            loss_dict["l_d"] = l_d.detach().mean()
            loss_dict["l_d_real"] = l_d_real.detach().mean()
            loss_dict["l_d_fake"] = l_d_fake.detach().mean()
            loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
            loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
            if self.is_accumulation_update(current_iter):
                self.amp_scaler.step(self.optimizer_d)
                self.optimizer_d.zero_grad()
                did_optimizer_step = True

        if did_optimizer_step:
            self.amp_scaler.update()

        with torch.no_grad():
            if grain_indices.max() <= 1:
                ratio = (grain_indices == 1).float().mean()
                loss_dict["fine_ratio"] = ratio
            else:
                loss_dict["fine_ratio"] = (grain_indices == 2).float().mean()
                loss_dict["median_ratio"] = (grain_indices == 1).float().mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0 and did_g_optimizer_step:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        with torch.no_grad():
            if hasattr(self, "net_g_ema"):
                self.net_g_ema.eval()
                self.output, _, _ = self.net_g_ema(self.gt)
            else:
                self.net_g.eval()
                self.output, _, _ = self.net_g(self.gt)
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
            img_name = osp.splitext(osp.basename(val_data.get("gt_path", [f"{idx:08d}"])[0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            gt_img = tensor2img([visuals["gt"]])
            if save_img:
                save_img_path = osp.join(self.opt["path"]["visualization"], dataset_name,
                                         f"{img_name}_{self.opt['name']}.png")
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
        out_dict["result"] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter, param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_network(self.net_d, "net_d", current_iter)
        self.save_training_state(epoch, current_iter)
