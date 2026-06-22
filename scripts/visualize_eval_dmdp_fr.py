import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from basicsr.archs import build_network
from basicsr.archs.dmdp_fr_arch import make_dual_targets, make_triple_targets
from basicsr.utils import img2tensor, imwrite, tensor2img
from basicsr.utils.options import ordered_yaml

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to parse --opt network configs.") from exc


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DATASET_LEVEL_METRICS = {"fid", "kid", "fid_dinov2", "inception_score", "is"}
REFERENCE_DATASET_METRICS = {"fid", "kid", "fid_dinov2"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DMDP-FR results and evaluate saved images with pyiqa."
    )
    parser.add_argument("-i", "--input_path", type=str, default=None,
                        help="Input LQ image or folder. Required unless --skip_inference is set.")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="GT image or folder for full-reference metrics and comparison images.")
    parser.add_argument("-o", "--save_root", type=str, default="results/dmdp_fr_vis",
                        help="Output root. Restored images are saved to <save_root>/restored.")
    parser.add_argument("--result_path", type=str, default=None,
                        help="Existing restored image folder for --skip_inference. Defaults to <save_root>/restored.")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip model inference and only calculate pyiqa metrics from saved restored images.")

    parser.add_argument("--opt", type=str, default="options/DMDP-FR_stage3_triple.yml",
                        help="DMDP-FR option file. Its network_g section is used to build the model.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="DMDP-FR net_g checkpoint, for example experiments/.../models/net_g_latest.pth.")

    parser.add_argument("--img_size", type=int, default=None,
                        help="Resize inputs to this size. Defaults to network_g.img_size from --opt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--w", type=float, default=None,
                        help="Fidelity weight for DMDP-FR forward. Defaults to train.fidelity_weight from --opt.")
    parser.add_argument("--adain", action="store_true",
                        help="Enable adain=True in DMDP-FR forward.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string such as cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Optional suffix for saved restored image names.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively collect input images from folders.")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save side-by-side LQ/restored(/GT) images to <save_root>/comparison.")
    parser.add_argument("--save_gate_map", action="store_true",
                        help="Save DMDP-FR latent granularity maps. Each 8x8 gate cell is rendered "
                             "as its assembled region on the full latent grid.")
    parser.add_argument("--save_gate_probs", action="store_true",
                        help="Also save per-grain gate probability heatmaps. Requires --save_gate_map.")
    parser.add_argument("--gate_overlay_alpha", type=float, default=0.45,
                        help="Alpha for blending latent granularity map over restored image.")
    parser.add_argument("--save_code_error_map", action="store_true",
                        help="Save GT/predicted code maps and code mismatch maps. Requires --gt_path and "
                             "network_vqgan in --opt.")

    parser.add_argument("--metrics", type=str, default="",
                        help="Comma-separated pyiqa metrics, e.g. psnr,ssim,lpips,niqe. Empty means no evaluation.")
    parser.add_argument("--crop_border", type=int, default=4,
                        help="crop_border passed to pyiqa psnr/ssim when supported.")
    parser.add_argument("--test_y_channel", action="store_true",
                        help="Evaluate psnr/ssim on the Y channel when supported by pyiqa.")
    parser.add_argument("--fid_ref_path", type=str, default=None,
                        help="Reference image folder for distribution metrics such as FID/KID. "
                             "If omitted, --gt_path is used when available.")
    parser.add_argument("--fid_dataset_name", type=str, default=None,
                        help="pyiqa built-in reference dataset_name for FID/KID when no reference folder is given.")
    parser.add_argument("--fid_dataset_res", type=int, default=None,
                        help="Optional pyiqa dataset_res for built-in FID/KID reference statistics.")
    parser.add_argument("--fid_dataset_split", type=str, default=None,
                        help="Optional pyiqa dataset_split for built-in FID/KID reference statistics.")
    parser.add_argument("--metric_output", type=str, default=None,
                        help="Metric output prefix. Defaults to <save_root>/metrics.")
    return parser.parse_args()


def load_yaml(opt_path):
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        return yaml.load(f, Loader=Loader)


def collect_image_paths(path, recursive=False):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    pattern = "**/*" if recursive else "*"
    img_paths = [p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(img_paths)


def build_gt_index(gt_path, recursive=False):
    if gt_path is None:
        return {}
    gt_paths = collect_image_paths(gt_path, recursive=recursive)
    return {p.stem: p for p in gt_paths}


def strip_suffix(stem, suffix):
    if suffix is None:
        return stem
    suffix = f"_{suffix}"
    if stem.endswith(suffix):
        return stem[:-len(suffix)]
    return stem


def get_mean_std(opt):
    dataset_opt = opt.get("datasets", {}).get("val", None) or opt.get("datasets", {}).get("train", {})
    mean = tuple(dataset_opt.get("mean", [0.5, 0.5, 0.5]))
    std = tuple(dataset_opt.get("std", [0.5, 0.5, 0.5]))
    return mean, std


def infer_tensor_min_max(mean, std):
    if all(abs(v - 0.5) < 1e-8 for v in mean) and all(abs(v - 0.5) < 1e-8 for v in std):
        return (-1, 1)
    if all(abs(v) < 1e-8 for v in mean) and all(abs(v - 1.0) < 1e-8 for v in std):
        return (0, 1)
    raise ValueError(
        f"Cannot infer output tensor range from mean={mean}, std={std}. "
        "Set the dataset mean/std to [0.5]/[0.5] or [0]/[1], or update the script explicitly."
    )


def build_dmdp_fr(opt, args, device):
    if args.ckpt_path is None:
        raise ValueError("--ckpt_path is required unless --skip_inference is set.")

    net = build_network(opt["network_g"]).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def preprocess_image(img_path, img_size, mean, std):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(img_tensor, mean, std, inplace=True)
    return img, img_tensor


def save_comparison_image(lq_img, restored_img, gt_path, save_path, img_size):
    panels = [lq_img, restored_img]
    if gt_path is not None:
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if gt_img is not None:
            if gt_img.shape[:2] != (img_size, img_size):
                gt_img = cv2.resize(gt_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            panels.append(gt_img)
    comparison = np.concatenate(panels, axis=1)
    imwrite(comparison, str(save_path))


def resize_bgr_to_size(img, out_size, interpolation=cv2.INTER_LINEAR):
    out_w, out_h = out_size
    if img.shape[1] == out_w and img.shape[0] == out_h:
        return img
    return cv2.resize(img, (out_w, out_h), interpolation=interpolation)


def add_panel_label(img, label, label_h=34):
    canvas = np.full((img.shape[0] + label_h, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_h:, :, :] = img
    cv2.putText(
        canvas,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_labeled_row(panels):
    labeled = [add_panel_label(img, label) for label, img in panels]
    return np.concatenate(labeled, axis=1)


def make_code_color_lut(num_codes):
    ids = np.arange(max(int(num_codes), 1), dtype=np.uint32)
    lut = np.stack([
        (ids * 109 + 101) % 256,
        (ids * 73 + 59) % 256,
        (ids * 37 + 17) % 256,
    ], axis=1).astype(np.uint8)
    lut[0] = np.array([40, 40, 40], dtype=np.uint8)
    return lut


def tensor_map_to_numpy(tensor):
    tensor = tensor.detach().long().cpu()
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.dim() == 3:
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise ValueError(f"Expected a 2D map or batched map, got shape {tuple(tensor.shape)}.")
    return tensor.numpy()


def render_code_map(code_map, out_size, codebook_size):
    code_map = np.asarray(code_map, dtype=np.int64)
    max_code = int(code_map.max()) if code_map.size else 0
    lut_size = max(int(codebook_size), max_code + 1)
    lut = make_code_color_lut(lut_size)
    color = lut[np.clip(code_map, 0, lut_size - 1)]
    return cv2.resize(color, out_size, interpolation=cv2.INTER_NEAREST)


def render_error_map(error_map, out_size, valid_mask=None):
    error_map = np.asarray(error_map, dtype=bool)
    canvas = np.full((*error_map.shape, 3), (245, 245, 245), dtype=np.uint8)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        canvas[:, :, :] = (220, 220, 220)
        canvas[valid_mask] = (245, 245, 245)
        canvas[valid_mask & error_map] = (0, 0, 230)
    else:
        canvas[error_map] = (0, 0, 230)
    return cv2.resize(canvas, out_size, interpolation=cv2.INTER_NEAREST)


def masked_error_rate(error_map, valid_mask=None):
    error_map = np.asarray(error_map, dtype=bool)
    if valid_mask is None:
        return float(error_map.mean()) if error_map.size else float("nan"), int(error_map.size)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    denom = int(valid_mask.sum())
    if denom == 0:
        return float("nan"), 0
    return float((error_map & valid_mask).sum() / denom), denom


def get_code_stage_specs(pred):
    if "median" in pred:
        return [("coarse", "coarse"), ("median", "medium"), ("fine", "fine")]
    return [("coarse", "coarse"), ("fine", "fine")]


def build_code_targets(idx_gt, grain_gt, grain_type):
    if grain_type == "triple":
        return make_triple_targets(idx_gt, grain_gt)
    if grain_type == "dual":
        return make_dual_targets(idx_gt, grain_gt)
    raise ValueError(f"Unsupported DQ grain_type for code error visualization: {grain_type}")


def build_gt_vqgan(opt, device):
    if "network_vqgan" not in opt:
        raise ValueError("--save_code_error_map requires a network_vqgan section in the option file.")
    vqgan = build_network(opt["network_vqgan"]).to(device)
    vqgan.eval()
    for param in vqgan.parameters():
        param.requires_grad = False
    return vqgan


def save_code_error_stats(rows, save_root):
    if not rows:
        return
    stats_path = Path(save_root) / "code_error_stats.csv"
    fieldnames = ["image", "latent_h", "latent_w"]
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_fields)
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Code error statistics saved to: {stats_path}")


def save_code_error_visualizations(
    net,
    vqgan,
    pred,
    batch_idx,
    lq_img,
    gt_img,
    gt_tensor,
    restored_img,
    save_stem,
    code_dirs,
    codebook_size,
    grain_type,
    device,
):
    if not hasattr(net, "logits_to_code_map"):
        raise ValueError("--save_code_error_map requires a DMDP-FR network with logits_to_code_map().")

    pred_single = {
        key: value[batch_idx:batch_idx + 1]
        for key, value in pred.items()
        if torch.is_tensor(value)
    }
    with torch.no_grad():
        idx_gt, grain_gt, _ = vqgan.encode_to_indices(gt_tensor.unsqueeze(0).to(device))
        pred_final, pred_grain, _ = net.logits_to_code_map(pred_single)

    idx_gt = idx_gt.long()
    grain_gt = grain_gt.long()
    pred_final = pred_final.long()
    pred_grain = pred_grain.long()
    targets = build_code_targets(idx_gt, grain_gt, grain_type)

    out_size = (restored_img.shape[1], restored_img.shape[0])
    gt_final_np = tensor_map_to_numpy(idx_gt)
    pred_final_np = tensor_map_to_numpy(pred_final)
    err_final_np = pred_final_np != gt_final_np
    gt_grain_np = tensor_map_to_numpy(grain_gt)
    pred_grain_np = tensor_map_to_numpy(pred_grain)
    route_err_np = pred_grain_np != gt_grain_np

    npz_payload = {
        "gt_final_code": gt_final_np,
        "pred_final_code": pred_final_np,
        "final_selected_error": err_final_np.astype(np.uint8),
        "gt_grain": gt_grain_np,
        "pred_grain": pred_grain_np,
        "route_error": route_err_np.astype(np.uint8),
    }

    stats = {
        "image": f"{save_stem}.png",
        "latent_h": int(gt_final_np.shape[0]),
        "latent_w": int(gt_final_np.shape[1]),
    }

    final_gt_img = render_code_map(gt_final_np, out_size, codebook_size)
    final_pred_img = render_code_map(pred_final_np, out_size, codebook_size)
    final_err_img = render_error_map(err_final_np, out_size)
    imwrite(final_gt_img, str(code_dirs["maps"] / f"{save_stem}_gt_final_selected_code.png"))
    imwrite(final_pred_img, str(code_dirs["maps"] / f"{save_stem}_pred_final_selected_code.png"))
    imwrite(final_err_img, str(code_dirs["errors"] / f"{save_stem}_final_selected_code_error.png"))
    imwrite(render_error_map(route_err_np, out_size), str(code_dirs["errors"] / f"{save_stem}_route_error.png"))

    final_rate, final_count = masked_error_rate(err_final_np)
    route_rate, route_count = masked_error_rate(route_err_np)
    stats["final_selected_error_rate"] = final_rate
    stats["final_selected_count"] = final_count
    stats["route_error_rate"] = route_rate
    stats["route_count"] = route_count

    stage_error_panels = []
    for pred_key, out_name in get_code_stage_specs(pred_single):
        pred_stage = pred_single[pred_key].argmax(dim=1).long()
        gt_stage = targets[pred_key].long()
        mask_stage = targets.get(f"mask_{pred_key}", None)

        pred_stage_np = tensor_map_to_numpy(pred_stage)
        gt_stage_np = tensor_map_to_numpy(gt_stage)
        err_stage_np = pred_stage_np != gt_stage_np
        mask_stage_np = tensor_map_to_numpy(mask_stage) if mask_stage is not None else None

        gt_stage_img = render_code_map(gt_stage_np, out_size, codebook_size)
        pred_stage_img = render_code_map(pred_stage_np, out_size, codebook_size)
        err_stage_img = render_error_map(err_stage_np, out_size, valid_mask=mask_stage_np)
        imwrite(gt_stage_img, str(code_dirs["maps"] / f"{save_stem}_gt_{out_name}_code.png"))
        imwrite(pred_stage_img, str(code_dirs["maps"] / f"{save_stem}_pred_{out_name}_code.png"))
        imwrite(err_stage_img, str(code_dirs["errors"] / f"{save_stem}_{out_name}_code_error.png"))

        selected_rate, selected_count = masked_error_rate(err_stage_np, mask_stage_np)
        all_rate, all_count = masked_error_rate(err_stage_np)
        stats[f"{out_name}_error_rate"] = selected_rate
        stats[f"{out_name}_selected_count"] = selected_count
        stats[f"{out_name}_error_rate_all"] = all_rate
        stats[f"{out_name}_count_all"] = all_count

        npz_payload[f"gt_{out_name}_code"] = gt_stage_np
        npz_payload[f"pred_{out_name}_code"] = pred_stage_np
        npz_payload[f"{out_name}_error"] = err_stage_np.astype(np.uint8)
        if mask_stage_np is not None:
            npz_payload[f"{out_name}_valid_mask"] = mask_stage_np.astype(np.uint8)
        stage_error_panels.append((f"{out_name.title()} Error", err_stage_img))

    stage_error_panels.append(("Final Selected Error", final_err_img))
    stage_summary = make_labeled_row(stage_error_panels)
    imwrite(stage_summary, str(code_dirs["figures"] / f"{save_stem}_multiscale_code_error_figure.png"))

    lq_panel = resize_bgr_to_size(lq_img, out_size)
    gt_panel = resize_bgr_to_size(gt_img, out_size)
    final_figure = make_labeled_row([
        ("LQ input", lq_panel),
        ("GT code map", final_gt_img),
        ("Predicted code map", final_pred_img),
        ("Error map", final_err_img),
        ("Restored image", restored_img),
        ("GT", gt_panel),
    ])
    imwrite(final_figure, str(code_dirs["figures"] / f"{save_stem}_final_selected_code_error_figure.png"))
    np.savez_compressed(code_dirs["raw"] / f"{save_stem}_code_maps.npz", **npz_payload)
    return stats


def get_gate_class_names(num_classes):
    if num_classes == 2:
        return ["coarse", "fine"]
    if num_classes == 3:
        return ["coarse", "median", "fine"]
    return [f"gate_{idx}" for idx in range(num_classes)]


def get_gate_colors(num_classes):
    # BGR colors: coarse=blue, median=orange, fine=green. Extra classes use OpenCV color maps.
    base_colors = np.array([
        [220, 80, 30],
        [0, 180, 255],
        [70, 210, 70],
        [220, 70, 220],
        [80, 220, 220],
        [180, 180, 180],
    ], dtype=np.uint8)
    if num_classes <= len(base_colors):
        return base_colors[:num_classes]

    values = np.linspace(0, 255, num_classes, dtype=np.uint8).reshape(num_classes, 1)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(num_classes, 3)


def infer_latent_grid_shape(pred, gate_shape, num_classes):
    fine_logits = pred.get("fine", None)
    if torch.is_tensor(fine_logits) and fine_logits.dim() >= 4:
        return int(fine_logits.shape[-2]), int(fine_logits.shape[-1])

    full_indices = pred.get("min_encoding_indices", None)
    if torch.is_tensor(full_indices):
        if full_indices.dim() >= 4 and full_indices.shape[-1] == 1:
            return int(full_indices.shape[-3]), int(full_indices.shape[-2])
        if full_indices.dim() >= 3:
            return int(full_indices.shape[-2]), int(full_indices.shape[-1])

    median_logits = pred.get("median", None)
    if torch.is_tensor(median_logits) and median_logits.dim() >= 4:
        return int(median_logits.shape[-2] * 2), int(median_logits.shape[-1] * 2)

    scale = 4 if num_classes == 3 else 2
    return int(gate_shape[0] * scale), int(gate_shape[1] * scale)


def expand_gate_to_latent_grid(gate_map, latent_shape):
    gate_h, gate_w = gate_map.shape
    latent_h, latent_w = latent_shape
    if latent_h % gate_h == 0 and latent_w % gate_w == 0:
        return np.repeat(
            np.repeat(gate_map, latent_h // gate_h, axis=0),
            latent_w // gate_w,
            axis=1,
        )

    resized = cv2.resize(gate_map.astype(np.float32), (latent_w, latent_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(gate_map.dtype, copy=False)


def expand_gate_prob_to_latent_grid(gate_prob, latent_shape):
    gate_np = gate_prob.numpy()
    return np.stack([
        expand_gate_to_latent_grid(gate_np[class_idx], latent_shape)
        for class_idx in range(gate_np.shape[0])
    ], axis=0)


def get_grain_subblock_size(class_idx, super_h, super_w, num_classes):
    if num_classes == 3:
        if class_idx == 0:
            return super_h, super_w
        if class_idx == 1:
            return max(1, super_h // 2), max(1, super_w // 2)
        if class_idx == 2:
            return 1, 1
    if num_classes == 2:
        if class_idx == 0:
            return super_h, super_w
        if class_idx == 1:
            return 1, 1
    return 1, 1


def latent_box_to_pixels(row0, row1, col0, col1, latent_shape, out_size):
    latent_h, latent_w = latent_shape
    out_w, out_h = out_size
    x0 = int(round(col0 * out_w / latent_w))
    x1 = int(round(col1 * out_w / latent_w))
    y0 = int(round(row0 * out_h / latent_h))
    y1 = int(round(row1 * out_h / latent_h))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def render_latent_granularity_map(gate_idx, latent_shape, out_size, colors, border_px=1):
    gate_h, gate_w = gate_idx.shape
    latent_h, latent_w = latent_shape
    if latent_h % gate_h != 0 or latent_w % gate_w != 0:
        raise ValueError(
            f"Latent grid {latent_shape} must be an integer expansion of gate grid {gate_idx.shape}."
        )

    super_h = latent_h // gate_h
    super_w = latent_w // gate_w
    out_w, out_h = out_size
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    border_color = (255, 255, 255)

    for gate_row in range(gate_h):
        for gate_col in range(gate_w):
            class_idx = int(gate_idx[gate_row, gate_col])
            color = colors[np.clip(class_idx, 0, len(colors) - 1)].tolist()
            sub_h, sub_w = get_grain_subblock_size(class_idx, super_h, super_w, len(colors))
            latent_row0 = gate_row * super_h
            latent_col0 = gate_col * super_w

            for row in range(latent_row0, latent_row0 + super_h, sub_h):
                for col in range(latent_col0, latent_col0 + super_w, sub_w):
                    row1 = min(row + sub_h, latent_row0 + super_h)
                    col1 = min(col + sub_w, latent_col0 + super_w)
                    x0, y0, x1, y1 = latent_box_to_pixels(row, row1, col, col1, latent_shape, out_size)
                    canvas[y0:y1, x0:x1] = color
                    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), border_color, border_px)
    return canvas


def save_latent_granularity_visualizations(pred, batch_idx, restored_img, save_stem, gate_dirs, alpha):
    if not isinstance(pred, dict) or "gate" not in pred:
        raise ValueError("DMDP-FR latent granularity visualization requires pred['gate'].")

    gate_tensor = pred["gate"][batch_idx].detach().float().cpu()
    gate_prob = gate_tensor if pred.get("gate_is_prob", False) else torch.softmax(gate_tensor, dim=0)
    gate_idx = gate_prob.argmax(dim=0).numpy().astype(np.int64)
    num_classes = int(gate_prob.shape[0])
    class_names = get_gate_class_names(num_classes)
    colors = get_gate_colors(num_classes)
    latent_shape = infer_latent_grid_shape(pred, gate_idx.shape, num_classes)
    latent_route_idx = expand_gate_to_latent_grid(gate_idx, latent_shape).astype(np.int64)

    out_size = (restored_img.shape[1], restored_img.shape[0])
    gate_map = render_latent_granularity_map(
        gate_idx=gate_idx,
        latent_shape=latent_shape,
        out_size=out_size,
        colors=colors,
        border_px=1,
    )
    overlay = cv2.addWeighted(restored_img, 1.0 - alpha, gate_map, alpha, 0.0)

    imwrite(gate_map, str(gate_dirs["granularity"] / f"{save_stem}_latent_granularity_{latent_shape[0]}x{latent_shape[1]}.png"))
    imwrite(overlay, str(gate_dirs["overlay"] / f"{save_stem}_latent_granularity_overlay.png"))

    if gate_dirs.get("prob") is not None:
        latent_prob = expand_gate_prob_to_latent_grid(gate_prob, latent_shape)
        for class_idx, class_name in enumerate(class_names):
            prob = latent_prob[class_idx]
            prob = cv2.resize(prob, out_size, interpolation=cv2.INTER_NEAREST)
            prob_img = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
            prob_heat = cv2.applyColorMap(prob_img, cv2.COLORMAP_TURBO)
            imwrite(prob_heat, str(gate_dirs["prob"] / f"{save_stem}_{class_name}_latent_prob.png"))

    counts = np.bincount(latent_route_idx.reshape(-1), minlength=num_classes).astype(np.float64)
    ratios = counts / max(float(latent_route_idx.size), 1.0)
    stats = {
        "image": f"{save_stem}.png",
        "gate_h": int(gate_idx.shape[0]),
        "gate_w": int(gate_idx.shape[1]),
        "latent_h": int(latent_shape[0]),
        "latent_w": int(latent_shape[1]),
    }
    for class_idx, class_name in enumerate(class_names):
        stats[f"{class_name}_latent_area_ratio"] = float(ratios[class_idx])
        stats[f"{class_name}_gate_mean_prob"] = float(gate_prob[class_idx].mean().item())
    return stats


def save_latent_granularity_stats(rows, save_root):
    if not rows:
        return
    stats_path = Path(save_root) / "latent_granularity_stats.csv"
    fieldnames = ["image", "gate_h", "gate_w", "latent_h", "latent_w"]
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_fields)
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latent granularity statistics saved to: {stats_path}")


def forward_for_visualization(net, batch, fidelity_weight, adain, network_type):
    if network_type == "DQDynamicVQVAE":
        output, _, info = net(batch)
        if not isinstance(info, dict) or "gate" not in info:
            raise ValueError("DQDynamicVQVAE visualization requires forward info['gate'].")
        pred = {
            "gate": info["gate"],
            "gate_is_prob": True,
        }
        if "min_encoding_indices" in info:
            pred["min_encoding_indices"] = info["min_encoding_indices"]
        return output, pred

    output, pred, _ = net(batch, w=fidelity_weight, adain=adain)
    return output, pred


def run_inference(opt, args, device):
    input_paths = collect_image_paths(args.input_path, recursive=args.recursive)
    if len(input_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.input_path}.")

    mean, std = get_mean_std(opt)
    tensor_min_max = infer_tensor_min_max(mean, std)
    img_size = args.img_size or int(opt["network_g"].get("img_size", 512))
    fidelity_weight = args.w
    if fidelity_weight is None:
        fidelity_weight = float(opt.get("train", {}).get("fidelity_weight", 1.0))
    network_type = opt["network_g"].get("type", "DMDPFR")

    net = build_dmdp_fr(opt, args, device)
    vqgan = None
    code_error_dirs = None
    code_error_stats = []
    codebook_size = int(opt["network_g"].get("codebook_size", opt.get("network_vqgan", {}).get("codebook_size", 1024)))
    grain_type = opt["network_g"].get("grain_type", "triple")
    restored_dir = Path(args.save_root) / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = Path(args.save_root) / "comparison"
    if args.save_comparison:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    gate_dirs = None
    gate_stats = []
    if args.save_gate_map:
        gate_dirs = {
            "granularity": Path(args.save_root) / "latent_granularity_maps",
            "overlay": Path(args.save_root) / "latent_granularity_overlays",
            "prob": Path(args.save_root) / "latent_granularity_probs" if args.save_gate_probs else None,
        }
        gate_dirs["granularity"].mkdir(parents=True, exist_ok=True)
        gate_dirs["overlay"].mkdir(parents=True, exist_ok=True)
        if gate_dirs["prob"] is not None:
            gate_dirs["prob"].mkdir(parents=True, exist_ok=True)
    gt_index = build_gt_index(args.gt_path, recursive=args.recursive)
    if args.save_code_error_map:
        if args.gt_path is None:
            raise ValueError("--save_code_error_map requires --gt_path.")
        if not hasattr(net, "logits_to_code_map"):
            raise ValueError("--save_code_error_map currently supports DMDP-FR inference only.")
        vqgan = build_gt_vqgan(opt, device)
        code_error_dirs = {
            "maps": Path(args.save_root) / "code_maps",
            "errors": Path(args.save_root) / "code_error_maps",
            "figures": Path(args.save_root) / "code_error_figures",
            "raw": Path(args.save_root) / "code_maps_npz",
        }
        for path in code_error_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    print(f"Running DMDP-FR inference on {len(input_paths)} images.")
    print(f"Save restored images to: {restored_dir}")
    for start in tqdm(range(0, len(input_paths), args.batch_size), desc="Inference"):
        batch_paths = input_paths[start:start + args.batch_size]
        tensors = []
        lq_imgs = []
        for img_path in batch_paths:
            lq_img, tensor = preprocess_image(img_path, img_size, mean, std)
            tensors.append(tensor)
            lq_imgs.append(lq_img)
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            output, pred = forward_for_visualization(net, batch, fidelity_weight, args.adain, network_type)

        for i, img_path in enumerate(batch_paths):
            basename = img_path.stem
            save_name = f"{basename}.png" if args.suffix is None else f"{basename}_{args.suffix}.png"
            save_stem = Path(save_name).stem
            restored_img = tensor2img(output[i], rgb2bgr=True, min_max=tensor_min_max)
            save_path = restored_dir / save_name
            imwrite(restored_img, str(save_path))

            gt_path = gt_index.get(basename)
            if args.save_comparison:
                save_comparison_image(lq_imgs[i], restored_img, gt_path, comparison_dir / save_name, img_size)
            if args.save_gate_map:
                gate_stats.append(save_latent_granularity_visualizations(
                    pred, i, restored_img, save_stem, gate_dirs, args.gate_overlay_alpha))
            if args.save_code_error_map:
                if gt_path is None:
                    raise FileNotFoundError(
                        f"No GT image matched input {img_path.name}; expected a GT file with stem {basename}."
                    )
                gt_img, gt_tensor = preprocess_image(gt_path, img_size, mean, std)
                code_error_stats.append(save_code_error_visualizations(
                    net=net,
                    vqgan=vqgan,
                    pred=pred,
                    batch_idx=i,
                    lq_img=lq_imgs[i],
                    gt_img=gt_img,
                    gt_tensor=gt_tensor,
                    restored_img=restored_img,
                    save_stem=save_stem,
                    code_dirs=code_error_dirs,
                    codebook_size=codebook_size,
                    grain_type=grain_type,
                    device=device,
                ))

        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_latent_granularity_stats(gate_stats, args.save_root)
    save_code_error_stats(code_error_stats, args.save_root)
    return restored_dir


def scalarize_score(score):
    if torch.is_tensor(score):
        return float(score.detach().cpu().mean().item())
    if isinstance(score, (list, tuple)):
        vals = [scalarize_score(v) for v in score]
        return float(np.mean(vals))
    return float(score)


def get_distribution_metric_kwargs(args):
    kwargs = {}
    if args.fid_dataset_name is not None:
        kwargs["dataset_name"] = args.fid_dataset_name
    if args.fid_dataset_res is not None:
        kwargs["dataset_res"] = args.fid_dataset_res
    if args.fid_dataset_split is not None:
        kwargs["dataset_split"] = args.fid_dataset_split
    return kwargs


def evaluate_one_metric(metric_name, result_paths, gt_index, args, device):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError("pyiqa is required for metric calculation. Install it with `pip install pyiqa`.") from exc

    metric_key = metric_name.lower()
    metric_kwargs = {}
    if metric_key in {"psnr", "ssim"}:
        metric_kwargs["crop_border"] = args.crop_border
        metric_kwargs["test_y_channel"] = args.test_y_channel
    try:
        metric = pyiqa.create_metric(metric_name, device=device, as_loss=False, **metric_kwargs)
    except TypeError:
        metric = pyiqa.create_metric(metric_name, device=device, as_loss=False)

    if metric_key in DATASET_LEVEL_METRICS:
        result_dir = str(Path(args.result_path or Path(args.save_root) / "restored"))
        ref_dir = args.fid_ref_path or args.gt_path
        if metric_key in REFERENCE_DATASET_METRICS:
            if ref_dir is not None:
                score = metric(result_dir, ref_dir)
            elif args.fid_dataset_name is not None:
                score = metric(result_dir, **get_distribution_metric_kwargs(args))
            else:
                raise ValueError(
                    f"{metric_name} requires a reference distribution. Provide --fid_ref_path "
                    "or --gt_path with reference images, or provide --fid_dataset_name for pyiqa "
                    "built-in reference statistics."
                )
        else:
            score = metric(result_dir)
        return None, scalarize_score(score)

    per_image = {}
    for result_path in tqdm(result_paths, desc=f"Metric {metric_name}"):
        gt_stem = strip_suffix(result_path.stem, args.suffix)
        gt_path = gt_index.get(gt_stem)
        try:
            if gt_path is not None:
                score = metric(str(result_path), str(gt_path))
            else:
                score = metric(str(result_path))
        except TypeError as exc:
            if gt_path is None:
                raise RuntimeError(
                    f"Metric {metric_name} appears to require GT, but --gt_path did not provide a match for "
                    f"{result_path.name}."
                ) from exc
            score = metric(str(result_path))
        per_image[result_path.name] = scalarize_score(score)
    avg_score = float(np.mean(list(per_image.values()))) if per_image else float("nan")
    return per_image, avg_score


def evaluate_saved_results(args, device):
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_names:
        return

    result_dir = Path(args.result_path) if args.result_path is not None else Path(args.save_root) / "restored"
    result_paths = collect_image_paths(result_dir, recursive=False)
    if len(result_paths) == 0:
        raise FileNotFoundError(f"No restored images found in {result_dir}.")
    gt_index = build_gt_index(args.gt_path, recursive=args.recursive)

    metric_prefix = Path(args.metric_output) if args.metric_output else Path(args.save_root) / "metrics"
    metric_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    per_image_rows = {p.name: {"image": p.name} for p in result_paths}

    for metric_name in metric_names:
        per_image, avg_score = evaluate_one_metric(metric_name, result_paths, gt_index, args, device)
        summary[metric_name] = avg_score
        if per_image is not None:
            for image_name, score in per_image.items():
                per_image_rows[image_name][metric_name] = score

    summary_path = metric_prefix.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metric summary saved to: {summary_path}")
    for metric_name, score in summary.items():
        print(f"{metric_name}: {score:.6f}")

    if any(len(row) > 1 for row in per_image_rows.values()):
        csv_path = metric_prefix.with_suffix(".csv")
        fieldnames = ["image"] + metric_names
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_name in sorted(per_image_rows.keys()):
                writer.writerow(per_image_rows[image_name])
        print(f"Per-image metrics saved to: {csv_path}")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}.")
    if args.skip_inference and args.save_code_error_map:
        raise ValueError("--save_code_error_map requires model inference; remove --skip_inference.")
    if args.skip_inference:
        if args.result_path is None:
            args.result_path = str(Path(args.save_root) / "restored")
    else:
        if args.input_path is None:
            raise ValueError("--input_path is required unless --skip_inference is set.")
        opt = load_yaml(args.opt)
        restored_dir = run_inference(opt, args, torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")))
        args.result_path = str(restored_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate_saved_results(args, device)


if __name__ == "__main__":
    main()
