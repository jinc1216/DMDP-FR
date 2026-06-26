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


def format_stem_examples(stems, limit=10):
    stems = sorted(stems)
    shown = ", ".join(stems[:limit])
    if len(stems) > limit:
        shown += f", ... (+{len(stems) - limit} more)"
    return shown


def build_unique_stem_index(paths, suffix=None, label="images"):
    stem_index = {}
    duplicates = {}
    for path in paths:
        stem = strip_suffix(path.stem, suffix)
        if stem in stem_index:
            duplicates.setdefault(stem, [stem_index[stem]]).append(path)
        else:
            stem_index[stem] = path

    if duplicates:
        details = []
        for stem in sorted(duplicates):
            names = ", ".join(p.name for p in duplicates[stem][:5])
            if len(duplicates[stem]) > 5:
                names += f", ... (+{len(duplicates[stem]) - 5} more)"
            details.append(f"{stem}: {names}")
            if len(details) >= 10:
                break
        raise ValueError(
            f"Duplicate filename stems found in {label} after suffix normalization. "
            "Metrics would be ambiguous.\n"
            + "\n".join(details)
        )
    return stem_index


def validate_restored_names_match_lq(args, result_paths):
    if args.input_path is None:
        return

    input_paths = collect_image_paths(args.input_path, recursive=args.recursive)
    input_index = build_unique_stem_index(input_paths, label="input LQ images")
    restored_index = build_unique_stem_index(
        result_paths, suffix=args.suffix, label="restored images"
    )

    input_stems = set(input_index.keys())
    restored_stems = set(restored_index.keys())
    missing = input_stems - restored_stems
    extra = restored_stems - input_stems
    if missing or extra:
        message = [
            "Restored image names do not match input LQ image names by filename stem.",
            f"Input LQ count: {len(input_stems)}",
            f"Restored count: {len(restored_stems)}",
        ]
        if missing:
            message.append(
                f"Missing restored images for {len(missing)} LQ stems: "
                f"{format_stem_examples(missing)}"
            )
        if extra:
            message.append(
                f"Extra restored images without matching LQ stems: "
                f"{format_stem_examples(extra)}"
            )
        message.append(
            "Use a clean --save_root/--result_path, remove stale restored files, "
            "or verify --suffix/--recursive settings."
        )
        raise ValueError("\n".join(message))

    print(f"Restored/LQ filename check passed: {len(input_stems)} matched images.")


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
    if network_type == "DMGQVAE":
        output, _, info = net(batch)
        if not isinstance(info, dict) or "gate" not in info:
            raise ValueError("DMGQVAE visualization requires forward info['gate'].")
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
    network_type = opt["network_g"].get("type", "DMDP-FR")

    net = build_dmdp_fr(opt, args, device)
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

            if args.save_comparison:
                gt_path = gt_index.get(basename)
                save_comparison_image(lq_imgs[i], restored_img, gt_path, comparison_dir / save_name, img_size)
            if args.save_gate_map:
                gate_stats.append(save_latent_granularity_visualizations(
                    pred, i, restored_img, save_stem, gate_dirs, args.gate_overlay_alpha))

        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_latent_granularity_stats(gate_stats, args.save_root)
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
    validate_restored_names_match_lq(args, result_paths)
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
