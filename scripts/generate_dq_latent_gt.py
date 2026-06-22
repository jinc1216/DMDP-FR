import argparse
import glob
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from basicsr.archs import build_network
from basicsr.utils import img2tensor
from basicsr.utils.options import ordered_yaml

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to parse --opt network configs.") from exc


def build_dqvae(args):
    if args.opt is not None:
        with open(args.opt, mode="r") as f:
            Loader, _ = ordered_yaml()
            opt = yaml.load(f, Loader=Loader)
        if opt.get("network_vqgan") is not None:
            net_opt = deepcopy(opt["network_vqgan"])
        elif opt.get("network_g", {}).get("type") == "DQDynamicVQVAE":
            net_opt = deepcopy(opt["network_g"])
        else:
            raise ValueError(
                "--opt must contain network_vqgan or a DQDynamicVQVAE network_g "
                "for latent GT extraction."
            )
        if args.ckpt_path is not None:
            net_opt["model_path"] = args.ckpt_path
        elif net_opt.get("model_path") is None:
            raise ValueError("--ckpt_path is required when the selected network config has no model_path.")
        args.grain_type = net_opt.get("grain_type", args.grain_type)
        args.img_size = net_opt.get("img_size", args.img_size)
        args.codebook_size = net_opt.get("codebook_size", args.codebook_size)
        return build_network(net_opt)

    if args.grain_type == "triple":
        ch_mult = [1, 1, 2, 2, 4, 4, 4]
        attn = [args.img_size // 64, args.img_size // 32, args.img_size // 16]
        decoder_ch_mult = [1, 1, 2, 2, 4]
    else:
        ch_mult = [1, 1, 2, 2, 4, 4, 4]
        attn = [args.img_size // 64, args.img_size // 32]
        decoder_ch_mult = [1, 1, 2, 2, 4, 4]
    latent_size = args.img_size // (2 ** (len(decoder_ch_mult) - 1))
    opt = dict(
        type="DQDynamicVQVAE",
        grain_type=args.grain_type,
        img_size=args.img_size,
        ch=128,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=attn,
        in_channels=3,
        z_channels=256,
        router_config={"num_channels": 256, "normalization_type": "group-32", "gate_type": "2layer-fc-SiLu"},
        decoder_ch_mult=decoder_ch_mult,
        latent_size=latent_size,
        decoder_attn_resolutions=[latent_size],
        decoder_position_type="fourier+learned",
        codebook_size=args.codebook_size,
        codebook_dim=256,
        model_path=args.ckpt_path,
    )
    return build_network(opt)


def collect_image_paths(test_path):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    img_paths = []
    for pat in patterns:
        img_paths.extend(glob.glob(os.path.join(test_path, pat)))
    return sorted(set(img_paths))


def preprocess_image(img_path, img_size, hflip=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    if hflip:
        img = cv2.flip(img, 1)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return img


def code_dtype(codebook_size):
    return np.int16 if codebook_size <= np.iinfo(np.int16).max else np.int32


def save_latents_for_aug(vqgan, img_paths, aug, args, device, latent):
    hflip = aug == "hflip"
    desc = f"{aug} latent"
    idx_dtype = code_dtype(args.codebook_size)
    for start in tqdm(range(0, len(img_paths), args.batch_size), desc=desc):
        batch_paths = img_paths[start:start + args.batch_size]
        imgs = []
        valid_paths = []
        for img_path in batch_paths:
            try:
                imgs.append(preprocess_image(img_path, args.img_size, hflip=hflip))
                valid_paths.append(img_path)
            except FileNotFoundError as exc:
                print(exc)
        if not imgs:
            continue
        img_tensor = torch.stack(imgs, dim=0).to(device)
        with torch.no_grad():
            indices, grain_indices, _ = vqgan.encode_to_indices(img_tensor)
        indices = indices.cpu().numpy().astype(idx_dtype, copy=False)
        grain_indices = grain_indices.cpu().numpy().astype(np.int8, copy=False)
        for i, img_path in enumerate(valid_paths):
            name = os.path.basename(img_path)[:-4]
            latent[aug][name] = {
                "indices": indices[i],
                "grain_indices": grain_indices[i],
            }
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_path", type=str, default="datasets/ffhq/ffhq_512",
                        help="Training GT image folder.")
    parser.add_argument("-o", "--save_root", type=str, default="./experiments/pretrained_models/dqvae")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="DQ-VAE checkpoint. Overrides model_path from --opt when both are set.")
    parser.add_argument("--opt", type=str, default=None,
                        help="Option yml containing network_vqgan or DQDynamicVQVAE network_g.")
    parser.add_argument("--grain_type", type=str, default="triple", choices=["dual", "triple"])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_hflip", action="store_true", help="Only save original-image latent codes.")
    args = parser.parse_args()

    if args.opt is None and args.ckpt_path is None:
        raise ValueError("--ckpt_path is required when --opt is not provided.")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, but got {args.batch_size}.")

    if args.save_root.endswith("/"):
        args.save_root = args.save_root[:-1]
    os.makedirs(args.save_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan = build_dqvae(args).to(device).eval()

    latent = {"orig": {}, "hflip": {}}
    img_paths = collect_image_paths(args.test_path)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No jpg/jpeg/png images found in {args.test_path}.")

    for aug in ["orig"] if args.no_hflip else ["orig", "hflip"]:
        save_latents_for_aug(vqgan, img_paths, aug, args, device, latent)

    latent_save_path = os.path.join(args.save_root, f"latent_gt_dq_{args.grain_type}_code{args.codebook_size}.pth")
    torch.save(latent, latent_save_path)
    print(f"\nSaved {len(img_paths)} DQ latent GT records to {latent_save_path}")
