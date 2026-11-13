#!/usr/bin/env python3
"""Evaluate saved face-restoration images, independently of any model repository.

Dependencies: torch, torchvision, numpy, Pillow, scikit-image, pyiqa, facexlib.
Paired defaults: FID, LPIPS, IDA, PSNR, SSIM. Unpaired defaults: FID, MUSIQ.
FID always needs a real-image distribution, including in unpaired mode.
IDA defaults to the ArcFace IR-SE50 embedding angle in degrees (lower is better).
Inputs must already be aligned face crops for a meaningful IDA comparison.
No detection, geometric alignment, brightness matching, or fidelity-image resize
is applied. Only ArcFace inputs are resized to its required 112x112 resolution.

Examples (run from the dataset project directory):
  python evaluate_face_metrics.py --mode paired \
    --pred-dir MCR/LLFormer/results/LLFormer_darkface_256/test \
    --gt-dir darkface_256_backup/test/gt --output-dir metrics/LLFormer_darkface
  python evaluate_face_metrics.py --mode unpaired \
    --pred-dir MCR/DarkIR/images/results/WildDark-Face \
    --fid-ref-dir darkface_256_backup/test/gt --output-dir metrics/DarkIR_wildface

summary.json contains dataset-level FID and means of per-image metrics.
per_image.csv contains individual scores; FID is never reported per image.
An exact reconstruction has infinite PSNR, encoded as "Infinity" in JSON.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
PAIRED_METRICS = ('fid', 'lpips', 'ida', 'psnr', 'ssim')
UNPAIRED_METRICS = ('fid', 'musiq')
ARC_FACE_URL = (
    'https://github.com/xinntao/facexlib/releases/download/v0.1.0/'
    'recognition_arcface_ir_se50.pth'
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('paired', 'unpaired'), required=True)
    parser.add_argument('--pred-dir', type=Path, required=True, help='Saved restored images, not LQ inputs.')
    parser.add_argument('--gt-dir', type=Path, help='Paired normal-light GT; required in paired mode.')
    parser.add_argument('--fid-ref-dir', type=Path, help='Real-image distribution for FID; defaults to GT in paired mode.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for summary.json and per_image.csv.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0.')
    parser.add_argument('--metrics', nargs='+', choices=sorted(set(PAIRED_METRICS + UNPAIRED_METRICS)),
                        help='Optional subset of the metrics supported by the chosen mode.')
    parser.add_argument('--recursive', action='store_true', help='Include image subdirectories.')
    parser.add_argument('--match-by', choices=('relative-stem', 'stem'), default='relative-stem',
                        help='Pair by relative path without extension, or globally unique filename stem.')
    parser.add_argument('--pred-suffix', default='', help='Trailing filename suffix to remove for pairing, including its underscore.')
    parser.add_argument('--crop-border', type=int, default=0, help='Border removed ONLY for PSNR/SSIM; default 0.')
    parser.add_argument('--lpips-net', choices=('alex', 'vgg'), default='alex')
    parser.add_argument('--ida-definition', choices=('angle', 'cosine'), default='angle',
                        help='angle: degrees/lower better; cosine: similarity/higher better.')
    parser.add_argument('--arcface-weights', type=Path, help='Optional local FaceXLib IR-SE50 checkpoint.')
    parser.add_argument('--cache-dir', type=Path,
                        default=Path(__file__).resolve().parent / '.cache' / 'face_metrics',
                        help='Torch/PyIQA/ArcFace weight cache. Missing pretrained weights are downloaded.')
    parser.add_argument('--fid-mode', choices=('clean', 'legacy_pytorch', 'legacy_tensorflow'), default='clean')
    parser.add_argument('--fid-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0, help='FID data-loader workers; 0 is portable to notebooks.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    args.metrics = list(dict.fromkeys(args.metrics or (PAIRED_METRICS if args.mode == 'paired' else UNPAIRED_METRICS)))
    allowed = set(PAIRED_METRICS if args.mode == 'paired' else UNPAIRED_METRICS)
    if not set(args.metrics) <= allowed:
        parser.error(f'{args.mode} mode supports only: {", ".join(sorted(allowed))}')
    if args.mode == 'paired' and args.gt_dir is None:
        parser.error('--gt-dir is required in paired mode.')
    if args.mode == 'unpaired' and args.gt_dir is not None:
        parser.error('Use --fid-ref-dir, not --gt-dir, for an unpaired reference distribution.')
    if args.mode == 'unpaired' and 'fid' in args.metrics and args.fid_ref_dir is None:
        parser.error('Unpaired FID still requires --fid-ref-dir with normal-light real images.')
    if args.crop_border < 0 or args.fid_batch_size < 1 or args.num_workers < 0:
        parser.error('crop-border/num-workers must be nonnegative; fid-batch-size must be positive.')
    return args


def collect_images(root, recursive=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {root}')
    paths = sorted(path for path in (root.rglob('*') if recursive else root.iterdir())
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found in {root}')
    return root, paths


def image_index(root, paths, match_by='relative-stem', suffix=''):
    index = {}
    for path in paths:
        relative = path.relative_to(root)
        stem = relative.stem
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        key = stem if match_by == 'stem' else str(relative.parent / stem)
        if key in index:
            raise ValueError(f'Duplicate pairing key {key!r}: {index[key]} and {path}')
        index[key] = path
    return index


def match_pairs(pred_root, predictions, gt_root, references, match_by='relative-stem', suffix=''):
    pred_index = image_index(pred_root, predictions, match_by, suffix)
    gt_index = image_index(gt_root, references, match_by)
    missing_gt = sorted(set(pred_index) - set(gt_index))
    missing_pred = sorted(set(gt_index) - set(pred_index))
    if missing_gt or missing_pred:
        raise ValueError(
            'Prediction/GT sets must match exactly; no images are silently skipped. '
            f'Missing GT: {len(missing_gt)} {missing_gt[:5]}; '
            f'missing predictions: {len(missing_pred)} {missing_pred[:5]}. '
            'Check the selected directories, --recursive, --match-by, and --pred-suffix.')
    return [(key, pred_index[key], gt_index[key]) for key in sorted(pred_index)]


def image_size(path):
    try:
        with Image.open(path) as image:
            if image.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                raise ValueError(f'Expected 8-bit images; found mode {image.mode}')
            return image.size
    except Exception as exc:
        raise ValueError(f'Cannot read image header: {path}: {exc}') from exc


def read_rgb(path):
    try:
        with Image.open(path) as image:
            return np.array(image.convert('RGB'), dtype=np.uint8, copy=True)
    except Exception as exc:
        raise ValueError(f'Cannot decode image: {path}: {exc}') from exc


def fidelity_scores(prediction, reference, metrics, crop_border=0):
    if prediction.shape != reference.shape:
        raise ValueError(f'Image sizes differ: {prediction.shape} versus {reference.shape}; images are not resized.')
    pred = prediction.astype(np.float64) / 255.0
    ref = reference.astype(np.float64) / 255.0
    if crop_border:
        if min(pred.shape[:2]) <= 2 * crop_border:
            raise ValueError('crop-border removes the entire image.')
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        ref = ref[crop_border:-crop_border, crop_border:-crop_border]
    scores = {}
    if 'psnr' in metrics:
        mse = float(np.mean(np.square(pred - ref)))
        scores['PSNR'] = float('inf') if mse == 0 else -10.0 * math.log10(mse)
    if 'ssim' in metrics:
        if min(pred.shape[:2]) < 11:
            raise ValueError('SSIM requires at least 11x11 pixels after cropping.')
        from skimage.metrics import structural_similarity
        scores['SSIM'] = float(structural_similarity(
            pred, ref, data_range=1.0, channel_axis=2, gaussian_weights=True,
            sigma=1.5, win_size=11, use_sample_covariance=False))
    return scores


def identity_scores(features, references, definition='angle'):
    """Return ArcFace angle in degrees and cosine; feature norm is irrelevant."""
    pred = np.asarray(features, dtype=np.float64)
    ref = np.asarray(references, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != ref.shape:
        raise ValueError('ArcFace embeddings must have matching [N, D] shapes.')
    norms = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
    if not np.isfinite(pred).all() or not np.isfinite(ref).all() or np.any(norms <= 0):
        raise ValueError('ArcFace produced invalid or zero embeddings.')
    cosine = np.clip(np.sum(pred * ref, axis=1) / norms, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return (angle if definition == 'angle' else cosine), cosine


class ArcFaceIDA:
    def __init__(self, device, cache_dir, weights_path=None, definition='angle'):
        import torch
        try:
            from facexlib.recognition.arcface_arch import Backbone
        except ImportError as exc:
            raise ImportError('IDA needs facexlib: install facexlib in the evaluation environment.') from exc
        self.torch = torch
        self.definition = definition
        # The FaceXLib convenience factory initializes CUDA even on CPU; use its
        # published architecture directly so that CPU evaluation also works.
        self.model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        if weights_path is None:
            state = torch.hub.load_state_dict_from_url(
                ARC_FACE_URL, model_dir=str(cache_dir / 'arcface'), map_location='cpu', progress=True)
            self.weights_path = cache_dir / 'arcface' / Path(ARC_FACE_URL).name
        else:
            self.weights_path = Path(weights_path).expanduser().resolve()
            state = torch.load(self.weights_path, map_location='cpu', weights_only=True)
        for key in ('state_dict', 'params'):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().requires_grad_(False).to(device)

    def __call__(self, prediction, reference):
        torch = self.torch
        faces = torch.cat([prediction, reference], dim=0)
        faces = torch.nn.functional.interpolate(faces, size=(112, 112), mode='bilinear', align_corners=False)
        embeddings = self.model(faces * 2.0 - 1.0).detach().cpu().numpy()
        angle, cosine = identity_scores(embeddings[:1], embeddings[1:], self.definition)
        return {'IDA': float(angle[0]), 'IDA_cosine': float(cosine[0])}


def score_scalar(value, name):
    import torch
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f'{name} must return one score per image; got {tuple(value.shape)}')
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} returned a non-finite score: {value}')
    return value


def load_learned_metrics(args, device):
    learned = set(args.metrics) & {'lpips', 'musiq', 'fid'}
    if not learned:
        return {}
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError('LPIPS/FID/MUSIQ need pyiqa: install pyiqa in the evaluation environment.') from exc
    models = {}
    if 'lpips' in learned:
        models['lpips'] = pyiqa.create_metric('lpips', device=device, net=args.lpips_net, version='0.1', as_loss=False)
    if 'musiq' in learned:
        models['musiq'] = pyiqa.create_metric('musiq', device=device, pretrained='koniq10k', as_loss=False)
    # Load Inception separately after per-image metrics to reduce peak memory.
    return models


def make_fid_view(paths, directory):
    """Expose exactly the chosen images, independent of library recursion rules."""
    directory.mkdir(parents=True)
    for number, source in enumerate(paths):
        destination = directory / f'{number:08d}{source.suffix.lower()}'
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copyfile(source, destination)


def calculate_fid(predictions, references, args, device):
    if min(len(predictions), len(references)) < 2:
        raise ValueError('FID needs at least two images in each population.')
    import pyiqa
    metric = pyiqa.create_metric('fid', device=device, dims=2048, as_loss=False)
    with tempfile.TemporaryDirectory(prefix='face-metrics-fid-') as temporary:
        root = Path(temporary)
        make_fid_view(predictions, root / 'predictions')
        make_fid_view(references, root / 'references')
        from pyiqa.utils.img_util import scandir_images
        if len(scandir_images(str(root / 'predictions'))) != len(predictions) or len(scandir_images(str(root / 'references'))) != len(references):
            raise ValueError('PyIQA does not recognize every selected FID image. Convert unsupported formats to PNG.')
        value = score_scalar(metric(
            str(root / 'predictions'), str(root / 'references'), mode=args.fid_mode,
            batch_size=args.fid_batch_size, num_workers=args.num_workers, verbose=True), 'FID')
    if value < -1e-3:
        raise ValueError(f'FID is negative ({value}); covariance calculation is numerically unstable.')
    return max(0.0, value)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'count': int(len(values)), 'mean': float(np.mean(values)),
        'std': float(np.std(values)) if np.isfinite(values).all() else None,
        'min': float(np.min(values)), 'max': float(np.max(values)),
        'positive_infinity_count': int(np.isposinf(values).sum()),
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN cannot be written into an evaluation report.')
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_reports(directory, rows, summary):
    directory.mkdir(parents=True, exist_ok=True)
    columns = ['image', 'prediction', 'reference'] + [
        name for name in ('PSNR', 'SSIM', 'LPIPS', 'IDA', 'IDA_cosine', 'MUSIQ') if any(name in row for row in rows)]
    csv_tmp = directory / 'per_image.csv.tmp'
    json_tmp = directory / 'summary.json.tmp'
    with csv_tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    json_tmp.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(csv_tmp, directory / 'per_image.csv')
    os.replace(json_tmp, directory / 'summary.json')


def evaluate(args):
    pred_root, predictions = collect_images(args.pred_dir, args.recursive)
    gt_root = None
    if args.mode == 'paired':
        gt_root, ground_truths = collect_images(args.gt_dir, args.recursive)
        entries = match_pairs(pred_root, predictions, gt_root, ground_truths, args.match_by, args.pred_suffix)
    else:
        index = image_index(pred_root, predictions, args.match_by)
        entries = [(key, index[key], None) for key in sorted(index)]

    fid_root, fid_references = None, []
    if 'fid' in args.metrics:
        fid_root, fid_references = collect_images(args.fid_ref_dir or gt_root, args.recursive)
        if min(len(predictions), len(fid_references)) < 2:
            raise ValueError('FID needs at least two images in each population.')
    # Validate pairing and headers before downloading any metric weights.
    for key, prediction, reference in entries:
        size = image_size(prediction)
        if reference is not None and size != image_size(reference):
            raise ValueError(f'{key}: prediction/GT dimensions differ. No automatic resizing is allowed.')
        if 'ssim' in args.metrics and min(size) - 2 * args.crop_border < 11:
            raise ValueError(f'{key}: too small for SSIM after crop-border.')
    for path in fid_references:
        image_size(path)

    import torch
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu or fix the GPU environment.')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set this before importing PyIQA, which derives its cache at import time.
    torch.hub.set_dir(str(cache_dir))
    models = load_learned_metrics(args, device)
    identity = ArcFaceIDA(device, cache_dir, args.arcface_weights, args.ida_definition) if 'ida' in args.metrics else None
    identity_weights = str(identity.weights_path) if identity is not None else None

    print(f'{args.mode}: {len(entries)} predictions; device={device}; metrics={", ".join(args.metrics)}', flush=True)
    rows = []
    with torch.inference_mode():
        for number, (key, prediction, reference) in enumerate(entries, 1):
            pred = read_rgb(prediction)
            ref = read_rgb(reference) if reference is not None else None
            scores = fidelity_scores(pred, ref, args.metrics, args.crop_border) if ref is not None else {}
            if models or identity is not None:
                pred_tensor = torch.from_numpy(pred.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
                ref_tensor = None if ref is None else torch.from_numpy(ref.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
                if 'lpips' in models:
                    scores['LPIPS'] = score_scalar(models['lpips'](pred_tensor, ref_tensor), 'LPIPS')
                if 'musiq' in models:
                    scores['MUSIQ'] = score_scalar(models['musiq'](pred_tensor), 'MUSIQ')
                if identity is not None:
                    scores.update(identity(pred_tensor, ref_tensor))
            if not all(math.isfinite(value) or (name == 'PSNR' and value == float('inf')) for name, value in scores.items()):
                raise ValueError(f'{key}: invalid metric values {scores}')
            rows.append({'image': key, 'prediction': str(prediction), 'reference': str(reference) if reference else '', **scores})
            if number == 1 or number % 50 == 0 or number == len(entries):
                print(f'[{number}/{len(entries)}] {key}', flush=True)

    # All per-image scores are materialized; release their models before FID.
    del models, identity
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    statistics = {name: summarize([row[name] for row in rows]) for name in rows[0] if name not in ('image', 'prediction', 'reference')}
    scores = {name: stats['mean'] for name, stats in statistics.items()}
    if 'fid' in args.metrics:
        scores['FID'] = calculate_fid(predictions, fid_references, args, device)

    versions = {'python': platform.python_version()}
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-image', 'pyiqa', 'facexlib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    summary = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'pred_dir': str(pred_root), 'gt_dir': str(gt_root) if gt_root else None,
        'fid_ref_dir': str(fid_root) if fid_root else None,
        'num_predictions': len(predictions), 'num_pairs': len(entries) if gt_root else 0,
        'num_fid_reference_images': len(fid_references), 'device': str(device),
        'requested_metrics': args.metrics, 'scores': scores, 'per_image_statistics': statistics,
        'protocol': {
            'input': '8-bit decoded RGB, [0,1], no geometric alignment or fidelity resize',
            'pairing': args.match_by, 'prediction_suffix_removed': args.pred_suffix,
            'PSNR': 'RGB, data_range=1, arithmetic mean of per-image dB scores; exact matches are Infinity',
            'SSIM': 'RGB channel mean; Gaussian 11x11, sigma=1.5, population covariance, data_range=1',
            'psnr_ssim_crop_border': args.crop_border,
            'LPIPS': f'PyIQA LPIPS v0.1, {args.lpips_net}',
            'IDA': 'ArcFace IR-SE50; pre-aligned crops resized bilinearly to 112x112 and normalized to [-1,1]',
            'ida_definition': args.ida_definition,
            'ida_unit': 'degrees' if args.ida_definition == 'angle' else 'cosine similarity',
            'arcface_weights': identity_weights,
            'arcface_weights_url': ARC_FACE_URL if identity_weights is not None and args.arcface_weights is None else None,
            'MUSIQ': 'PyIQA MUSIQ, KonIQ-10k pretrained model; mean over images',
            'FID': f'PyIQA InceptionV3 2048 features; {args.fid_mode}; one score for two image populations',
            'fid_batch_size': args.fid_batch_size, 'seed': args.seed,
            'lower_is_better': ['FID', 'LPIPS'] + (['IDA'] if args.ida_definition == 'angle' else []),
            'higher_is_better': ['PSNR', 'SSIM', 'MUSIQ', 'IDA_cosine'] + (['IDA'] if args.ida_definition == 'cosine' else []),
        },
        'versions': versions, 'cache_dir': str(cache_dir),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prediction_images': [str(path.relative_to(pred_root)) for path in predictions],
        'fid_reference_images': [str(path.relative_to(fid_root)) for path in fid_references],
    }
    write_reports(args.output_dir.expanduser().resolve(), rows, summary)
    print(json.dumps(json_safe(scores), ensure_ascii=False, indent=2, allow_nan=False))
    print(f'Reports: {args.output_dir.resolve() / "summary.json"} and per_image.csv', flush=True)
    return summary


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == '__main__':
    main()
