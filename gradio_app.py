#!/usr/bin/env python3
"""DMDP-FR whole-image low-light face restoration.

python gradio_app.py
python gradio_app.py --device cpu --input ../WildDark-Face/1.png
python gradio_app.py --ckpt experiments/.../models/net_g_latest.pth

RetinaFace detection -> 256px alignment -> restoration -> parsing-based blending.
FaceXlib provides the CodeFormer-style alignment and blending pipeline.
Gamma/CLAHE improves detection only; restoration uses the original dark crops.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOGGER = logging.getLogger('dmdp_fr_demo')
DEFAULT_OPT = ROOT / 'options/DMDP-FR_darkface_stage3_triple.yml'
DEFAULT_CKPT = ROOT / 'experiments/DMDP-FR_darkface_stage3_triple/models/net_g_latest.pth'
DETECTION_MODES = ['Original + enhanced (recommended)', 'Original only', 'Enhanced only']
APP_CSS = """
body, button, input, textarea, select,
.gradio-container, .gradio-container * {
    font-family: "Times New Roman", Times, serif !important;
}
.gradio-container {
    --font: "Times New Roman", Times, serif;
    --font-mono: "Times New Roman", Times, serif;
}
#face-results-column {
    align-self: stretch;
}
#face-results-gallery {
    flex: 1 1 0%;
    min-height: 520px;
}
#face-results-gallery .gallery-container {
    position: absolute;
    inset: 0;
}
#face-results-gallery .grid-wrap {
    height: 100% !important;
    box-sizing: border-box;
}
#face-results-gallery .grid-container {
    grid-template-rows: none !important;
    grid-auto-rows: max-content !important;
    align-content: start;
}
#face-results-gallery .gallery-item {
    aspect-ratio: 2 / 1;
    width: 100%;
    max-width: 400px;
    justify-self: center;
    height: auto !important;
    min-height: 0;
}
#face-results-gallery .thumbnail-lg {
    aspect-ratio: 2 / 1 !important;
    width: 100%;
    height: 100%;
}
#face-results-gallery .preview {
    max-height: none;
}
"""


def enhance_for_detection(bgr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Brighten with gamma < 1 for detection, never for restoration input."""
    if not 0 < gamma <= 1:
        raise ValueError('Detection gamma must be in (0, 1].')
    table = np.rint((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    bright = cv2.LUT(bgr, table)
    lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_faces(detector, bgr, mode=DETECTION_MODES[0], gamma=0.5,
                 confidence=0.8, max_side=1280, min_eye_distance=4.0,
                 only_largest=False, max_faces=32):
    """Merge detections from both views in original coordinates using NMS.

    Each RetinaFace row contains x1,y1,x2,y2,score and five landmark pairs.
    Scale coordinates only, leaving confidence scores unchanged.
    """
    from facexlib.detection.retinaface_utils import py_cpu_nms

    if mode not in DETECTION_MODES:
        raise ValueError(f'Unknown detection mode: {mode}')
    if not 0 < confidence < 1 or max_side < 32 or max_faces < 1:
        raise ValueError('Invalid detection threshold, image size, or maximum face count.')
    height, width = bgr.shape[:2]
    scale = min(1.0, int(max_side) / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    views = []
    if mode != DETECTION_MODES[2]:
        views.append(small)
    if mode != DETECTION_MODES[1]:
        views.append(enhance_for_detection(small, gamma))
    candidates = []
    with torch.inference_mode():
        for view in views:
            rows = detector.detect_faces(view, conf_threshold=float(confidence),
                                         nms_threshold=0.4, use_origin_size=True)
            if rows is None or len(rows) == 0:
                continue
            rows = np.asarray(rows, dtype=np.float32).copy()
            # Use actual axis scales to account for integer resize rounding.
            rows[:, [0, 2, 5, 7, 9, 11, 13]] *= width / size[0]
            rows[:, [1, 3, 6, 8, 10, 12, 14]] *= height / size[1]
            eyes = np.linalg.norm(rows[:, 5:7] - rows[:, 7:9], axis=1)
            valid = (np.isfinite(rows).all(axis=1) & (eyes >= min_eye_distance)
                     & (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1]))
            candidates.extend(rows[valid])
    if not candidates:
        return np.empty((0, 15), dtype=np.float32)
    rows = np.asarray(candidates)
    rows = rows[np.argsort(-rows[:, 4])]
    rows = rows[py_cpu_nms(rows[:, :5], 0.4)]
    if only_largest:
        clipped = rows[:, :4].copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
        areas = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        rows = rows[[np.argmax(areas)]]
    elif len(rows) > max_faces:
        LOGGER.warning('Detected %d faces; processing the %d highest-confidence faces.', len(rows), max_faces)
        rows = rows[:max_faces]
    return rows


class DMDPRestorer:
    """Strictly load a full Stage III checkpoint without separate Stage I/II weights."""

    def __init__(self, opt_path, checkpoint_path, device):
        from basicsr.archs import build_network

        opt_path, checkpoint_path = Path(opt_path), Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'DMDP-FR checkpoint not found: {checkpoint_path}. Specify it with --ckpt.')
        with opt_path.open(encoding='utf-8') as handle:
            self.options = yaml.safe_load(handle)
        network = deepcopy(self.options['network_g'])
        if network.get('type') != 'DMDPFR' or network.get('img_size') != 256:
            raise ValueError('This demo requires matching weights with network_g.type=DMDPFR and img_size=256.')
        for key in ('stage1_model_path', 'vqgan_path', 'model_path'):
            network.pop(key, None)
        dataset = (self.options.get('datasets', {}).get('val')
                   or self.options.get('datasets', {}).get('train', {}))
        if list(dataset.get('mean', [0.5] * 3)) != [0.5] * 3 or list(
                dataset.get('std', [0.5] * 3)) != [0.5] * 3:
            raise ValueError('Inference requires mean and std to both be [0.5, 0.5, 0.5].')
        self.device = torch.device(device)
        self.net = build_network(network)
        # mmap reduces peak memory; weights_only avoids loading arbitrary objects.
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        except RuntimeError as exc:
            if 'mmap' not in str(exc):
                raise
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state = checkpoint
        self.param_key = 'raw_state_dict'
        for name in ('params_ema', 'params', 'state_dict'):
            if name in checkpoint:
                state, self.param_key = checkpoint[name], name
                break
        # Accept DataParallel exports while strictly checking all parameters.
        state = {name.removeprefix('module.'): value for name, value in state.items()}
        self.net.load_state_dict(state, strict=True)
        del state, checkpoint
        self.net.eval().requires_grad_(False).to(self.device)
        LOGGER.info('DMDP-FR loaded: %s (%s)', checkpoint_path, self.param_key)

    @torch.inference_mode()
    def restore(self, bgr, fidelity=1.0):
        if bgr.shape != (256, 256, 3):
            raise ValueError(f'DMDP-FR requires a 256x256 BGR input; received {bgr.shape}.')
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(rgb).float().div_(255).sub_(0.5).div_(0.5)
        result = self.net(tensor.unsqueeze(0).to(self.device), w=float(fidelity), adain=False)[0]
        if result.shape != (1, 3, 256, 256) or not torch.isfinite(result).all():
            raise RuntimeError('DMDP-FR returned an invalid image tensor.')
        rgb = result[0].clamp(-1, 1).add(1).mul(127.5).round().byte()
        return np.ascontiguousarray(rgb.permute(1, 2, 0).cpu().numpy()[:, :, ::-1])


def save_bgr(path, image):
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


class WholeImagePipeline:
    def __init__(self, opt=DEFAULT_OPT, ckpt=DEFAULT_CKPT, device=None,
                 model_dir=ROOT / 'weights/facelib', output_dir=ROOT / 'results/gradio'):
        self.opt, self.ckpt = Path(opt).resolve(), Path(ckpt).resolve()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_dir, self.output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
        self.helper = self.restorer = None
        self.lock = threading.RLock()

    def load(self):
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        with self.lock:
            if self.helper is None:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info('Loading RetinaFace-ResNet50 / ParseNet; missing weights are downloaded from FaceXlib releases.')
                self.helper = FaceRestoreHelper(1, face_size=256, crop_ratio=(1, 1),
                                               det_model='retinaface_resnet50', use_parse=True,
                                               device=self.device, model_rootpath=str(self.model_dir))
            if self.restorer is None:
                self.restorer = DMDPRestorer(self.opt, self.ckpt, self.device)

    def run(self, image, mode=DETECTION_MODES[0], gamma=0.5, confidence=0.8,
            fidelity=1.0, only_largest=False, max_side=1280):
        if image is None:
            raise ValueError('Please upload an image first.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.width * image.height > 16_000_000:
            raise ValueError('Please resize the image to at most 16 million pixels.')
        if min(image.size) < 16:
            raise ValueError('Image width and height must each be at least 16 pixels.')
        if not 0 <= fidelity <= 1:
            raise ValueError('Fusion weight must be in [0, 1].')
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        start = time.perf_counter()
        # Serialize requests: FaceRestoreHelper stores mutable intermediate state.
        with self.lock:
            self.load()
            helper = self.helper
            helper.clean_all()
            try:
                helper.read_image(bgr)
                rows = detect_faces(helper.face_det, bgr, mode, gamma, confidence,
                                    int(max_side), only_largest=only_largest)
                for row in rows:
                    helper.det_faces.append(row[:5].copy())
                    helper.all_landmarks_5.append(row[5:].reshape(5, 2).copy())
                helper.align_warp_face(border_mode='reflect101')
                # Restore sequentially to limit memory use for multi-face images.
                for crop in helper.cropped_faces:
                    helper.add_restored_face(self.restorer.restore(crop, fidelity))
                if len(rows):
                    helper.get_inverse_affine(None)
                    result = helper.paste_faces_to_input_image()
                else:
                    result = bgr.copy()
                detected = bgr.copy()
                for i, row in enumerate(rows, 1):
                    x1, y1, x2, y2 = np.rint(row[:4]).astype(int)
                    cv2.rectangle(detected, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(detected, f'{i}: {row[4]:.2f}', (max(0, x1), max(15, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    for point in row[5:].reshape(5, 2):
                        cv2.circle(detected, tuple(np.rint(point).astype(int)), 2, (0, 255, 0), -1)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                folder = Path(tempfile.mkdtemp(prefix='restoration_', dir=self.output_dir))
                save_bgr(folder / 'input.png', bgr)
                save_bgr(folder / 'detections.png', detected)
                save_bgr(folder / 'restored.png', result)
                save_bgr(folder / 'comparison.png', np.concatenate((bgr, result), axis=1))
                gallery, face_records = [], []
                for i, (crop, restored, row) in enumerate(zip(
                        helper.cropped_faces, helper.restored_faces, rows), 1):
                    save_bgr(folder / f'face_{i:02d}_input.png', crop)
                    save_bgr(folder / f'face_{i:02d}_restored.png', restored)
                    comparison = np.concatenate((crop, restored), axis=1)
                    gallery.append((cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
                                    f'Face {i}: Original / Restored'))
                    face_records.append({'index': i, 'bbox_xyxy': row[:4].tolist(),
                                         'confidence': float(row[4]),
                                         'landmarks_5': row[5:].reshape(5, 2).tolist(),
                                         'affine': helper.affine_matrices[i - 1].tolist(),
                                         'inverse_affine': helper.inverse_affine_matrices[i - 1].tolist()})
                elapsed = time.perf_counter() - start
                metadata = {'detector': 'retinaface_resnet50 (WIDER FACE pretrained)',
                            'opt': str(self.opt), 'checkpoint': str(self.ckpt),
                            'checkpoint_param_key': self.restorer.param_key,
                            'device': str(self.device), 'detection_mode': mode,
                            'detection_gamma': gamma, 'confidence_threshold': confidence,
                            'fidelity_weight': fidelity, 'detection_max_side': int(max_side),
                            'only_largest': bool(only_largest), 'max_faces': 32,
                            'face_size': 256, 'input_size': list(image.size),
                            'seconds': elapsed, 'faces': face_records}
                (folder / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                       encoding='utf-8')
                archive = folder / 'results.zip'
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                    for path in sorted(folder.iterdir()):
                        if path != archive:
                            bundle.write(path, path.name)
                status = (f'Restored {len(rows)} face(s), up to 32 per image, in {elapsed:.1f} seconds.'
                          if len(rows) else 'No faces met the detection threshold. The original image is returned. '
                          'Try lowering the threshold or adjusting detection gamma.')
                return (cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(detected, cv2.COLOR_BGR2RGB), gallery, str(archive), status)
            finally:
                helper.clean_all()


def build_demo(pipeline):
    import gradio as gr

    def infer(*args):
        try:
            # Keep the restored full image in the ZIP, without a display panel.
            result = pipeline.run(*args)
            # Each request has its own output folder, so these links only expose
            # the restored face PNGs from the current request.
            folder = Path(result[3]).parent
            face_files = [str(path) for path in sorted(folder.glob('face_*_restored.png'))]
            return (*result[1:], face_files or None)
        except Exception as exc:
            LOGGER.exception('Whole-image restoration failed')
            raise gr.Error(f'Restoration failed: {exc}') from exc

    # Gradio 6 accepts CSS at launch; Gradio 4/5 accept it on Blocks.
    style = {'css': APP_CSS} if int(gr.__version__.split('.')[0]) < 6 else {}
    with gr.Blocks(title='DMDP-FR Low-Light Face Restoration', analytics_enabled=False, **style) as demo:
        gr.Markdown('# DMDP-FR Low-Light Face Restoration')
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=280):
                source = gr.Image(type='pil', image_mode='RGB', label='Low-light input image', height=280)
                with gr.Accordion('Restoration settings', open=False):
                    mode = gr.Radio(DETECTION_MODES, value=DETECTION_MODES[0], label='Detection mode')
                    gamma = gr.Slider(0.2, 1.0, value=0.5, step=0.05,
                                      label='Detection gamma (lower values brighten more)')
                    confidence = gr.Slider(0.2, 0.95, value=0.8, step=0.05,
                                           label='Face detection confidence threshold')
                    fidelity = gr.Slider(0, 1, value=1, step=0.05, label='DMDP-FR fusion weight',
                                         info='Default: 1.0, matching the low-light Stage III training configuration.')
                    largest = gr.Checkbox(value=False, label='Restore only the largest face')
                    max_side = gr.Slider(512, 2048, value=1280, step=128,
                                         label='Maximum detection image side length')
                submit = gr.Button('Restore faces', variant='primary')
                detected = gr.Image(label='Face detection boxes and five landmarks',
                                    format='png', height=280)
            with gr.Column(scale=3, min_width=400, elem_id='face-results-column'):
                gallery = gr.Gallery(label='Detected faces: Original (left) / Restored (right)',
                                     columns=2, height='100%', scale=1,
                                     elem_id='face-results-gallery', object_fit='contain', format='png')
        face_downloads = gr.File(label='Download individual restored faces (PNG)',
                                 file_count='multiple', interactive=False)
        with gr.Row():
            status = gr.Textbox(label='Processing status', interactive=False)
            download = gr.File(label='Download full images, face crops, and coordinates (ZIP)')
        submit.click(infer, [source, mode, gamma, confidence, fidelity, largest, max_side],
                     [detected, gallery, download, status, face_downloads], concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--opt', type=Path, default=DEFAULT_OPT)
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'weights/facelib')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/gradio')
    parser.add_argument('--device', default=None, help='cuda, cuda:0, or cpu; selected automatically by default')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--lazy-load', action='store_true', help='Load models on the first request')
    parser.add_argument('--input', type=Path, help='Process one image without starting the web server')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Keep warnings/errors while hiding routine HTTP request logs.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be greater than 0')
    if not torch.cuda.is_available() or args.device == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    pipeline = WholeImagePipeline(args.opt, args.ckpt, args.device, args.model_dir, args.output_dir)
    if args.input:
        with Image.open(args.input) as image:
            *_, archive, status = pipeline.run(image)
        print(status)
        print(f'Download archive: {archive}')
    else:
        if not args.lazy_load:
            pipeline.load()
        import gradio as gr

        style = {'css': APP_CSS} if int(gr.__version__.split('.')[0]) >= 6 else {}
        demo = build_demo(pipeline)
        LOGGER.info('Web interface: http://%s:%d', args.host, args.port)
        demo.launch(server_name=args.host, server_port=args.port, share=False, quiet=True, **style)


if __name__ == '__main__':
    main()
