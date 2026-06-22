import argparse
import glob
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
from torchvision.transforms.functional import normalize

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from basicsr.archs import build_network
from basicsr.utils import img2tensor, imwrite, tensor2img
from basicsr.utils.misc import get_device
from basicsr.utils.options import ordered_yaml
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to parse --opt network configs.") from exc


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")


def parse_args():
    parser = argparse.ArgumentParser(description="DMDP-FR inference, following CodeFormer quick inference.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./inputs/whole_imgs",
        help="Input image, video, or folder. Default: inputs/whole_imgs")
    parser.add_argument(
        "--recursive_input", action="store_true",
        help="Recursively collect images from subfolders. Useful for LFW/WIDER-style datasets.")
    parser.add_argument(
        "-o", "--output_path", type=str, default=None,
        help="Output folder. Default: results/<input_name>_<w>")
    parser.add_argument(
        "-w", "--fidelity_weight", type=float, default=0.5,
        help="Balance the quality and fidelity. Default: 0.5")
    parser.add_argument(
        "-s", "--upscale", type=int, default=2,
        help="The final upsampling scale for whole-image restoration. Default: 2")
    parser.add_argument(
        "--has_aligned", action="store_true",
        help="Input images are cropped and aligned faces. Default: False")
    parser.add_argument(
        "--only_center_face", action="store_true",
        help="Only restore the center face for whole-image input. Default: False")
    parser.add_argument(
        "--draw_box", action="store_true",
        help="Draw face boxes on whole-image output. Default: False")
    parser.add_argument(
        "--detection_model", type=str, default="retinaface_resnet50",
        help="Face detector: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, or dlib.")
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Suffix of restored faces/results. Default: None")
    parser.add_argument(
        "--save_video_fps", type=float, default=None,
        help="Frame rate for saving video. Default: input video fps")
    parser.add_argument(
        "--measure_speed", action="store_true",
        help="Measure CUDA inference speed for no-GT image datasets. Reports GPU-synchronized s/image.")
    parser.add_argument(
        "--speed_warmup", type=int, default=5,
        help="Number of valid images to process before speed measurement. Default: 5")
    parser.add_argument(
        "--speed_log_interval", type=int, default=50,
        help="Print running average every N measured images. Set <=0 to disable. Default: 50")
    parser.add_argument(
        "--speed_no_save", action="store_true",
        help="Skip saving images/videos during speed measurement.")

    parser.add_argument(
        "--opt", type=str, default="options/DMDP-FR_stage3_triple.yml",
        help="DMDP-FR option file. Its network_g section is used to build the model.")
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="DMDP-FR net_g checkpoint, for example experiments/.../models/net_g_latest.pth.")
    parser.add_argument(
        "--param_key", type=str, default="params_ema",
        help="Checkpoint parameter key. Use params_ema for final EMA weights when available.")
    parser.add_argument(
        "--non_strict_load", action="store_true",
        help="Load only checkpoint keys that match the current model shape.")
    parser.add_argument(
        "--use_stage1_init", action="store_true",
        help="Keep network_g stage-1 init paths from the option file before loading --ckpt_path.")
    parser.add_argument(
        "--img_size", type=int, default=None,
        help="Aligned face size. Defaults to network_g.img_size from --opt.")
    parser.add_argument(
        "--adain", dest="adain", action="store_true",
        help="Enable adain=True in DMDP-FR forward. This is the default.")
    parser.add_argument(
        "--no_adain", dest="adain", action="store_false",
        help="Disable adain in DMDP-FR forward.")
    parser.set_defaults(adain=True)
    return parser.parse_args()


def load_yaml(opt_path):
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        return yaml.load(f, Loader=Loader)


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
        "Set mean/std to [0.5]/[0.5] or [0]/[1], or update this script explicitly."
    )


def get_cuda_device_name(device):
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_name(device_index)


def collect_inputs(input_path, recursive_input=False):
    input_video = False
    video_name = None
    audio = None
    fps = None

    if input_path.lower().endswith(IMG_EXTENSIONS):
        input_list = [input_path]
        result_root = f"results/test_img"
    elif input_path.endswith(VIDEO_EXTENSIONS):
        from basicsr.utils.video_util import VideoReader

        input_list = []
        vidreader = VideoReader(input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps()
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        result_root = f"results/{video_name}"
        input_video = True
        vidreader.close()
    else:
        input_path = input_path.rstrip("/")
        input_list = []
        for ext in IMG_EXTENSIONS:
            if recursive_input:
                input_list.extend(glob.glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
                input_list.extend(glob.glob(os.path.join(input_path, "**", f"*{ext.upper()}"), recursive=True))
            else:
                input_list.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
                input_list.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        input_list = sorted(set(input_list))
        result_root = f"results/{os.path.basename(input_path)}"

    return input_list, result_root, input_video, video_name, audio, fps


def build_dmdp_fr(opt, args, device):
    net_opt = deepcopy(opt["network_g"])
    if not args.use_stage1_init:
        net_opt["stage1_model_path"] = None
        net_opt["vqgan_path"] = None
        net_opt["model_path"] = None

    net = build_network(net_opt).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and args.param_key in checkpoint:
        state_dict = checkpoint[args.param_key]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
        print(f"Checkpoint key {args.param_key!r} not found. Falling back to 'params'.")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("model."):
            key = key[6:]
        cleaned_state[key] = value

    if args.non_strict_load:
        current_state = net.state_dict()
        filtered_state = {
            key: value for key, value in cleaned_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        skipped = sorted(set(cleaned_state.keys()) - set(filtered_state.keys()))
        if skipped:
            print(f"Skipped {len(skipped)} unmatched checkpoint keys.")
        missing, unexpected = net.load_state_dict(filtered_state, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    else:
        net.load_state_dict(cleaned_state, strict=True)

    net.eval()
    return net


def restore_face(net, cropped_face, mean, std, tensor_min_max, w, adain, device, empty_cache=True):
    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
    normalize(cropped_face_t, mean, std, inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            output = net(cropped_face_t, w=w, adain=adain)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=tensor_min_max)
        del output
        if empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as error:
        print(f"\tFailed inference for DMDP-FR: {error}")
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=tensor_min_max)
    return restored_face.astype("uint8")


def main():
    args = parse_args()
    device = get_device()
    if args.measure_speed and device.type != "cuda":
        raise RuntimeError("--measure_speed only supports CUDA GPU inference, e.g. L40 or A800.")
    if args.measure_speed:
        args.speed_warmup = max(args.speed_warmup, 0)
        if args.input_path.lower().endswith(tuple(ext.lower() for ext in VIDEO_EXTENSIONS)):
            raise ValueError("--measure_speed is intended for image/folder no-GT datasets, not video input.")

    opt = load_yaml(args.opt)
    mean, std = get_mean_std(opt)
    tensor_min_max = infer_tensor_min_max(mean, std)
    img_size = args.img_size or int(opt["network_g"].get("img_size", 512))
    w = args.fidelity_weight

    recursive_input = args.recursive_input or args.measure_speed
    input_img_list, result_root, input_video, video_name, audio, fps = collect_inputs(
        args.input_path, recursive_input=recursive_input)
    result_root = f"{result_root}_{w}"
    if args.output_path is not None:
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError(
            "No input image/video is found. For video input, --input_path should end with .mp4|.mov|.avi."
        )

    bg_upsampler = None
    face_upsampler = None

    net = build_dmdp_fr(opt, args, device)

    if not args.has_aligned:
        print(f"Face detection model: {args.detection_model}")
    print("Background upsampling: False, Face upsampling: False")
    print(f"DMDP-FR settings: w={w}, adain={args.adain}, img_size={img_size}")
    if args.measure_speed:
        print(
            "Speed benchmark: CUDA GPU={}, warmup={} images, unit=s/image, save_skipped={}".format(
                get_cuda_device_name(device), args.speed_warmup, args.speed_no_save
            )
        )

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=img_size,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    processed_img_count = 0
    speed_count = 0
    speed_total_sec = 0.0
    speed_total_faces = 0

    for i, img_path in enumerate(input_img_list):
        face_helper.clean_all()
        restored_img = None

        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename = os.path.splitext(img_name)[0]
            print(f"[{i + 1}/{test_img_num}] Processing: {img_name}")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"\tSkip unreadable image: {img_path}")
                continue
        else:
            frame_name = str(i).zfill(6)
            basename = f"{video_name}_{frame_name}" if input_video else frame_name
            print(f"[{i + 1}/{test_img_num}] Processing: {basename}")
            img = img_path

        processed_img_count += 1
        measure_this_image = args.measure_speed and processed_img_count > args.speed_warmup
        if measure_this_image:
            torch.cuda.synchronize()
            speed_start_time = time.perf_counter()

        if args.has_aligned:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print("Grayscale input: True")
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            print(f"\tdetect {num_det_faces} faces")
            face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            restored_face = restore_face(
                net,
                cropped_face,
                mean,
                std,
                tensor_min_max,
                w,
                args.adain,
                device,
                empty_cache=not args.measure_speed,
            )
            face_helper.add_restored_face(restored_face, cropped_face)

        if not args.has_aligned:
            if bg_upsampler is not None:
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        if measure_this_image:
            torch.cuda.synchronize()
            elapsed_sec = time.perf_counter() - speed_start_time
            speed_count += 1
            speed_total_sec += elapsed_sec
            speed_total_faces += len(face_helper.cropped_faces)
            if args.speed_log_interval > 0 and speed_count % args.speed_log_interval == 0:
                print(f"\tSpeed [{speed_count} images]: {speed_total_sec / speed_count:.4f} s/image")

        if args.speed_no_save:
            continue

        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            if not args.has_aligned:
                save_crop_path = os.path.join(result_root, "cropped_faces", f"{basename}_{idx:02d}.png")
                imwrite(cropped_face, save_crop_path)

            if args.has_aligned:
                save_face_name = f"{basename}.png"
            else:
                save_face_name = f"{basename}_{idx:02d}.png"
            if args.suffix is not None:
                save_face_name = f"{save_face_name[:-4]}_{args.suffix}.png"
            save_restore_path = os.path.join(result_root, "restored_faces", save_face_name)
            imwrite(restored_face, save_restore_path)

        if not args.has_aligned and restored_img is not None:
            save_basename = f"{basename}_{args.suffix}" if args.suffix is not None else basename
            save_restore_path = os.path.join(result_root, "final_results", f"{save_basename}.png")
            imwrite(restored_img, save_restore_path)

    if args.measure_speed:
        measured_warmup = min(args.speed_warmup, processed_img_count)
        if speed_count == 0:
            print(
                "\nSpeed benchmark: no measured images. "
                f"Valid images={processed_img_count}, warmup={measured_warmup}. "
                "Reduce --speed_warmup or provide more input images."
            )
        else:
            print("\nSpeed benchmark summary")
            print(f"\tGPU: {get_cuda_device_name(device)}")
            print(f"\tWarmup images: {measured_warmup}")
            print(f"\tMeasured images: {speed_count}")
            print(f"\tRestored faces: {speed_total_faces} ({speed_total_faces / speed_count:.2f} faces/image)")
            print(f"\tAverage speed: {speed_total_sec / speed_count:.4f} s/image")

    if input_video and not args.speed_no_save:
        from basicsr.utils.video_util import VideoWriter

        print("Video Saving...")
        video_frames = []
        img_list = sorted(glob.glob(os.path.join(result_root, "final_results", "*.[jp][pn]g")))
        for img_path in img_list:
            img = cv2.imread(img_path)
            if img is not None:
                video_frames.append(img)
        if video_frames:
            height, width = video_frames[0].shape[:2]
            save_video_name = f"{video_name}_{args.suffix}" if args.suffix is not None else video_name
            save_restore_path = os.path.join(result_root, f"{save_video_name}.mp4")
            vidwriter = VideoWriter(
                save_restore_path,
                height,
                width,
                fps if args.save_video_fps is None else args.save_video_fps,
                audio,
            )
            for frame in video_frames:
                vidwriter.write_frame(frame)
            vidwriter.close()

    if args.speed_no_save:
        print("\nResult saving skipped by --speed_no_save")
    else:
        print(f"\nAll results are saved in {result_root}")


if __name__ == "__main__":
    main()
