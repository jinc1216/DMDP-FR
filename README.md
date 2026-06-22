# DMDP-FR

Implementation of **DMDP-FR: Dynamic Multi-Granularity Discrete Prior Representation for Blind Face Restoration**. The repository keeps only the code path needed by DMDP-FR: dynamic DQ-VAE prior learning, coarse-to-fine code prediction, granularity-aware fusion, training configs, inference, and visualization/evaluation utilities.

[//]: # (> Note: the Python registry names use `DMDPFR` and `DMDPFRModel` because Python identifiers cannot contain `-`. User-facing files, commands, and documentation use the paper name `DMDP-FR`.)

## Framework

DMDP-FR restores blind degraded faces in three stages:

| Stage | Module | Purpose |
| --- | --- | --- |
| Stage I | Dynamic Multi-Granularity Quantization (DMGQ) | Learns a discrete HQ face prior with adaptive coarse/medium/fine token allocation. |
| Stage II | Coarse-to-Fine Code Prediction (CFCP) | Predicts HQ-aligned discrete codes from LQ inputs, recovering global structure before local details. |
| Stage III | Granularity-Aware Multi-Scale Fusion (GAMF) | Injects LQ-guided features into the fixed HQ decoder using predicted granularity distributions. |

![DMDP-FR framework](assets/figures/framework.png)

## Results From The DMDP-FR

### Synthetic CelebA-Test

| Method | FID (lower) | LPIPS (lower) | NIQE (lower) | IDA (lower) | PSNR (higher) | SSIM (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DR2 | 88.504 | 0.484 | 5.656 | 1.203 | 20.805 | 0.609 |
| BFRffusion | 77.338 | 0.478 | 4.461 | 1.185 | 20.560 | 0.536 |
| DiffBIR | 69.962 | 0.375 | 5.184 | 1.073 | 21.707 | 0.615 |
| CodeFormer | 63.621 | 0.365 | 4.516 | **1.019** | 21.451 | 0.581 |
| RestoreFormer++ | 57.972 | 0.450 | 3.958 | 1.160 | 20.146 | 0.500 |
| VSPBFR | 56.725 | 0.434 | <u>3.466</u> | 1.195 | 20.114 | 0.526 |
| GPEN | 56.145 | 0.425 | 3.913 | 1.141 | 20.545 | 0.552 |
| VQFR | 55.456 | 0.463 | **3.314** | 1.197 | 19.487 | 0.481 |
| RestoreFormer | 55.425 | 0.463 | 4.003 | 1.179 | 20.149 | 0.500 |
| AuthFace | 54.624 | 0.389 | 6.378 | 1.136 | 20.618 | 0.567 |
| DAEFR | 52.987 | 0.388 | 4.417 | <u>1.044</u> | 19.932 | 0.559 |
| InterLCM | 51.524 | 0.398 | 3.943 | 1.103 | 20.061 | 0.541 |
| DifFace | <u>51.247</u> | <u>0.347</u> | 4.631 | 1.060 | <u>22.190</u> | <u>0.633</u> |
| GFP-GAN | **46.958** | 0.453 | 4.061 | 1.268 | 19.574 | 0.522 |
| DMDP-FR (ours) | 52.096 | **0.346** | 5.003 | **1.019** | **22.216** | **0.649** |

![Synthetic visual comparison](assets/figures/synthetic_results.png)

### Real-World Benchmarks

| Method | LFW FID (lower) | LFW MUSIQ (higher) | WebPhoto FID (lower) | WebPhoto MUSIQ (higher) | WIDER FID (lower) | WIDER MUSIQ (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DR2 | 58.414 | 69.578 | 119.802 | 58.600 | 63.433 | 57.133 |
| BFRffusion | 51.731 | 69.606 | 87.577 | 62.331 | 58.586 | 61.581 |
| RestoreFormer++ | 51.870 | 72.249 | <u>77.637</u> | 71.486 | 46.363 | 71.511 |
| DiffBIR | 48.337 | 70.836 | 89.463 | 71.864 | 49.524 | 68.662 |
| RestoreFormer | 49.905 | 73.074 | 79.655 | 69.840 | 51.474 | 67.840 |
| CodeFormer | 54.407 | 71.430 | 86.456 | 74.000 | 40.012 | 69.310 |
| VSPBFR | <u>47.781</u> | 74.737 | 78.739 | 73.231 | 38.984 | 62.596 |
| GPEN | 57.582 | 73.590 | 95.207 | <u>75.576</u> | 54.007 | 65.326 |
| VQFR | 51.821 | <u>74.745</u> | 78.251 | 72.009 | 45.122 | 64.013 |
| AuthFace | **47.431** | 73.140 | 92.724 | 72.991 | 43.351 | 63.325 |
| DAEFR | 48.849 | 73.840 | **77.336** | 72.708 | <u>37.703</u> | 64.146 |
| InterLCM | 57.178 | 74.692 | 79.492 | **75.798** | 41.685 | 65.447 |
| DifFace | 48.222 | 69.848 | 83.266 | 65.170 | 37.915 | 65.121 |
| GFP-GAN | 51.499 | 73.569 | 91.539 | 72.097 | 40.468 | **72.814** |
| DMDP-FR (ours) | 50.110 | **74.810** | 88.624 | 69.133 | **35.230** | <u>71.859</u> |

![Real-world visual comparison](assets/figures/realworld_results.png)

### Ablation Summary

| Component | What It Adds |
| --- | --- |
| DMGQ | Adaptive token allocation for structurally complex facial regions. |
| CFCP | Hierarchical prediction that stabilizes global layout before fine details. |
| GAMF | Input-dependent feature fusion for perception-fidelity control. |
| Triple granularity `{16,32,64}` | Best reported WIDER-Test setting: FID 35.230 with 608 tokens and 0.0941 sec/img latency. |

![Ablation and route visualizations](assets/figures/ablation_visuals.png)

## Installation

```bash
conda create -n dmdp-fr python=3.10 -y
conda activate dmdp-fr
pip install -r requirements.txt
```

This repository is designed to run from the repository root. No package installation step is required.

## Dataset Layout

Edit the `dataroot_*` fields in `options/*.yml` for your environment. The default configs expect:

```text
datasets/
  ffhq/ffhq_512/
  faces/validation/lq/
  faces/validation/gt/
```

## Checkpoint Layout

```text
experiments/pretrained_models/
  dqvae/dqvae_stage1_triple.pth
  dmdp_fr_stage2/net_g_latest.pth
  dmdp_fr/dmdp_fr_stage3.pth
weights/
  lpips/vgg.pth
```

Stage-2 and Stage-3 use the DQ-VAE prior checkpoint through `stage1_model_path` and `network_vqgan.model_path`.

## Training

Train the triple-granularity DQ-VAE prior:

```bash
python basicsr/train.py -opt options/DMDP-FR_stage1_triple.yml
```

Optional dual prior:

```bash
python basicsr/train.py -opt options/DMDP-FR_stage1_dual.yml
```

Train Stage II coarse-to-fine code prediction:

```bash
python basicsr/train.py -opt options/DMDP-FR_stage2_triple.yml
```

Train Stage III granularity-aware fusion:

```bash
python basicsr/train.py -opt options/DMDP-FR_stage3_triple.yml
```

Distributed training example:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29434 basicsr/train.py -opt options/DMDP-FR_stage3_triple.yml --launcher pytorch
```

Optional task-specific configs:

```bash
python basicsr/train.py -opt options/DMDP-FR_colorization.yml
python basicsr/train.py -opt options/DMDP-FR_inpainting.yml
```

## Inference

Aligned cropped faces:

```bash
python inference_dmdp_fr.py \
  -i inputs/cropped_faces \
  --has_aligned \
  --opt options/DMDP-FR_stage3_triple.yml \
  --ckpt_path experiments/pretrained_models/dmdp_fr/dmdp_fr_stage3.pth \
  -w 0.5
```

Whole images with face detection and paste-back:

```bash
python inference_dmdp_fr.py \
  -i inputs/whole_imgs \
  --opt options/DMDP-FR_stage3_triple.yml \
  --ckpt_path experiments/pretrained_models/dmdp_fr/dmdp_fr_stage3.pth \
  --detection_model retinaface_resnet50 \
  -w 0.5
```

Folder benchmark without saving restored images:

```bash
python inference_dmdp_fr.py \
  -i datasets/faces/validation/lq \
  --recursive_input \
  --has_aligned \
  --measure_speed \
  --speed_no_save \
  --opt options/DMDP-FR_stage3_triple.yml \
  --ckpt_path experiments/pretrained_models/dmdp_fr/dmdp_fr_stage3.pth
```

## Visualization And Evaluation

Generate restored images, side-by-side comparisons, granularity maps, and metrics:

```bash
python scripts/visualize_eval_dmdp_fr.py \
  -i datasets/faces/validation/lq \
  --gt_path datasets/faces/validation/gt \
  --opt options/DMDP-FR_stage3_triple.yml \
  --ckpt_path experiments/pretrained_models/dmdp_fr/dmdp_fr_stage3.pth \
  --save_comparison \
  --save_gate_map \
  --metrics psnr,ssim,lpips,niqe
```

Precompute DQ latent GT codes for faster Stage-II/III training:

```bash
python scripts/generate_dq_latent_gt.py \
  -i datasets/ffhq/ffhq_512 \
  --opt options/DMDP-FR_stage1_triple.yml \
  --ckpt_path experiments/pretrained_models/dqvae/dqvae_stage1_triple.pth \
  -o experiments/pretrained_models/dqvae
```

[//]: # (## Repository Contents)

[//]: # ()
[//]: # (```text)

[//]: # (basicsr/)

[//]: # (  archs/dmdp_fr_arch.py      # DMDP-FR network and target builders)

[//]: # (  archs/dqvae_arch.py        # Dynamic multi-granularity DQ-VAE prior)

[//]: # (  models/dmdp_fr_model.py    # Stage-II/III training model)

[//]: # (  models/dqvae_model.py      # Stage-I prior training model)

[//]: # (facelib/                     # face detection, alignment, parsing, paste-back)

[//]: # (options/                     # DMDP-FR training configs)

[//]: # (scripts/                     # latent GT generation and visualization/evaluation)

[//]: # (inference_dmdp_fr.py         # inference entry point)

[//]: # (```)

[//]: # (## License)

[//]: # (This repository preserves the license from the original CodeFormer/BasicSR-derived implementation. See `LICENSE`.)
