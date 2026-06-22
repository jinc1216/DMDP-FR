# DMDP-FR

Official-style compact implementation of **DMDP-FR: Dynamic Multi-Granularity Discrete Prior Representation for Blind Face Restoration**. The repository keeps only the code path needed by DMDP-FR: dynamic DQ-VAE prior learning, coarse-to-fine code prediction, granularity-aware fusion, training configs, inference, and visualization/evaluation utilities.

> Note: the Python registry names use `DMDPFR` and `DMDPFRModel` because Python identifiers cannot contain `-`. User-facing files, commands, and documentation use the paper name `DMDP-FR`.

## Framework

DMDP-FR restores blind degraded faces in three stages:

| Stage | Module | Purpose |
| --- | --- | --- |
| Stage I | Dynamic Multi-Granularity Quantization (DMGQ) | Learns a discrete HQ face prior with adaptive coarse/medium/fine token allocation. |
| Stage II | Coarse-to-Fine Code Prediction (CFCP) | Predicts HQ-aligned discrete codes from LQ inputs, recovering global structure before local details. |
| Stage III | Granularity-Aware Multi-Scale Fusion (GAMF) | Injects LQ-guided features into the fixed HQ decoder using predicted granularity distributions. |

![DMDP-FR framework](assets/figures/framework.png)

## Results From The TCSVT Manuscript

### Synthetic CelebA-Test

| Method | FID (lower) | LPIPS (lower) | NIQE (lower) | IDA (lower) | PSNR (higher) | SSIM (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeFormer | 63.621 | 0.365 | 4.516 | 1.019 | 21.451 | 0.581 |
| VQFR | 55.456 | 0.463 | 3.314 | 1.197 | 19.487 | 0.481 |
| DifFace | 51.247 | 0.347 | 4.631 | 1.060 | 22.190 | 0.633 |
| GFP-GAN | 46.958 | 0.453 | 4.061 | 1.268 | 19.574 | 0.522 |
| **DMDP-FR** | 52.096 | **0.346** | 5.003 | **1.019** | **22.216** | **0.649** |

![Synthetic visual comparison](assets/figures/synthetic_results.png)

### Real-World Benchmarks

| Method | LFW FID (lower) | LFW MUSIQ (higher) | WebPhoto FID (lower) | WebPhoto MUSIQ (higher) | WIDER FID (lower) | WIDER MUSIQ (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeFormer | 54.407 | 71.430 | 86.456 | 74.000 | 40.012 | 69.310 |
| VQFR | 51.821 | 74.745 | 78.251 | 72.009 | 45.122 | 64.013 |
| GFP-GAN | 51.499 | 73.569 | 91.539 | 72.097 | 40.468 | **72.814** |
| InterLCM | 57.178 | 74.692 | 79.492 | **75.798** | 41.685 | 65.447 |
| **DMDP-FR** | 50.110 | **74.810** | 88.624 | 69.133 | **35.230** | 71.859 |

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

This compact repository is designed to run from the repository root. No package installation step is required.

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

## Repository Contents

```text
basicsr/
  archs/dmdp_fr_arch.py      # DMDP-FR network and target builders
  archs/dqvae_arch.py        # Dynamic multi-granularity DQ-VAE prior
  models/dmdp_fr_model.py    # Stage-II/III training model
  models/dqvae_model.py      # Stage-I prior training model
facelib/                     # face detection, alignment, parsing, paste-back
options/                     # DMDP-FR training configs
scripts/                     # latent GT generation and visualization/evaluation
inference_dmdp_fr.py         # inference entry point
```

## License

This repository preserves the license from the original CodeFormer/BasicSR-derived implementation. See `LICENSE`.
