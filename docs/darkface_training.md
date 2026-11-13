# Three-stage training with triple granularity on darkface

These configurations target the paired synthetic low-light [FFHQ](https://github.com/NVlabs/ffhq-dataset) data generated for this project: 256 x 256 RGB images, with 10,000 training pairs, 1,000 validation pairs, and 1,000 test pairs.
Source images are randomly sampled from FFHQ without replacement. Each high-quality face image serves as the normally illuminated GT, with a corresponding synthetic low-light LQ image forming a matched pair.

LQ synthesis combines general facial degradation with low-light simulation. The general degradation procedures follow [GFRNet (ECCV 2018)](https://github.com/csxmli2016/GFRNet) and [DFDNet (ECCV 2020)](https://github.com/csxmli2016/DFDNet), while low-light simulation follows [PASDiff (ECCV 2026)](https://github.com/YlNi26/PASDiff). Together, these procedures produce face restoration samples with low illumination and accompanying image degradation.

## Data paths and working directory

Run all commands from the DMDP-FR repository root. The default directory layout is:

```text
parent/
  DMDP-FR/
    basicsr/
    options/
  darkface/
    train/gt/  train/lq/
    val/gt/    val/lq/
    test/gt/   test/lq/
```

If the data are stored elsewhere, update `datasets.*.dataroot_gt` and `dataroot_lq` in all three YAML files.
Use only the `train` and `val` splits during training and validation; reserve `test` for the final evaluation.

## Adaptations for the three stages

| Stage | Data | Objective |
| --- | --- | --- |
| Stage I | train/gt | Reconstruct normally illuminated HQ faces and learn the codebook and routing. |
| Stage II | train/lq + train/gt | Predict HQ codes, routing, and features from low-light inputs. |
| Stage III | train/lq + train/gt | Retain code and routing supervision while adding image-level supervision with L1, LPIPS, and GAN losses. |

The network retains three latent grids of 8 x 8, 16 x 16, and 32 x 32. The encoder's `ch_mult` is changed to six levels, and the decoder to four levels; three upsampling operations convert the 32 x 32 latent representation into a 256 x 256 image.
The prior architecture is consistent across all three stages. Stages II and III fuse features at resolutions of 64 and 128, without additional fusion at the final 256 resolution.
The original repository's 512 x 512 weights and latent-code caches cannot be used directly as strictly matching checkpoints for this architecture.

Stages II and III use `PairedImageDataset` to read the saved low-light images without applying additional online CodeFormer degradations.
Horizontal flips are synchronized between LQ and GT; no rotation, color jitter, or additional resizing is applied.
Code and routing supervision are extracted online from the augmented GT using the same frozen Stage-I teacher. Keep `latent_gt_path` unset.
Stage III disables `use_large_de_train` and applies image-level supervision at every iteration.
Stage-I validation evaluates HQ reconstruction, while the later stages evaluate low-light restoration. All stages report PSNR and SSIM on the full RGB images.

## Dependencies and perceptual loss weights

For Stage I, `weights/lpips/vgg.pth` must contain the LPIPS VGG linear weights; it cannot be replaced with the full TorchVision VGG16 weights.
If the file is not available, copy it from the installed `lpips` package:

```bash
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import shutil

source = Path(find_spec('lpips').origin).parent / 'weights' / 'v0.1' / 'vgg.pth'
target = Path('weights/lpips/vgg.pth')
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    shutil.copy2(source, target)
print(target.resolve())
PY
```

The TorchVision VGG16 feature weights must also be available; they may be downloaded automatically when the perceptual loss is first constructed. For offline training, prepare the corresponding Torch Hub cache in advance.
