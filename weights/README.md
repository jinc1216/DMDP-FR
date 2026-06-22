# Weights

[//]: # (Large pretrained weights are intentionally not tracked in this compact repository.)

Expected layout:

```text
weights/
  lpips/vgg.pth
  facelib/
  dlib/
experiments/pretrained_models/
  dmgqvae/dmgqvae_stage1_triple.pth
  dmdp_fr_stage2/net_g_latest.pth
  dmdp_fr/dmdp_fr_stage3.pth
```

The DMGQ-VAE checkpoint path is referenced by the stage-2/stage-3 YAML files. Edit `stage1_model_path`, `network_vqgan.model_path`, or `path.pretrain_network_g` if your checkpoints are stored elsewhere.
