# Dehaze as You Desire: Intensity-Controllable and Region-Specific Dehazing

**DYD-Dehazing** is a one-step diffusion framework with controllable dehazing intensity, Gated Detail Compensation (GDCM), and region selection through a spatial mask.

## Setup

Python 3.10 and an NVIDIA GPU with CUDA are required.

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Prepare SD2.1-base, RAM weights, the BERT tokenizer, and training data locally. Set their paths in `configs/train_dehazing.json`. Datasets and model weights are not included.

## Training

Run commands from the repository root.

**1. Train the dehazing model** — `scripts/train_dehazing.py`

```bash
python scripts/train_dehazing.py --config configs/train_dehazing.json
```

The supplied configuration continues from step 3,000 to step 5,000. Set `resume_checkpoint` to the matching initialization checkpoint; the initial 3,000-step training procedure is not included.

**2. Train GDCM** — `scripts/train_gdcm.py`

```bash
python scripts/train_gdcm.py --config configs/train_gdcm.json
```

Set `base_checkpoint`, `base_step`, and `base_sha256` in the GDCM configuration to your trained checkpoint. The dehazing model is frozen while GDCM learns to compensate for details attenuated by VAE compression.

## Inference

**Entry point:** `scripts/infer_dehazing.py`

```bash
python scripts/infer_dehazing.py \
  --checkpoint outputs/gdcm/checkpoints/gdcm-002000 \
  --input path/to/hazy-image-or-folder \
  --output path/to/results \
  --alpha 1.0 --beta 1.0
```

`--alpha` controls dehazing strength; `--beta` controls detail compensation. For a single image, add `--mask path/to/mask.png` to select the processing region (white: apply corrections; black: suppress corrections). Outputs retain the input dimensions. Masked regions pass through the VAE, so unselected pixels are not guaranteed to be identical to the input.

The checkpoint folder must contain `config.json`, `adapters.safetensors`, and `gdcm.safetensors`. Use `--model`, `--ram-weights`, and `--ram-tokenizer` to override resource paths stored in the checkpoint.

## Acknowledgements

This project builds on [PiSA-SR](https://github.com/csslc/PiSA-SR) and [Recognize Anything](https://github.com/xinyu1205/recognize-anything).
