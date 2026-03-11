# GPU Setup — Multi-Device Training

## Overview

The training code is **100% GPU-agnostic**. You maintain separate conda environments per GPU, and the same scripts run everywhere. Checkpoints are fully portable between devices.

---

## Environment 1: GTX 1080 (CUDA 12.1)

```bash
conda env create -f environment.yml
conda activate gtx-1080-IP
```

Expected: `PyTorch 2.3.1, CUDA available: True, GPU: NVIDIA GeForce GTX 1080`

---

## Environment 2: RTX 4060 (CUDA 12.4)

```bash
conda create -n rtx-4060-IP python=3.11 -y
conda activate rtx-4060-IP
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The RTX 4060 has 8 GB VRAM and supports FP16/BF16 — same config as GTX 1080 works fine. Optionally, you can use larger patches (128³) since the 4060 has more efficient memory management:

```yaml
# Optional RTX 4060 override (save as configs/rtx4060.yaml)
preprocessing:
  patch_size: [128, 128, 128]
```

---

## Environment 3: Google Colab

### Quick Setup (run in first cell)

```python
!git clone https://github.com/YOUR_USER/integrative-project.git
%cd integrative-project
!python scripts/setup_colab.py --data_dir /content/drive/MyDrive/brats_data
```

### Manual Setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install -q monai[all]==1.4.0 nibabel scikit-learn matplotlib tqdm wandb pyyaml scipy optuna

# Upload or link data
!ln -s /content/drive/MyDrive/brats_data data/raw

# Verify
!python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')"
```

### Colab Tips
- **Free tier**: Usually T4 (16 GB VRAM) — can use `128³` patches
- **Runtime disconnects**: Always use `--resume` flag to continue
- **Data**: Upload your BraTS data zip to Drive, extract once, then symlink

---

## Multi-Device Orchestration

### Recommended Assignment

| Device | What to Run | Estimated Time |
|---|---|---|
| GTX 1080 (you) | HP tuning → `unet_4ch` → `cnn_4ch` | ~3 days |
| RTX 4060 (friend) | `unet_flair`, `unet_t1`, `unet_t1ce`, `unet_t2` | ~2 days |
| Colab | `unet_flair_t2`, `unet_t1_t1ce`, cross-modality tests | ~2 days |

### Running Experiments

```bash
# On any device:
python scripts/run_experiment.py --experiment unet_4ch --config full

# Resume after interruption:
python scripts/run_experiment.py --experiment unet_4ch --config full --resume

# On Colab (add ! prefix):
!python scripts/run_experiment.py --experiment unet_flair_t2 --config full
```

### Transferring Results

```bash
# From friend's PC → USB drive
xcopy /E outputs\results\unet_t1 F:\results\unet_t1\

# From Colab → Drive (in Colab cell)
!cp -r outputs/results/ /content/drive/MyDrive/brats_results/

# Merge all results locally
python scripts/merge_results.py \
  --sources outputs/results F:\results /path/to/colab/results \
  --target outputs/results_merged
```

### Checkpoint Portability

Checkpoints are fully portable. A model trained on one GPU loads on any other:

```python
checkpoint = torch.load("checkpoints/unet_4ch/best_model.pth", map_location="cuda:0")
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA not available` | Check `nvidia-smi`. Ensure correct env is active. |
| `CUDA out of memory` | Use `--config dev` for testing. For full: batch size is already 1. |
| `No kernel image` | Wrong PyTorch/CUDA version for GPU. Check env. |
| Import errors | `conda activate <env>` in a fresh terminal. |
| Colab disconnects | Use `--resume` flag. Checkpoints save every epoch. |
| `num_workers` deadlock (Windows) | Set `num_workers: 0` in your config YAML. |
