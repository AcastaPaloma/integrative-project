# GPU Setup — Switching Between GTX 1080 and RTX 4060

## Overview

The training code is **100% GPU-agnostic** — the only difference between the two setups is the PyTorch + CUDA build installed in the conda environment. You maintain two separate conda environments and activate whichever matches your current GPU.

---

## Environment 1: GTX 1080 (CUDA 12.1)

### Create

```bash
conda env create -f environment.yml
conda activate gtx-1080-IP
```

### Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `PyTorch 2.3.1, CUDA available: True, GPU: NVIDIA GeForce GTX 1080`

---

## Environment 2: RTX 4060 (CUDA 12.4)

### Create

```bash
conda create -n rtx-4060-IP python=3.11 -y
conda activate rtx-4060-IP
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `PyTorch 2.5.1, CUDA available: True, GPU: NVIDIA GeForce RTX 4060`

---

## Switching Between GPUs

Just activate the correct environment:

```bash
# On the GTX 1080 machine
conda activate gtx-1080-IP

# On the RTX 4060 machine
conda activate rtx-4060-IP
```

Then run any script normally — no code changes needed:

```bash
python scripts/train.py --config full
```

---

## Transferring Checkpoints Between GPUs

Checkpoints are fully portable. A model trained on one GPU loads on the other:

```python
# Saved on GTX 1080, loaded on RTX 4060 (or vice versa)
checkpoint = torch.load("checkpoints/best_model.pth", map_location="cuda:0")
```

You can start training on the GTX 1080 for testing, transfer the checkpoint, and continue (or retrain from scratch) on the RTX 4060.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA not available` | Check `nvidia-smi` works. Ensure CUDA toolkit matches driver. |
| `CUDA out of memory` | Use `--config dev` for testing (64³ patches). For full training, batch size is already 1. |
| `RuntimeError: CUDA error: no kernel image` | PyTorch build doesn't support this GPU's compute capability. Ensure correct env is active. |
| Import errors after switching envs | Run `conda activate <env>` in a **fresh** terminal. |
