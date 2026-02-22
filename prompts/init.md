

# 🧠 Brain Tumor Segmentation — Training-Only Implementation

## Objective

Train a **3D U-Net** on the **BraTS 2015 dataset** from Kaggle and produce:

* Trained model checkpoint
* Quantitative evaluation metrics
* Visual qualitative results
* Reproducible training pipeline

No Docker.
No frontend.
No API.

Pure research training pipeline.

---

# 📁 Dataset

Dataset:

**Brain Tumor Segmentation in MRI (BraTS 2015)**
[https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data)

Dataset size: ~8GB

### Is Download Necessary?

Yes.

There is no reliable way to train properly without downloading the dataset locally (or using Kaggle runtime directly).

If disk space is tight:

* Download full dataset once
* Optionally create a smaller dev subset

---

# 🎯 Training Goal

Segment tumor subregions from multi-modal MRI:

Input:

* T1
* T1ce
* T2
* FLAIR

Output:

* Multi-class tumor mask

---

# 🛠 Required Stack

Core:

* Python 3.10+
* PyTorch (CUDA enabled)
* MONAI
* NumPy
* NiBabel
* scikit-learn
* matplotlib
* tqdm

Optional:

* TensorBoard

---

# 🧩 Project Structure (Required)

```
brain_tumor_segmentation/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── dataset/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│
├── configs/
├── logs/
├── checkpoints/
└── README.md
```

No notebooks as final deliverable.
Scripts only.

---

# PHASE 1 — Dataset Validation (Mandatory First Step)

Agent must:

1. Inspect dataset structure.
2. Confirm NIfTI format.
3. Identify label values (likely 0, 1, 2, 4).
4. Compute:

   * Class voxel distribution
   * Volume size distribution
5. Create train/val/test split (e.g., 70/15/15).

Deliverable:

* Short report summarizing dataset statistics.

---

# PHASE 2 — Preprocessing Pipeline

Must implement using MONAI transforms.

Required steps:

1. Load images and labels
2. Ensure channel-first format
3. Standardize orientation
4. Resample to consistent spacing
5. Normalize intensities per modality
6. Convert labels to multi-channel (if multi-class)
7. Patch-based sampling (e.g., 128×128×128)
8. Balanced sampling (foreground emphasis)

Validation must use:

* No random augmentations

Training must use:

* Random flips
* Random rotations
* Intensity shifts

Deliverable:

* Preprocessing config
* Logging of applied transforms

---

# PHASE 3 — Model Training

Model:

* 3D U-Net

Training requirements:

* Patch size: 128×128×128
* Batch size: 1 (if 8GB GPU)
* Mixed precision enabled
* Dice + Cross Entropy loss
* AdamW optimizer
* Learning rate scheduler
* Seed control for reproducibility

Training must:

* Save best checkpoint (based on validation Dice)
* Save last checkpoint
* Log training + validation curves

---

# PHASE 4 — Inference Pipeline

Must implement sliding window inference for full 3D volume.

Requirements:

* Handle full resolution MRI
* Reconstruct final segmentation mask
* Save output as NIfTI file

Must test on:

* At least 3 validation cases

---

# PHASE 5 — Evaluation

## Quantitative Metrics

Compute:

* Dice per class
* Mean Dice
* Hausdorff distance
* Precision & recall

Report:

* Mean ± standard deviation
* Per-case breakdown

---

## Qualitative Evaluation

Generate:

* Overlay MRI + ground truth
* Overlay MRI + prediction
* Difference heatmap

For at least 5 cases.

Deliver as:

* Markdown report with embedded images
  OR
* PDF report

---

# 🧪 Testing Requirements

Before completion:

* Confirm no data leakage
* Confirm validation not augmented
* Confirm reproducibility (same seed → similar results)
* Confirm sliding window works

---

# 💻 Hardware Assumptions

Minimum:

* GPU with 8GB VRAM (GTX 1080 or RTX 4060 acceptable)
* 16GB RAM
* 500GB SSD

If GPU unavailable:

* Must train on Kaggle GPU runtime

---

# 📊 Expected Deliverables

Agent must provide:

1. Clean training code
2. Config file for reproducibility
3. Best model checkpoint
4. Training logs
5. Evaluation report
6. Example prediction outputs
7. Instructions to retrain model from scratch

---

# ❌ What NOT To Do

* Do not build frontend
* Do not build API
* Do not containerize yet
* Do not skip evaluation
* Do not report only one Dice number

---

# 🧠 Future Phases (Not Now)

After stable training:

Phase 6 → API
Phase 7 → Docker
Phase 8 → Frontend

But not before the model is proven.

---

# 🔥 Final Instruction to Agent

The goal is a **robust, reproducible, research-grade 3D medical segmentation pipeline**.

Quality of evaluation is more important than architectural complexity.

If any step fails, fix root cause before proceeding.

