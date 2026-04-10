# Brain Tumor Segmentation — Pseudo-Code Reference

This document describes the logic of every module in the codebase in plain pseudo-code.
The project trains and evaluates 3D deep-learning models (U-Net, plain CNN) on the BraTS brain-tumour dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Configuration (`src/utils/config.py`)](#2-configuration)
3. [Reproducibility (`src/utils/seed.py`)](#3-reproducibility)
4. [Logging (`src/utils/logging.py`)](#4-logging)
5. [Dataset Discovery (`src/data/dataset.py`)](#5-dataset-discovery)
6. [Train / Val / Test Splits (`src/data/splits.py`)](#6-data-splits)
7. [Transform Pipelines (`src/data/transforms.py`)](#7-transform-pipelines)
8. [Models](#8-models)
   - [3D U-Net (`src/models/unet3d.py`)](#81-3d-u-net)
   - [Plain 3D CNN (`src/models/cnn3d.py`)](#82-plain-3d-cnn)
9. [Loss Function (`src/training/losses.py`)](#9-loss-function)
10. [Trainer (`src/training/trainer.py`)](#10-trainer)
11. [Inference (`src/inference/predict.py`)](#11-inference)
12. [Evaluation Metrics (`src/evaluation/metrics.py`)](#12-evaluation-metrics)
13. [Visualisation (`src/evaluation/visualize.py`)](#13-visualisation)
14. [Evaluation Report (`src/evaluation/report.py`)](#14-evaluation-report)
15. [Statistical Tests (`src/evaluation/statistical_tests.py`)](#15-statistical-tests)
16. [Scripts](#16-scripts)
    - [train.py](#161-scriptstrainpy)
    - [predict.py](#162-scriptspredictpy)
    - [evaluate.py](#163-scriptsevaluatepy)
    - [run_experiment.py](#164-scriptsrun_experimentpy)
    - [compare_models.py](#165-scriptscompare_modelspy)
    - [tune_hyperparams.py](#166-scriptstune_hyperparamspy)

---

## 1. Project Overview

```
PROJECT
├── configs/          YAML configuration files (default, dev, full, tuned, experiments)
├── notebooks/        Jupyter notebooks (inspect, train, predict, evaluate)
├── scripts/          Entry-point scripts (train, predict, evaluate, experiment, compare, tune)
└── src/
    ├── data/         Dataset discovery, train/val/test splits, MONAI transform pipelines
    ├── models/       3D U-Net (MONAI) and plain 3D CNN architectures
    ├── training/     Loss function and Trainer class
    ├── inference/    Sliding-window inference, NIfTI saving
    ├── evaluation/   Metrics, visualisation, report generation, statistical tests
    └── utils/        Config loader, seed, WandB logging helpers
```

**Data format:** BraTS NIfTI volumes — four MRI modalities (FLAIR, T1, T1ce, T2) + segmentation mask.

**Segmentation targets (3 overlapping binary channels):**
- Channel 0 — Whole Tumour (WT): labels 1, 2, 4
- Channel 1 — Tumour Core  (TC): labels 1, 4
- Channel 2 — Enhancing Tumour (ET): label 4

---

## 2. Configuration

**File:** `src/utils/config.py`

```
FUNCTION load_config(config_name):
    load default.yaml into base_config

    IF config_name != "default":
        load {config_name}.yaml into overrides
        base_config = deep_merge(base_config, overrides)
            // deep_merge recursively overrides nested keys

    FOR each key in base_config["paths"]:
        resolve path relative to PROJECT_ROOT

    RETURN base_config


FUNCTION get_config_from_args():
    parse --config CLI argument  (choices: default | dev | full)
    RETURN load_config(parsed_config_name)


CLASS ConfigAccessor(dict):
    // Wraps a config dict so keys are accessible as dot-notation attributes
    // e.g. cfg.training.lr  instead of  cfg["training"]["lr"]
    CONSTRUCTOR(d):
        FOR each (key, value) in d:
            IF value is dict:
                self.key = ConfigAccessor(value)
            ELSE:
                self.key = value
```

---

## 3. Reproducibility

**File:** `src/utils/seed.py`

```
FUNCTION set_seed(seed):
    random.seed(seed)
    numpy.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set cudnn.deterministic = True
    set cudnn.benchmark     = False
    set env PYTHONHASHSEED  = seed
    print confirmation message
```

---

## 4. Logging

**File:** `src/utils/logging.py`

```
GLOBAL _run = None   // active WandB run handle


FUNCTION init_wandb(config, run_name):
    IF wandb disabled in config OR wandb not installed:
        print "WandB disabled"
        RETURN
    _run = wandb.init(project, name=run_name, config=config, mode)


FUNCTION log_metrics(metrics_dict, step):
    IF _run is active:
        _run.log(metrics_dict, step=step)


FUNCTION log_image(key, figure, caption, step):
    IF _run is active:
        _run.log({key: wandb.Image(figure, caption)}, step=step)


FUNCTION log_table(key, columns, data):
    IF _run is active:
        _run.log({key: wandb.Table(columns, data)})


FUNCTION finish_wandb():
    IF _run is active:
        _run.finish()
        _run = None


FUNCTION print_log(message, level="INFO"):
    print "[{level}] {message}"
```

---

## 5. Dataset Discovery

**File:** `src/data/dataset.py`

```
FUNCTION discover_brats_samples(data_root, max_samples=None):
    // Scan for BraTS-format patient directories

    patient_dirs = sorted directories under data_root whose name contains "Training"

    IF patient_dirs is empty:
        // Try BraTS 2015 layout (HGG / LGG sub-folders)
        FOR subdir_name IN ["HGG", "LGG"]:
            patient_dirs += sorted subdirectories of data_root/subdir_name

    samples = []

    FOR each patient_dir IN patient_dirs:
        patient_id = directory name
        nii_files   = all *.nii, *.nii.gz, *.mha files in patient_dir

        FOR each modality IN [flair, t1ce, t1, t2, seg]:
            matched = None
            FOR each file IN nii_files:
                // match file name to modality (handle t1 vs t1ce overlap)
                IF file matches modality:
                    matched = file path
            IF matched is None:
                print WARNING "Missing {modality} for {patient_id}"
            sample[modality] = matched

        IF all required modalities (flair, t1, t1ce, t2, seg) are present:
            samples.append(sample)
        ELSE:
            print SKIP with list of missing modalities

    IF max_samples is set:
        samples = samples[:max_samples]

    print "Found {N} valid samples"
    RETURN samples


FUNCTION get_monai_file_list(samples, modalities=None, zero_pad_to=None):
    // Convert sample dicts to MONAI-compatible file lists

    IF modalities is None:
        modalities = [flair, t1, t1ce, t2]   // all 4

    file_list = []
    FOR each sample IN samples:
        IF zero_pad_to is set:
            // Build 4-channel image list; use None for channels not in modalities
            // (None channels are zeroed by transforms — cross-modality inference)
            image_paths = [sample[mod] if mod in modalities ELSE None
                           for mod in [flair, t1, t1ce, t2]]
        ELSE:
            image_paths = [sample[mod] for mod in modalities]

        file_list.append({
            "image":      image_paths,
            "label":      sample["seg"],
            "patient_id": sample["patient_id"],
            "modalities": modalities,
        })

    RETURN file_list
```

---

## 6. Data Splits

**File:** `src/data/splits.py`

```
FUNCTION create_splits(samples, ratios, seed, splits_dir, force=False):
    // Create or load a deterministic train / val / test split

    IF split.json exists AND force is False:
        RETURN _load_splits(split.json, samples)

    // Create new split
    ASSERT sum(ratios) == 1.0

    patient_ids = [s["patient_id"] for s in samples]

    // Two-stage stratified split
    train_ids, val_test_ids = train_test_split(
        patient_ids,
        test_size = val_ratio + test_ratio,
        random_state = seed
    )

    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size = test_ratio / (val_ratio + test_ratio),
        random_state = seed
    )

    save {seed, ratios, train_ids, val_ids, test_ids} to split.json

    RETURN (
        [sample for pid in train_ids],
        [sample for pid in val_ids],
        [sample for pid in test_ids]
    )


FUNCTION _load_splits(split_file, samples):
    load split_data from split.json
    id_to_sample = {patient_id → sample}

    train_samples = [id_to_sample[pid] for pid in split_data["train"] if pid exists]
    val_samples   = [id_to_sample[pid] for pid in split_data["val"]   if pid exists]
    test_samples  = [id_to_sample[pid] for pid in split_data["test"]  if pid exists]

    // Guard for small-sample dev mode where saved split may not cover all available samples
    IF len(samples) >= 3 AND (val_samples is empty OR test_samples is empty):
        print "Subset too small — re-splitting available samples"
        re-split patient_ids with same ratios using train_test_split

    RETURN (train_samples, val_samples, test_samples)
```

---

## 7. Transform Pipelines

**File:** `src/data/transforms.py`

### ConvertBraTSLabels

```
CLASS ConvertBraTSLabels (MONAI MapTransform):
    // Convert integer BraTS labels {0, 1, 2, 4} → 3-channel binary tensor

    CALL(data):
        label = data["label"]   // shape: (1, D, H, W)

        wt = (label == 1) OR (label == 2) OR (label == 4)  // Whole Tumour
        tc = (label == 1) OR (label == 4)                   // Tumour Core
        et = (label == 4)                                    // Enhancing Tumour

        data["label"] = concatenate([wt, tc, et], dim=0)    // shape: (3, D, H, W)
        RETURN data
```

### Training Transform Pipeline

```
FUNCTION get_train_transforms(cfg):
    RETURN sequential pipeline:
        1. LoadImaged          // load NIfTI files from disk
        2. EnsureChannelFirstd // shape → (C, D, H, W)
        3. Orientationd        // standardise to RAS orientation
        4. Spacingd            // resample to cfg.spacing (bilinear for image, nearest for label)
        5. NormalizeIntensityd // z-score normalisation per channel, nonzero voxels only
        6. ConvertBraTSLabels  // integer labels → 3-channel binary masks
        7. CropForegroundd     // remove empty background (margin = 10 voxels)
        8. RandCropByPosNegLabeld  // random patch: 3:1 foreground:background ratio
        // ---- Augmentation ----
        9.  RandFlipd (axis 0, 1, 2) with cfg.random_flip_prob
        10. RandRotate90d            with cfg.random_rotate90_prob
        11. RandShiftIntensityd      ± cfg.intensity_shift_offset  (prob 0.5)
        12. RandScaleIntensityd      ± cfg.intensity_scale_range   (prob 0.5)
```

### Validation Transform Pipeline

```
FUNCTION get_val_transforms(cfg):
    RETURN sequential pipeline (NO augmentation, NO random cropping):
        1. LoadImaged
        2. EnsureChannelFirstd
        3. Orientationd
        4. Spacingd
        5. NormalizeIntensityd
        6. ConvertBraTSLabels
```

---

## 8. Models

### 8.1 3D U-Net

**File:** `src/models/unet3d.py`

```
FUNCTION get_model(cfg):
    architecture = cfg["model"]["architecture"]  // "UNet" or "CNN"

    IF architecture == "CNN":
        RETURN get_cnn_model(cfg)   // see §8.2

    // Default: MONAI U-Net (encoder-decoder with skip connections)
    model = MONAI_UNet(
        spatial_dims  = 3,
        in_channels   = cfg.model.in_channels,
        out_channels  = cfg.model.out_channels,
        channels      = cfg.model.channels,          // e.g. [32, 64, 128, 256]
        strides       = cfg.model.strides,           // downsampling per level
        num_res_units = cfg.model.num_res_units,
        norm          = InstanceNorm,
        dropout       = cfg.model.dropout,
    )
    print parameter count
    RETURN model


FUNCTION load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print epoch and best_dice from checkpoint
    RETURN checkpoint
```

### 8.2 Plain 3D CNN

**File:** `src/models/cnn3d.py`

```
CLASS ConvBlock3D:
    // Double convolution block: two × (Conv3d → InstanceNorm → LeakyReLU)
    // Optional Dropout3d appended if dropout > 0

    FORWARD(x):
        RETURN sequential_block(x)


CLASS CNN3D:
    // Encoder-decoder WITHOUT skip connections (baseline vs U-Net)
    // Same channel counts and depth as U-Net for fair comparison

    CONSTRUCTOR(in_channels, out_channels, channels, dropout):
        // Encoder
        FOR each ch in channels:
            encoders.append( ConvBlock3D(prev_ch → ch) )
            pools.append   ( MaxPool3d(2)               )

        // Bottleneck
        bottleneck = ConvBlock3D(channels[-1] → channels[-1] * 2)

        // Decoder (no skip connection concatenation)
        FOR each ch in reversed(channels):
            upconvs.append ( ConvTranspose3d(prev_ch → ch, stride=2) )
            decoders.append( ConvBlock3D(ch → ch)                     )

        final_conv = Conv3d(channels[0] → out_channels, kernel=1)

    FORWARD(x):
        // Encoder: just downsample, save nothing
        FOR (encoder, pool) in zip(encoders, pools):
            x = encoder(x)
            x = pool(x)

        x = bottleneck(x)

        // Decoder: upsample only (no skip concatenation)
        FOR (upconv, decoder) in zip(upconvs, decoders):
            x = upconv(x)
            x = decoder(x)

        RETURN final_conv(x)


FUNCTION get_cnn_model(cfg):
    model = CNN3D(
        in_channels  = cfg.model.in_channels,
        out_channels = cfg.model.out_channels,
        channels     = cfg.model.channels,
        dropout      = cfg.model.dropout,
    )
    print parameter count
    RETURN model
```

---

## 9. Loss Function

**File:** `src/training/losses.py`

```
FUNCTION get_loss_function(cfg):
    loss_cfg = cfg["training"]["loss"]

    // MONAI DiceCE: weighted sum of Dice loss + Binary Cross-Entropy
    // sigmoid=True because labels are multi-label (channels not mutually exclusive)
    loss_fn = DiceCELoss(
        to_onehot_y  = False,
        sigmoid      = True,
        lambda_dice  = loss_cfg.lambda_dice,   // default 1.0
        lambda_ce    = loss_cfg.lambda_ce,     // default 1.0
    )
    RETURN loss_fn
```

---

## 10. Trainer

**File:** `src/training/trainer.py`

```
CLASS Trainer:

    CONSTRUCTOR(model, loss_fn, optimizer, scheduler, train_loader, val_loader, cfg, device, callbacks):
        self.model     = model.to(device)
        self.use_amp   = cfg.training.mixed_precision    // mixed-precision training flag
        self.scaler    = GradScaler(enabled=use_amp)
        // gradient clipping, non-finite handling, early stopping loaded from cfg
        self.best_dice = 0.0


    FUNCTION train():
        FOR epoch in range(start_epoch, total_epochs):
            train_loss               = _train_epoch(epoch)
            dice_per_class, mean_dice, accuracy = _validate_epoch(epoch)

            scheduler.step()

            log_metrics to console and WandB

            _save_checkpoint(epoch, mean_dice, is_last=True)

            IF mean_dice > best_dice + min_delta:
                best_dice = mean_dice
                _save_checkpoint(epoch, mean_dice, is_best=True)
                epochs_without_improvement = 0
            ELSE:
                epochs_without_improvement += 1

            IF early_stopping AND epochs_without_improvement >= patience:
                BREAK

            FOR callback in callbacks:
                callback(epoch, mean_dice)

        RETURN training history dict


    FUNCTION _train_epoch(epoch):
        model.train()
        epoch_loss = 0; step_count = 0

        FOR batch in train_loader:
            inputs, labels = batch["image"], batch["label"]

            // Stability check on inputs
            IF inputs or labels contain NaN/Inf:
                skip batch; reduce LR if not reduced yet; CONTINUE

            WITH autocast(enabled=use_amp):
                outputs = model(inputs)

            IF outputs contain NaN/Inf:
                skip backward; reduce LR; CONTINUE

            loss = loss_fn(outputs [cast to fp32 if amp+loss_in_fp32], labels)

            IF loss is NaN/Inf:
                skip backward; reduce LR; CONTINUE

            scaler.scale(loss).backward()

            IF gradient_clip enabled:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_norm)
                IF grad_norm is NaN/Inf:
                    skip optimizer step; update scaler; reduce LR; CONTINUE

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1

        // If too many skipped batches, disable AMP for remaining epochs
        IF skipped_batches > 0 AND amp_fallback_enabled:
            use_amp = False

        RETURN epoch_loss / step_count


    FUNCTION _validate_epoch(epoch):
        model.eval()
        dice_metric.reset()
        total_correct = 0; total_voxels = 0

        FOR batch in val_loader:
            inputs, labels = batch

            // Full-volume inference via sliding window
            outputs = sliding_window_inference(
                inputs, roi_size=val_roi_size,
                sw_batch_size, predictor=model, overlap=sw_overlap
            )

            // Post-process: sigmoid → threshold at 0.5
            preds = [sigmoid(o) > 0.5 for o in decollate_batch(outputs)]

            dice_metric.update(preds, labels)
            accumulate voxel-wise accuracy

            // Log sample visualisation every N epochs
            IF batch_idx == 0 AND (epoch+1) % log_images_every == 0:
                _log_sample_visualization(inputs[0], labels[0], preds[0], epoch)

        dice_per_class = dice_metric.aggregate()   // [WT, TC, ET]
        mean_dice      = mean(dice_per_class)
        accuracy       = total_correct / total_voxels

        RETURN (dice_per_class, mean_dice, accuracy)


    FUNCTION _save_checkpoint(epoch, dice, is_best=False, is_last=False):
        checkpoint = {epoch, model_state, optimizer_state, scheduler_state,
                      scaler_state, best_dice, config}
        IF is_best: save to "best_model.pth"
        IF is_last: save to "last_model.pth"


    FUNCTION resume_from_checkpoint(checkpoint_path):
        load checkpoint
        restore model, optimizer, scheduler, scaler states
        self.best_dice   = checkpoint.best_dice
        self.start_epoch = checkpoint.epoch + 1


    FUNCTION _backoff_learning_rate():
        // Called when NaN/Inf detected; halves LR down to a minimum floor
        FOR each param_group in optimizer:
            new_lr = max(old_lr * backoff_factor, min_lr)
            param_group["lr"] = new_lr
```

---

## 11. Inference

**File:** `src/inference/predict.py`

```
FUNCTION run_inference(cfg, checkpoint_path, samples, output_dir, device, num_cases, modalities):

    // Setup
    output_dir.mkdir()
    samples = samples[:num_cases] if num_cases is set

    // Build model and load weights
    model = get_model(cfg)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    // Prepare data
    val_transforms = get_val_transforms(cfg)
    file_list      = get_monai_file_list(samples, modalities)
    dataset        = MONAI_Dataset(file_list, transform=val_transforms)

    inferer = SlidingWindowInferer(
        roi_size    = cfg.preprocessing.patch_size,
        sw_batch_size, overlap, mode="gaussian"
    )
    post_pred = sigmoid → threshold(0.5)

    results = []

    FOR idx, data in enumerate(dataset):
        patient_id = samples[idx]["patient_id"]
        image      = data["image"].unsqueeze(0)  // (1, C, D, H, W)

        ASSERT image.channels == cfg.model.in_channels   // channel mismatch guard

        WITH torch.no_grad() AND autocast:
            output = inferer(image, model)   // (1, 3, D, H, W)

        pred = post_pred(output[0])          // (3, D, H, W) binary
        pred_np = pred.cpu().numpy()

        // Convert binary channels back to BraTS integer labels for NIfTI
        seg_map = zeros(D, H, W)
        seg_map[pred[0] == 1] = 2   // WT → edema
        seg_map[pred[1] == 1] = 1   // TC → NCR/NET
        seg_map[pred[2] == 1] = 4   // ET → enhancing

        // Save NIfTI using original FLAIR affine
        ref_nii  = nibabel.load(samples[idx]["flair"])
        pred_nii = Nifti1Image(seg_map, affine=ref_nii.affine)
        nibabel.save(pred_nii, output_dir / "{patient_id}_pred.nii.gz")

        results.append({patient_id, prediction_path, prediction (numpy), label (numpy)})

    RETURN results
```

---

## 12. Evaluation Metrics

**File:** `src/evaluation/metrics.py`

```
FUNCTION dice_coefficient(pred, gt):
    intersection = sum(pred * gt)
    IF sum(pred) + sum(gt) == 0:
        RETURN 1.0   // both empty = perfect match
    RETURN (2 * intersection) / (sum(pred) + sum(gt))


FUNCTION precision_score(pred, gt):
    tp = sum(pred * gt)
    fp = sum(pred * (1 - gt))
    IF tp + fp == 0: RETURN 0.0
    RETURN tp / (tp + fp)


FUNCTION recall_score(pred, gt):
    tp = sum(pred * gt)
    fn = sum((1 - pred) * gt)
    IF tp + fn == 0: RETURN 0.0
    RETURN tp / (tp + fn)


FUNCTION hausdorff_distance_95(pred, gt):
    // 95th-percentile Hausdorff distance via KD-tree nearest-neighbour lookup

    pred_points = voxel coordinates where pred == 1
    gt_points   = voxel coordinates where gt   == 1

    IF either set is empty:
        IF both empty: RETURN 0.0
        ELSE:          RETURN inf

    dist_pred_to_gt = nearest-neighbour distance from each pred point to gt
    dist_gt_to_pred = nearest-neighbour distance from each gt   point to pred
    all_dists       = concatenate both distance arrays

    RETURN percentile(all_dists, 95)


FUNCTION compute_case_metrics(pred, gt):
    // pred, gt shape: (3, D, H, W) — channels WT, TC, ET

    FOR each channel c in [0, 1, 2]:
        dice.append       ( dice_coefficient(pred[c], gt[c])  )
        hausdorff95.append( hausdorff_distance_95(pred[c], gt[c]) )
        precision.append  ( precision_score(pred[c], gt[c])   )
        recall.append     ( recall_score(pred[c], gt[c])      )

    RETURN {dice, hausdorff95, precision, recall}


FUNCTION aggregate_metrics(all_case_metrics):
    // Compute mean ± std per class per metric across all cases

    FOR each metric in [dice, hausdorff95, precision, recall]:
        FOR each class c in [WT, TC, ET]:
            values = [case[metric][c] for case in all_case_metrics]
            result["{metric}/{class}/mean"] = mean(values)
            result["{metric}/{class}/std"]  = std(values)
            result["{metric}/{class}/values"] = values

        result["{metric}/mean_all"] = mean across all classes and cases

    RETURN result
```

---

## 13. Visualisation

**File:** `src/evaluation/visualize.py`

```
FUNCTION create_overlay_figure(image, gt, pred, patient_id, slice_idx, save_path):
    // 3-panel figure: [GT overlay | Pred overlay | Difference map]

    mri = image[0]   // FLAIR channel (D, H, W)

    IF slice_idx is None:
        slice_idx = axial slice with most WT voxels  // auto-select tumour centre

    fig = plt.subplots(1, 3)

    Panel 1 — MRI + Ground Truth:
        display mri[slice_idx] in greyscale
        overlay(gt[:, slice_idx])   // colour-coded WT=green, TC=blue, ET=red

    Panel 2 — MRI + Prediction:
        display mri[slice_idx]
        overlay(pred[:, slice_idx])

    Panel 3 — Difference (WT channel):
        display mri[slice_idx]
        TP = pred[0] == 1 AND gt[0] == 1  → blue
        FP = pred[0] == 1 AND gt[0] == 0  → red
        FN = pred[0] == 0 AND gt[0] == 1  → green

    IF save_path: save PNG at 150 dpi
    RETURN figure


FUNCTION create_multi_view_figure(image, gt, pred, patient_id, save_path):
    // 3×3 figure: rows = [axial, coronal, sagittal], cols = [GT, Pred, Diff]

    mri = image[0]
    center = centroid of WT mask  (fallback: volume centre)

    FOR each view (axial, coronal, sagittal) at corresponding center coordinate:
        extract 2-D slice of mri, gt, pred in that view
        draw GT overlay, Pred overlay, Difference map

    IF save_path: save PNG
    RETURN figure
```

---

## 14. Evaluation Report

**File:** `src/evaluation/report.py`

```
FUNCTION generate_report(aggregated_metrics, case_metrics, patient_ids, visualization_dir, output_path):

    lines = []

    // Section 1: Summary table (mean ± std per class)
    lines += markdown table header
    FOR each class (WT, TC, ET):
        lines += row with Dice, HD95, Precision, Recall (mean ± std)
    lines += "Mean Dice (all classes): {value}"

    // Section 2: Per-case breakdown table
    lines += markdown table header
    FOR each (patient_id, metrics) pair:
        lines += row with Dice WT / TC / ET and HD95 WT / TC / ET

    // Section 3: Qualitative results (embedded PNGs)
    FOR each PNG in visualization_dir:
        lines += "### {filename}"
        lines += "![image]({filename})"

    write lines to output_path (.md)
    write aggregated_metrics to output_path (.json)

    RETURN output_path
```

---

## 15. Statistical Tests

**File:** `src/evaluation/statistical_tests.py`

```
FUNCTION paired_wilcoxon_test(scores_a, scores_b, name_a, name_b):
    // Non-parametric paired test for significance between two models
    statistic, p_value = scipy.stats.wilcoxon(scores_a, scores_b, two-sided)
    significant = p_value < 0.05
    better = name_a if mean(scores_a) > mean(scores_b) else name_b
    RETURN {test, model names, means, mean_diff, median_diff, statistic, p_value,
            significant, interpretation, n_patients}


FUNCTION friedman_test(scores_dict):
    // Non-parametric repeated-measures ANOVA for >2 models
    score_arrays = list of per-patient arrays per model
    statistic, p_value = scipy.stats.friedmanchisquare(*score_arrays)

    // Compute average rank per model (higher Dice = lower rank)
    ranks_matrix = rank each model's score per patient (descending)
    avg_ranks    = mean rank per model
    ranking      = models sorted by avg_rank

    RETURN {test, statistic, p_value, significant, rankings, model_means}


FUNCTION bootstrap_confidence_interval(scores, n_bootstrap=10000, confidence=0.95, seed):
    bootstrap_means = []
    FOR i in range(n_bootstrap):
        sample = resample(scores, with_replacement=True)
        bootstrap_means.append(mean(sample))

    alpha = 1 - confidence
    ci_lower = percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = percentile(bootstrap_means, (1-alpha/2) * 100)

    RETURN {mean, std, ci_lower, ci_upper, confidence, n_bootstrap}


FUNCTION cohens_d(scores_a, scores_b):
    diff = scores_a - scores_b
    d    = mean(diff) / std(diff)
    magnitude = "negligible" if |d| < 0.2
              = "small"      if |d| < 0.5
              = "medium"     if |d| < 0.8
              = "large"      otherwise
    RETURN {cohens_d, magnitude, interpretation}


FUNCTION mcnemar_test(scores_a, scores_b, threshold=0.5):
    // Compare segmentation failure rates between two models
    success_a = scores_a >= threshold
    success_b = scores_b >= threshold

    both_success = count(success_a AND success_b)
    a_only       = count(success_a AND NOT success_b)
    b_only       = count(NOT success_a AND success_b)
    both_fail    = count(NOT success_a AND NOT success_b)

    n_discordant = a_only + b_only
    IF n_discordant == 0:
        p_value = 1.0
    ELSE IF n_discordant < 25:
        p_value = exact binomial test
    ELSE:
        p_value = chi-squared approximation with continuity correction

    RETURN {test, threshold, contingency_table, statistic, p_value, failure_rates}


FUNCTION run_full_comparison(results, class_names):
    // Runs all tests on a dict of {model_name: {dice_per_patient}}

    FOR each model:
        compute bootstrap_confidence_interval for overall and per-class Dice

    FOR each pair (model_a, model_b):
        paired_wilcoxon_test(mean Dice per patient)
        cohens_d(mean Dice per patient)
        mcnemar_test(mean Dice per patient)

    IF more than 2 models:
        friedman_test(all mean Dice arrays)

    RETURN comprehensive comparison dict
```

---

## 16. Scripts

### 16.1 `scripts/train.py`

```
FUNCTION main():
    parse --config, --resume from CLI

    cfg    = load_config(args.config)
    seed   = set_seed(cfg.seed)
    device = cuda if available else cpu
    init_wandb(cfg)

    // Data
    samples                           = discover_brats_samples(cfg.data_root)
    train_samples, val_samples, _     = create_splits(samples, cfg.split_ratios)
    train_files, val_files            = get_monai_file_list(train_samples / val_samples)
    train_transforms, val_transforms  = get_train_transforms / get_val_transforms(cfg)
    train_ds = CacheDataset(train_files, train_transforms, cache_rate)
    val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    // Model, loss, optimiser, scheduler
    model     = get_model(cfg)
    loss_fn   = get_loss_function(cfg)
    optimizer = AdamW(model.parameters(), lr, weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)

    trainer = Trainer(model, loss_fn, optimizer, scheduler,
                      train_loader, val_loader, cfg, device)

    IF args.resume:
        trainer.resume_from_checkpoint(args.resume)

    history = trainer.train()
    finish_wandb()
```

---

### 16.2 `scripts/predict.py`

```
FUNCTION main():
    parse --config, --checkpoint, --num_cases, --split (val|test)

    cfg    = load_config(args.config)
    samples = discover_brats_samples(cfg.data_root)
    _, val_samples, test_samples = create_splits(samples, cfg.split_ratios)
    target_samples = val_samples if split=="val" else test_samples

    device  = cuda if available else cpu
    results = run_inference(
        cfg, checkpoint_path, target_samples,
        device=device, num_cases=args.num_cases
    )

    print "{N} predictions saved"
```

---

### 16.3 `scripts/evaluate.py`

```
FUNCTION main():
    parse --config, --checkpoint, --split, --skip-inference

    cfg    = load_config(args.config)
    samples = discover_brats_samples(cfg.data_root)
    _, val_samples, test_samples = create_splits(samples, cfg.split_ratios)
    target_samples = val_samples or test_samples

    // Step 1: Run inference (or load existing NIfTI predictions)
    IF NOT args.skip_inference:
        results = run_inference(cfg, checkpoint_path, target_samples, device)
    ELSE:
        FOR each patient:
            load *_pred.nii.gz from predictions_dir
            convert integer labels → 3-channel binary (same logic as ConvertBraTSLabels)
            append to results

    // Step 2: Compute metrics
    FOR each result:
        case_metrics = compute_case_metrics(pred, gt)
        log Dice WT / TC / ET for this patient

    agg = aggregate_metrics(all_case_metrics)
    print aggregated summary table

    // Step 3: Visualisations
    FOR first N results:
        load image data via val_transforms
        create_overlay_figure   → save PNG
        create_multi_view_figure → save PNG

    // Step 4: Report
    generate_report(agg, case_metrics, patient_ids, vis_dir, output_path)
```

---

### 16.4 `scripts/run_experiment.py`

```
FUNCTION main():
    parse --experiment, --config, --resume

    cfg     = load_config(args.config)
    exp_cfg = load experiments.yaml

    experiment = exp_cfg["experiments"][args.experiment]
    // Override model config for this experiment
    cfg["model"]["architecture"] = experiment["model"]
    cfg["model"]["in_channels"]  = len(experiment["modalities"])
    cfg["paths"]["checkpoint_dir"] = checkpoints/{experiment_name}/

    // Same setup as train.py but with modality filtering
    train_files = get_monai_file_list(train_samples, modalities=experiment["modalities"])
    val_files   = get_monai_file_list(val_samples,   modalities=experiment["modalities"])

    // Build, train
    trainer = build_trainer(cfg, device)

    IF args.resume AND last_model.pth exists:
        trainer.resume_from_checkpoint(last_model.pth)

    history = trainer.train()
    save history to training_history.json
    finish_wandb()
```

---

### 16.5 `scripts/compare_models.py`

```
FUNCTION main():
    parse --config, --results_dir

    // Load per-patient Dice arrays from all experiment result directories
    results = {}
    FOR each experiment directory in results_dir:
        IF evaluation_results.json exists:
            results[experiment_name] = {dice_per_patient: array(n_patients, 3)}
        ELSE IF training_history.json exists:
            results[experiment_name] = {training_history, has_evaluation=False}

    eval_results = filter to experiments with evaluation data
    ASSERT len(eval_results) >= 2

    // Run all statistical tests
    comparison = run_full_comparison(eval_results)

    // Export tables
    generate_comparison_tables(comparison, output_dir):
        save confidence_intervals.csv
        save pairwise_wilcoxon.csv
        save effect_sizes.csv
        save model_summary.csv

    save full_comparison.json

    // Print summary to console
    print bootstrap CI per model
    print Friedman test result (if >2 models)
    print pairwise Wilcoxon p-values
```

---

### 16.6 `scripts/tune_hyperparams.py`

```
FUNCTION objective(trial, args, base_cfg, train_samples, val_samples):
    // Optuna objective — sample HPs, short train, return best Dice

    cfg = deep_copy(base_cfg)

    // Sample hyperparameters
    lr            = trial.suggest_float(1e-5, 5e-3, log=True)
    weight_decay  = trial.suggest_float(1e-6, 1e-3, log=True)
    dropout       = trial.suggest_float(0.0,  0.4,  step=0.05)
    channels      = trial.suggest_categorical(["small", "medium"])
    patch_size    = trial.suggest_categorical(["small", "medium"])

    // Apply sampled HPs to config
    cfg.training.optimizer.lr           = lr
    cfg.training.optimizer.weight_decay = weight_decay
    cfg.model.dropout                   = dropout
    cfg.model.channels                  = channels_map[channels]
    cfg.preprocessing.patch_size        = patch_map[patch_size]
    cfg.training.epochs                 = args.epochs_per_trial
    cfg.logging.use_wandb               = False

    // Build model and trainer (standard pipeline)
    set_seed(base_seed + trial.number)
    model, trainer = build_and_train_short(cfg, train_samples, val_samples,
                                           prune_callback=trial.report + trial.should_prune)

    best_dice = trainer.best_dice
    free GPU memory and collected garbage
    RETURN best_dice


FUNCTION main():
    parse --n_trials, --epochs_per_trial, --max_samples, --config

    cfg = load_config(args.config)
    samples = discover_brats_samples(cfg.data_root, max_samples)

    // Ephemeral train/val split (not persisted to disk)
    train_samples, val_samples = train_test_split(samples, test_size=0.3)

    // Optuna study
    study = optuna.create_study(
        direction = "maximize",
        pruner    = MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials)

    // Save best hyperparameters as configs/tuned.yaml
    best = study.best_trial.params
    tuned_config = {preprocessing: {patch_size}, model: {channels, dropout},
                    training: {optimizer: {lr, weight_decay}}}
    yaml.dump(tuned_config, "configs/tuned.yaml")

    print best trial number, Dice, and hyperparameters
```

---

*End of pseudo-code reference.*
