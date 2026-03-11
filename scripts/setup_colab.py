"""
Google Colab setup script.

Auto-installs dependencies, mounts Drive, and sets up the project.

Usage (in Colab cell):
    !git clone https://github.com/YOUR_USER/integrative-project.git
    %cd integrative-project
    !python scripts/setup_colab.py --data_dir /content/drive/MyDrive/brats_data
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup Colab environment")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to BraTS data on Drive")
    parser.add_argument("--skip_mount", action="store_true",
                        help="Skip Drive mount (if already mounted)")
    args = parser.parse_args()

    print("=" * 60)
    print("COLAB SETUP")
    print("=" * 60)

    # 1. Check if we're in Colab
    in_colab = "google.colab" in sys.modules or os.path.exists("/content")
    if not in_colab:
        print("WARNING: Not running in Google Colab. Proceeding anyway...")

    # 2. Mount Google Drive
    if not args.skip_mount and in_colab:
        print("\n[1/5] Mounting Google Drive...")
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("  Drive mounted successfully")
        except Exception as e:
            print(f"  Drive mount failed: {e}")
            print("  Run this in a Colab cell instead:")
            print("  from google.colab import drive; drive.mount('/content/drive')")
    else:
        print("\n[1/5] Skipping Drive mount")

    # 3. Install dependencies
    print("\n[2/5] Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "monai[all]==1.4.0", "nibabel>=5.2", "scikit-learn>=1.4",
        "matplotlib>=3.8", "tqdm>=4.66", "wandb>=0.19", "pyyaml>=6.0",
        "scipy>=1.12", "numpy>=1.24,<2.0", "optuna>=3.0",
    ], check=True)

    # 4. Symlink data directory
    print("\n[3/5] Linking data directory...")
    data_source = Path(args.data_dir)
    data_target = Path("data/raw")

    if data_source.exists():
        # Remove existing data dir if it's empty
        if data_target.exists() and not any(data_target.iterdir()):
            data_target.rmdir()
        elif data_target.is_symlink():
            data_target.unlink()

        if not data_target.exists():
            data_target.symlink_to(data_source)
            print(f"  Linked {data_target} -> {data_source}")
        else:
            print(f"  Data dir already exists: {data_target}")

        # Count samples
        patient_dirs = [d for d in data_source.iterdir()
                        if d.is_dir() and not d.name.startswith(".")]
        print(f"  Found {len(patient_dirs)} patient directories")
    else:
        print(f"  WARNING: Data source not found: {data_source}")
        print(f"  Please upload your BraTS data to: {data_source}")
        print(f"  Or extract the zip there: unzip brats_data.zip -d {data_source}")

    # 5. Verify GPU
    print("\n[4/5] Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  GPU: {gpu_name} ({vram:.1f} GB)")
        else:
            print("  WARNING: No GPU detected! Go to Runtime > Change runtime type > GPU")
    except ImportError:
        print("  PyTorch not installed — run step 2 first")

    # 6. Verify imports
    print("\n[5/5] Verifying imports...")
    try:
        from src.utils.config import load_config
        from src.models.unet3d import get_model
        from src.data.dataset import discover_brats_samples
        print("  All imports successful!")
    except ImportError as e:
        print(f"  Import error: {e}")
        print("  Make sure you're in the project root directory")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  # Run an experiment:")
    print("  !python scripts/run_experiment.py --experiment unet_flair --config full")
    print("")
    print("  # Resume a paused experiment:")
    print("  !python scripts/run_experiment.py --experiment unet_flair --config full --resume")
    print("")
    print("  # Copy results to Drive for transfer:")
    print(f"  !cp -r outputs/results/ {args.data_dir}/../results/")


if __name__ == "__main__":
    main()
