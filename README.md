# Brain Tumor MRI Platform – Project README

A web-based, privacy-preserving decision-support tool for automated brain tumor detection and segmentation on MRI using state-of-the-art deep learning, with an accessible interface built on React and Next.js, and a containerized backend for clinical deployment. 
***

## Table of Contents

1. Project Overview  
2. System Architecture  
3. Step 0 – Prerequisites (Non‑coders’ On‑Ramp)  
4. Step 1 – Domain & Model Research  
5. Step 2 – Lightweight Baseline Training & Model Selection  
6. Step 3 – Full Training, Tuning, and Evaluation  
7. Step 4 – Deployment (API + Web App + Containerization)  
8. Decision Trees (Key Design Choices)  
9. Data Privacy: Client vs Server Inference  
10. Project Management & Team Roles  
11. Annex A – MRI & Neuroanatomy Resources  
12. Annex B – Deep Learning & PyTorch Resources  
13. Annex C – Brain Tumor Segmentation & BraTS Tutorials  
14. Annex D – Web App, Next.js, Vercel, and Security  
15. Annex E – Docker, DevOps, and MLOps Tutorials  

***

## 1. Project Overview

This project builds a full stack application that performs brain tumor detection and segmentation on MRI data and exposes it via a secure, user-friendly web interface for clinicians and researchers. Models output both voxel-wise tumor masks and probability scores for tumor presence, strictly as a decision-support aid, not a diagnostic device. [nature](https://www.nature.com/articles/s41598-025-94267-9)

### Core Goals

- Learn the MRI and neuro-oncology basics needed to interpret outputs responsibly. [radiologymasterclass.co](https://www.radiologymasterclass.co.uk/tutorials/mri/mri_scan)
- Evaluate several modern 3D segmentation architectures (e.g., 3D U-Net, nnU-Net, Swin UNETR, attention U-Net variants) on public datasets like BraTS 2023/2024. [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
- Train and tune the best-performing architecture and measure clinically meaningful metrics (Dice, Hausdorff).? [arxiv](https://arxiv.org/html/2407.08470v1)
- Deploy the model as a secure API and integrate it into a Next.js UI, with options for local and cloud inference while preserving patient privacy. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deploying-pytorch-models-with-torchserve/)

***

## 2. System Architecture

At a high level, the system has four layers:

- **Data & Domain Layer** – MRI brain volumes, tumor labels, neuroanatomy knowledge, MRI physics. [case](https://case.edu/med/neurology/NR/MRI%20Basics.htm)
- **Model Layer** – 3D CNN / U-Net–family segmentation models trained on BraTS-style data. [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- **Serving Layer** – PyTorch-based inference API (TorchServe / FastAPI) running in Docker, or equivalent. [serverless](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)
- **UI Layer** – Next.js/React app that lets users upload images, run inference, and visualize 3D predictions, with clear disclaimers and security controls. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

***

## 3. Step 0 – Prerequisites (Non‑coders’ On‑Ramp)

Your teammates can learn what they need if they follow a structured path.

### 0.1 Conceptual Prereqs

Action points:

- Learn MRI basics: what an MRI is, T1/T2/FLAIR, voxel, slice, sequence. [youtube](https://www.youtube.com/watch?v=5rjIMQqPukk)
- Learn what a brain tumor is anatomically and at a high level (glioma, meningioma, metastasis – only to understand labels, not to self-diagnose).? [arxiv](https://arxiv.org/html/2510.25058v1)
- Understand what segmentation is vs classification (mask vs single label).? [nature](https://www.nature.com/articles/s41598-025-94267-9)

Key resources (Annex A/B/C contain more):

- “MRI interpretation – Introduction” (Radiology Masterclass).? [radiologymasterclass.co](https://www.radiologymasterclass.co.uk/tutorials/mri/mri_scan)
- “MRI Basics” (Case Western).? [case](https://case.edu/med/neurology/NR/MRI%20Basics.htm)
- Intro MRI physics and interpretation YouTube series. [youtube](https://www.youtube.com/watch?v=5rjIMQqPukk)
- BraTS challenge description: multi-modal 3D MRI, tumor labels, use cases. [arxiv](https://arxiv.org/html/2510.25058v1)

### 0.2 Technical Prereqs

Action points (for absolute beginners):

- Learn basic Python syntax (variables, loops, functions, lists, dicts).  
- Learn basic PyTorch tensors and training loops (MNIST or CIFAR first).? [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- Learn command line basics (bash, virtualenv/conda, Python envs).  
- Learn basic Git/GitHub usage (clone, branch, commit, push).  
- Learn JavaScript + React fundamentals (components, props, hooks).  
- Learn minimal Next.js: file-based routing, API routes, environment variables. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

Curated resources are listed in Annex B & D.

***

## 4. Step 1 – Domain & Model Research

Goal: Pick a short list of candidate architectures and understand MRI and dataset characteristics well enough to design a realistic experiment plan. [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)

### 4.1 Understand the Data & Task

You will likely use BraTS-style data (multi-modal 3D brain MRI with tumor masks).? [learnopencv](https://learnopencv.com/3d-u-net-brats/)

Action points:

- Read the BraTS challenge overview to understand modalities (T1, T1ce, T2, FLAIR), voxel sizes, labels (whole tumor, tumor core, enhancing tumor). [arxiv](https://arxiv.org/html/2510.25058v1)
- Note that you are doing **3D** volumetric segmentation; your models must accept volumes (C×D×H×W) not just 2D slices. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)
- Decide: will you start with single modality (e.g., FLAIR only) or full multi-modal input? (Simpler vs better performance).? [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)

### 4.2 Choose Candidate Model Families (2025–2026 State of the Art)

From recent reviews and BraTS results, the following are strong candidates:

- **3D U-Net / advanced 3D U-Net variants** (baseline, well understood).? [arxiv](https://arxiv.org/html/2407.08470v1)
- **nnU-Net**: self-configuring 3D U-Net system that achieved top BraTS rankings; strong baseline for medical segmentation. [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
- **Swin UNETR and UNETR (transformer-based)**: strong 3D medical segmentation performance with MONAI support. [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- **Multi-scale attention U-Net with EfficientNet encoder**: modern architecture that improves tumor boundary delineation. [nature](https://www.nature.com/articles/s41598-025-94267-9)
- **U-Net with VGG19 backbone + Focal Tversky loss**: recent transfer-learning variant performing well on brain tumors. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12312496/)

You likely cannot implement all of these end-to-end, but you can:

- Use **3D U-Net** and/or **nnU-Net** as baselines (PyTorch or MONAI implementations). [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- Add one **transformer-based model (Swin UNETR)** via MONAI as an advanced candidate. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)
- Optionally test **one attention variant** if time allows. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12312496/)

### 4.3 Model Research Action Checklist

- Read a recent brain tumor segmentation review (focus on 3D U-Net and nnU-Net sections).? [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
- Skim a top BraTS 2023 solution (e.g., Auto3DSeg-based, transformer ensemble) to understand what the leaders do, even if you don’t replicate it. [arxiv](https://arxiv.org/html/2403.09262v1)
- Decide your **shortlist** (example):

  - Baseline: Simple 3D U-Net.? [arxiv](https://arxiv.org/html/2407.08470v1)
  - Auto-configured: nnU-Net (if you can handle the framework).? [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
  - Advanced: Swin UNETR (via MONAI tutorial).? [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

- Decide metrics to track: Dice coefficient per class, Hausdorff distance (optional), inference speed, VRAM usage. [arxiv](https://arxiv.org/html/2407.08470v1)

Resources (see Annex C):

- 3D U-Net for BraTS training tutorial with PyTorch. [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- MONAI Swin UNETR BraTS notebook. [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- 3D medical image segmentation with transformers tutorial (UNETR, BraTS).? [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)

***

## 5. Step 2 – Lightweight Baseline Training & Model Selection

Goal: Train each candidate model briefly with fixed, modest hyperparameters on a subset of data to see which one is most promising.? [learnopencv](https://learnopencv.com/3d-u-net-brats/)

### 5.1 Data Pipeline Setup

Action points:

- Set up a Python environment with PyTorch, MONAI (if used), nibabel, and segmentation libraries. [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- Implement data loading for NIfTI volumes (.nii or .nii.gz), including:

  - Normalization (z-score per modality).  
  - Resampling to common voxel spacing, e.g., 1×1×1 mm (or rely on MONAI transforms). [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)
  - Cropping/padding to fixed patch size (e.g., 128×128×128 or similar). [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)

- Implement train/val split (e.g., 80/20 or 5-fold; for lightweight testing you can use a single split).? [arxiv](https://arxiv.org/html/2510.25058v1)

Use reference code:

- MONAI dictionary transforms pipeline for BraTS (spacing, orientation, crop, normalize, ToTensor).? [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- LearnOpenCV’s 3D U-Net BraTS tutorial for NIfTI reading and conversion to tensors. [learnopencv](https://learnopencv.com/3d-u-net-brats/)

### 5.2 Lightweight Training Setup

For each candidate model:

- Fix a simple optimizer (Adam), learning rate (e.g., 1e-4 or 2e-4), batch size (1–2 depending on VRAM).? [arxiv](https://arxiv.org/html/2407.08470v1)
- Train for a small, fixed number of epochs (e.g., 10–20) on a **reduced** training set (e.g., 20–30 cases).  
- Use a standard segmentation loss (Dice loss or Dice+CrossEntropy) initially. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12312496/)

Action points:

- Implement a single training script with a switch for model choice, e.g., `--model 3d_unet` vs `--model swin_unetr`.  
- Log training/validation Dice per class; save checkpoints with best validation Dice.  
- Record GPU memory usage and training time per epoch for each model.

Resources:

- 3D U-Net BraTS training script (reference for losses, logging, and patch handling). [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- MONAI Swin UNETR BraTS notebook (shows complete training pipeline).? [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

### 5.3 Model Selection Criteria

After the lightweight runs:

Evaluate each model on:

- Validation Dice for whole tumor and tumor core. [arxiv](https://arxiv.org/html/2407.08470v1)
- Stability of training curve (does it converge cleanly?).? [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- VRAM footprint and speed (practical for deployment?).  

Decision rule sketch:

- If a model is significantly better (e.g., >0.1 Dice improvement) and computationally feasible, pick it.  
- If performance is similar but one model is simpler and faster (e.g., plain 3D U-Net vs Swin UNETR), pick the simpler one.  
- If nnU-Net is used and auto-configuration yields strong results with minimal tuning, it’s a very strong candidate. [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)

***

## 6. Step 3 – Full Training, Tuning, and Evaluation

Goal: Take the selected model and train it more seriously with hyperparameter tuning, more data, and robust evaluation.? [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)

### 6.1 Hyperparameter Tuning (Limited but Targeted)

Action points:

- Tune learning rate using a small LR sweep (e.g., 1e-5, 3e-5, 1e-4, 3e-4).  
- Try different loss functions:

  - Dice + CrossEntropy.  
  - Focal Tversky loss for class imbalance and small structures (motivated by VGG19 U-Net paper).? [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12312496/)

- Tune patch size and batch size according to GPU limits.? [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- Use data augmentation: random flips, rotations, intensity shifts, possibly elastic deformations. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)

You can use a simple grid search or manual tuning; full AutoML is optional.

### 6.2 Full Training

Action points:

- Train on the full training set (or as much as you can) with k-fold cross-validation if feasible. [arxiv](https://arxiv.org/html/2403.09262v1)
- Regularly evaluate on a held-out validation set and implement early stopping.  
- Save best checkpoints with full training metadata (hyperparameters, data preprocessing settings, code commit hash).

### 6.3 Model Evaluation & Reporting

Action points:

- Compute metrics by region: Dice for whole tumor, core, and enhancing tumor if your labels support that, or a simpler binary tumor/no-tumor mask. [arxiv](https://arxiv.org/html/2510.25058v1)
- Compute per-case statistics (mean, std of Dice) and visualize a histogram of performance.  
- Do qualitative evaluation:

  - Visualize overlays of MRI slices with predicted masks vs ground truth.  
  - Discuss typical failure modes (e.g., small lesions missed, false positives on artifacts).

- Write a **Model Card**:

  - Training data description (public BraTS-style, non-clinical).? [arxiv](https://arxiv.org/html/2510.25058v1)
  - Intended use (education and research, decision-support only).  
  - Limitations and non-intended use (no real-time clinical deployment without validation, no non-Brain MRI, no self-diagnosis).  
  - Ethical and bias considerations.

Resources:

- BraTS challenge papers and results for target metrics and typical reporting format. [arxiv](https://arxiv.org/html/2403.09262v1)
- LearnOpenCV BraTS 3D U-Net evaluation examples (loading best checkpoint and running inference).? [learnopencv](https://learnopencv.com/3d-u-net-brats/)

***

## 7. Step 4 – Deployment (API + Web App + Containerization)

Goal: Wrap the trained model in an API, integrate it into a secure Next.js UI, and package everything in Docker for reproducible deployment on local or clinical infrastructure. [youtube](https://www.youtube.com/watch?v=6Lu4vyYRTEo)

### 7.1 Model Serving on the Backend

You have three realistic options:

1. **TorchServe (PyTorch-native)** – good for production inference. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deploying-pytorch-models-with-torchserve/)
2. **FastAPI / Flask + PyTorch** – easier to understand for beginners.  
3. **MONAI bundle + MONAI Deploy** – if you go deep into the MONAI ecosystem (optional).? [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

Given your team’s learning goals, FastAPI or TorchServe are reasonable.

Action points (FastAPI-style):

- Write a Python service that:

  - Loads the trained PyTorch model at startup.  
  - Exposes endpoints:

    - `/infer` – accepts MRI volume upload (or path reference), runs preprocessing and inference, returns mask and probability scores.  
    - `/health` – simple health check.

  - Handles NIfTI file parsing (using nibabel) and wraps outputs (e.g., as NIfTI or compressed NumPy).

- Implement basic input validation:

  - File type check.  
  - Size limits.  
  - Sanitize or strip metadata.

Action points (TorchServe-style, if chosen):

- Follow a PyTorch/TorchServe deployment tutorial:

  - Export model to `.pt`.  
  - Write a custom `handler.py` to preprocess uploaded MRI volumes and post-process predictions.  
  - Archive into `.mar` and launch TorchServe to expose HTTP endpoints. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deploying-pytorch-models-with-torchserve/)

Resources:

- “Deploying PyTorch Models with TorchServe” step-by-step guide. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deploying-pytorch-models-with-torchserve/)
- Serverless deployment of PyTorch model with Docker and AWS Lambda (for concepts and patterns). [serverless](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)

### 7.2 Data Privacy & Security – Client vs Server Inference

See Section 9 for a dedicated decision tree; here, link architecture to choices.

If inference is **local (preferred for privacy)**:

- Run the Dockerized API on a clinician’s workstation or hospital server behind their firewall; MRI files never leave the local network.  
- Next.js app calls the local API endpoint (e.g., `https://hospital-local-api:8080/infer`), with HTTPS and local auth controls. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

If inference is **cloud-based (e.g., Vercel frontend + cloud API)**:

- Use HTTPS everywhere, strict authentication, and encryption of data in transit and, if stored, at rest. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
- De-identify MRI data before upload (no DICOM PHI, no names, IDs, dates). [arxiv](https://arxiv.org/html/2510.25058v1)
- Avoid persistent storage; process in-memory and discard after inference.  
- Use access control (JWT, OAuth, or institutional SSO) and rate limiting.  
- Use environment variables and server-side route handlers in Next.js to protect secrets. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

Resources:

- Securing a Next.js AI application deployed on Vercel: environment variables, route handlers, WAF, attack challenge mode. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
- Building a secure healthcare app with Next.js and Supabase (RLS, data encryption).? [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
- Next.js healthcare platform security discussion (static delivery, middleware, serverless functions).? [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

### 7.3 Web Interface (Next.js + React)

Action points:

- Build a Next.js app with:

  - Authenticated dashboard (if needed).  
  - File upload page for MRI volumes or zipped files.  
  - Progress indicators and error handling during inference.  
  - Visualization:

    - 2D slice viewer with mask overlay.  
    - Basic 3D viewer using a JS visualization library (optional but powerful).

- Make clear **disclaimers**:

  - This tool is for research and educational purposes, and is a decision-support aid only.  
  - It does not provide medical diagnosis or treatment recommendations.  
  - Clinical decisions remain with licensed healthcare professionals.

- Implement security best practices:

  - Use Next.js API routes or Route Handlers to proxy calls to backend APIs so keys are not exposed to the browser. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
  - Use middleware for authentication and header hardening (e.g., CSP, secure cookies).? [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)
  - Store secrets only in Vercel or server environment variables, never in client code. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

Resources:

- “How to Secure a Next.js AI Application Deployed on Vercel” – excellent walkthrough of common pitfalls and fixes. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
- “Building a Healthcare App with Next.js and Supabase” – shows auth, security considerations, and AI integration patterns. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
- “Building Healthcare Platforms with Next.js” – discusses security, middleware, and serverless patterns. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

### 7.4 Containerization & Deployment

Action points:

- Write a **Dockerfile** for the backend (FastAPI/TorchServe + model + dependencies). [youtube](https://www.youtube.com/watch?v=6Lu4vyYRTEo)
- Optionally write a separate Dockerfile for the Next.js frontend (if not using Vercel) or deploy frontend serverlessly and backend to a container platform.  
- Use `docker-compose` or Kubernetes (advanced) to orchestrate components.

Typical local deployment:

- `docker-compose up` starts:

  - `backend` (FastAPI or TorchServe on port 8080).  
  - `frontend` (Next.js or static assets served by nginx).  

- User accesses the UI in a browser pointing to a local address.

Resources:

- Quick YouTube guide on setting up PyTorch in Docker (for environment basics).? [youtube](https://www.youtube.com/watch?v=6Lu4vyYRTEo)
- “Deploying PyTorch Model as a Serverless Service” – covers Docker image building & pushing to cloud runtimes. [serverless](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)

If you deploy the frontend on Vercel:

- Connect GitHub repo to Vercel and configure build settings. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
- Set environment variables (API URL, auth secrets) in Vercel dashboard. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
- Use Vercel security features (WAF rules, attack challenge mode, spend management) if app becomes public. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

***

## 8. Decision Trees (Key Design Choices)

Below are simplified decision trees guiding major choices.

### 8.1 Model Architecture Choice

1. Do you want maximum automation and proven BraTS performance, at the cost of framework complexity?  
   - Yes → Use **nnU-Net** (if you can handle its config and training workflow). [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
   - No → Go to 2.

2. Do you have decent GPU memory and want SOTA-ish performance using modern transformers?  
   - Yes → Use **Swin UNETR** via MONAI and follow the BraTS tutorial. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)
   - No → Go to 3.

3. Do you want a simple, robust baseline with many tutorials and code examples?  
   - Yes → Use **3D U-Net** (possibly with small enhancements).? [arxiv](https://arxiv.org/html/2407.08470v1)
   - No → Consider an **attention U-Net** or transfer-learning variant (EfficientNet, VGG19) but only after you have a 3D U-Net baseline.? [nature](https://www.nature.com/articles/s41598-025-94267-9)

### 8.2 Inference: Client vs Server

1. Are you deploying inside a hospital or research network with local machines and strict privacy rules?  
   - Yes → Prefer **local Docker deployment**; run API and model on local server/workstation. MRI never leaves the network.  
   - No → Go to 2.

2. Are you ok with cloud deployment and can you de-identify data and implement strong security?  
   - Yes → Use Vercel for frontend + cloud backend (container or serverless); implement HTTPS, auth, de-identification. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
   - No → Keep usage to **local install only** (e.g., for research labs), and distribute Docker images / offline installers.

### 8.3 Data Input Format

1. Are you working directly with DICOM containing PHI?  
   - Yes → Add a pre-processing pipeline that converts to NIfTI and removes metadata before anything touches your model or leaves the local network. [arxiv](https://arxiv.org/html/2510.25058v1)
   - No → If you already have anonymized NIfTI volumes, keep that pipeline and document de-identification.

***

## 9. Data Privacy: Procedures for Client vs Server Inference

### 9.1 Common Privacy Practices (Both Modes)

Action points:

- Work only with publicly available, **fully anonymized** datasets for development (BraTS, etc.).? [arxiv](https://arxiv.org/html/2510.25058v1)
- If ever handling real patient data:

  - Implement and document de-identification by design (tools specific to DICOM are outside scope here but must be used).  
  - Avoid storing raw images after inference unless necessary; if stored, encrypt at rest and control access. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)

- Maintain audit logs of access and inferences on clinical deployments (institution-specific).  
- Display clear privacy notices and disclaimers in the UI.

### 9.2 Client-Side / Local Inference

Procedure:

- Install the Dockerized application on a local machine or hospital server under their IT governance. [serverless](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)
- Ensure:

  - Only local network addresses are used.  
  - HTTPS is enabled via local certificates or reverse proxy.  
  - OS-level user permissions restrict who can access local UI.  

- MRI data is loaded from local PACS / file system and never transmitted externally.  
- Documentation clearly states that data remains within the institution’s network, with no telemetry or cloud logging.

### 9.3 Server-Side / Cloud Inference

Procedure:

- De-identification step before upload (anonymous NIfTI only).? [arxiv](https://arxiv.org/html/2510.25058v1)
- Data in transit: enforce HTTPS for all connections (frontend ↔ backend). [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
- Data at rest: either avoid retention entirely or store encrypted with strict access control and short retention windows. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)
- Access control:

  - Implement user accounts and authentication (e.g., NextAuth, Supabase auth, or hospital SSO). [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
  - Use server-side Next.js API routes or middleware to guard sensitive endpoints. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

- Secrets and configuration:

  - Keep API keys and model endpoints in environment variables (Vercel dashboard or server env), never client-side. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

- Security hardening:

  - Apply Next.js middleware for request validation and rate limiting. [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)
  - Configure Vercel WAF rules and attack challenge mode if exposed widely. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)

- Documentation must state:

  - Where data is processed (region, provider).  
  - What is logged and for how long.  
  - That this is **not** a clinical-grade, certified medical device.

***

## 10. Project Management & Team Roles

Suggested roles:

- **Domain Lead** – learns MRI basics and tumor pathology, curates examples for the team. [radiologymasterclass.co](https://www.radiologymasterclass.co.uk/tutorials/mri/mri_scan)
- **ML Lead** – focuses on model architecture, training scripts, and hyperparameter tuning. [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- **Infra Lead** – focuses on Docker, backend serving, and deployment. [youtube](https://www.youtube.com/watch?v=6Lu4vyYRTEo)
- **Frontend Lead** – builds Next.js UI and integrates with API, handles UX and security middleware. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)

Working style:

- Define phases (Research → Baselines → Selection → Full training → Deployment).  
- Use a shared Kanban board (GitHub Projects) with specific, small tasks so non-coders can pick up well-scoped items.
- Distributed Training: Rely on local network sharing or Chrome Remote Desktop files transfer to distribute pre-extracted data to team PCs (GTX 1080, RTX 4060, and GTX 1660 Super), rather than everyone downloading independently.

***

## 11. Annex A – MRI & Neuroanatomy Resources

- “MRI interpretation – Introduction” (Radiology Masterclass): Basics of MRI physics, sequences, and interpretation. [radiologymasterclass.co](https://www.radiologymasterclass.co.uk/tutorials/mri/mri_scan)
- “MRI Basics” (Case Western Neurology): MRI physics explanation, how signals become images. [case](https://case.edu/med/neurology/NR/MRI%20Basics.htm)
- “MRI Made Easy” PDF – introductory booklet explaining steps of an MRI exam and basic concepts. [rads.web.unc](https://rads.web.unc.edu/wp-content/uploads/sites/12234/2018/05/Phy-MRI-Made-Easy.pdf)
- YouTube MRI basics series: signal, T1/T2, pulse sequences, typical protocols. [youtube](https://www.youtube.com/watch?v=5rjIMQqPukk)

***

## 12. Annex B – Deep Learning & PyTorch Resources

Even though not all are MRI-specific, they provide the foundation your non-coder teammates need.

- Learn PyTorch & tensors, optimizers, and training loops via standard beginner tutorials (e.g., classification tasks).? [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- 3D U-Net code examples show using PyTorch for volumetric segmentation, including dataloaders, losses, and checkpointing. [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- 3D medical segmentation with transformers (UNETR) tutorial demonstrates advanced architectures in PyTorch with MONAI. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)

***

## 13. Annex C – Brain Tumor Segmentation & BraTS Tutorials

- BraTS challenge pages and papers – dataset description and evaluation framework for multi-modal 3D MRI brain tumor segmentation. [arxiv](https://arxiv.org/html/2510.25058v1)
- “Training 3D U-Net for Brain Tumor Segmentation Challenge” – end-to-end PyTorch tutorial for BraTS 3D U-Net, including preprocessing and inference. [learnopencv](https://learnopencv.com/3d-u-net-brats/)
- “3D Medical image segmentation with transformers” – tutorial using UNETR for 3D medical contours with MONAI. [theaisummer](https://theaisummer.com/medical-segmentation-transformers/)
- MONAI Swin UNETR BraTS21 3D segmentation notebook – full pipeline using transformer-based 3D U-Net. [github](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)
- Comprehensive review of brain tumor segmentation methods with emphasis on 3D U-Net and nnU-Net. [scirp](https://www.scirp.org/journal/paperinformation?paperid=140886)
- Papers proposing improved U-Net variants (multi-scale attention with EfficientNet encoder; VGG19 encoder + Focal Tversky loss) for more accurate segmentation. [nature](https://www.nature.com/articles/s41598-025-94267-9)

***

## 14. Annex D – Web App, Next.js, Vercel, and Security

- “How to Secure a Next.js AI Application Deployed on Vercel” – covers route handlers, env vars, Vercel security features, and how to lock down endpoints. [freecodecamp](https://www.freecodecamp.org/news/how-to-secure-a-nextjs-ai-application-deployed-on-vercel/)
- “Building a Healthcare App with Next.js and Supabase” – demonstrates auth, row-level security, encryption, and AI integration patterns relevant to health data. [vaibhav-parmar-portfolio.vercel](https://vaibhav-parmar-portfolio.vercel.app/blog/building-healthcare-app-with-nextjs-supabase)
- “Building Healthcare Platforms with Next.js” – explains why Next.js aligns well with healthcare security needs (static delivery, middleware, serverless functions). [pagepro](https://pagepro.co/blog/building-healthcare-platforms-with-next-js/)

***

## 15. Annex E – Docker, DevOps, and MLOps Tutorials

- “Setting Up PyTorch in Docker in 5 Minutes!” – quick intro to building a PyTorch-ready Docker container. [youtube](https://www.youtube.com/watch?v=6Lu4vyYRTEo)
- “Deploying PyTorch Models with TorchServe” – official-style guide, including Docker usage and HTTP inference endpoints. [geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deploying-pytorch-models-with-torchserve/)
- “Deploying PyTorch Model as a Serverless Service” – shows building Docker images and deploying them to serverless runtimes like AWS Lambda. [serverless](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)

This README is intended to be your project’s living **roadmap**; as you implement each step, update it with your actual scripts, dataset versions, and results so future collaborators (even non-coders) can reproduce the entire pipeline.


