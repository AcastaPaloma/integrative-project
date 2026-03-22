# Brain Tumor Segmentation — Web Application Specification

> **Purpose of this document:** Complete implementation specification for a locally-deployable,
> full-stack web application that serves trained 3D brain tumor segmentation models.
> This document is written for a capable LLM or developer building the system from scratch.
> UI/visual design decisions are intentionally left out — the implementor receives design direction separately.
> Focus here is on architecture, UX flows, data contracts, and every functional requirement.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Filesystem Layout](#4-filesystem-layout)
5. [Pages & UX Flows](#5-pages--ux-flows)
6. [Component Inventory](#6-component-inventory)
7. [Backend API Specification](#7-backend-api-specification)
8. [Background Job System](#8-background-job-system)
9. [3D Viewer Requirements](#9-3d-viewer-requirements)
10. [Export System](#10-export-system)
11. [Hardware Auto-Detection](#11-hardware-auto-detection)
12. [Dark / Light Mode](#12-dark--light-mode)
13. [Containerization & Deployment](#13-containerization--deployment)
14. [ML Model Integration](#14-ml-model-integration)
15. [Error Handling & Edge Cases](#15-error-handling--edge-cases)
16. [Future Extensibility Notes](#16-future-extensibility-notes)

---

## 1. Project Overview

### What this is

A locally-deployed web application that lets a physician (or researcher) upload 1–4 NIfTI
brain MRI modality files (T1, T1ce, T2, FLAIR) for a single patient, select a trained
segmentation model, run inference, and interactively visualize the resulting 3D tumor masks.
Results are persisted to local disk and can be exported.

### What this is NOT

- Not a cloud-hosted SaaS
- Not a diagnostic device or clinical tool (must display a prominent disclaimer on every results page)
- Not multi-tenant (no login/accounts required; designed for a single user or small trusted team on one machine)

### Core user journey

```
Upload MRI files → name/tag the case → pick a model → run inference
  → wait (or leave and come back) → view 3D results → export
```

---

## 2. Tech Stack

### Frontend

| Concern | Choice | Rationale |
|---|---|---|
| Framework | **Next.js 14+ (App Router)** | File-based routing, API routes, SSE support, easy local dev |
| Language | **TypeScript** | Type safety across the whole frontend |
| State management | **Zustand** | Lightweight, no boilerplate, easy async |
| 3D / NIfTI viewer | **NiiVue** (`@niivue/niivue`) | WebGL-based, purpose-built for NIfTI, handles MPR + volume render natively |
| HTTP client | **fetch / SWR** | SWR for polling/revalidation of job status |
| Notifications/toasts | **Sonner** | Minimal, accessible |
| File upload | Native HTML5 drag-and-drop + `<input>` | No external dep needed |

### Backend

| Concern | Choice | Rationale |
|---|---|---|
| Framework | **FastAPI** | Python-native, async, WebSocket/SSE built in, same ecosystem as PyTorch |
| Language | **Python 3.11** | Matches training environment |
| Database | **SQLite via `aiosqlite`** | Zero external dependency, file-based, persistent |
| ORM/query | **raw SQL or `databases` package** | Keep it simple; no full ORM needed |
| Inference runtime | **Subprocess / ProcessPoolExecutor** | Isolate GPU work from FastAPI event loop |
| Log streaming | **Server-Sent Events (SSE)** | Stateless reconnection; simpler than WebSocket for one-way log stream |
| File serving | **FastAPI `FileResponse` / `StaticFiles`** | Serve NIfTI files directly to NiiVue |

### Shared

| Concern | Choice |
|---|---|
| Containerization | **Docker Compose** (2 services: `frontend`, `backend`) |
| Dev mode | `npm run dev` (frontend) + `uvicorn` (backend) — both launachable with a single `start.sh` |
| Inter-service communication | HTTP only (`http://localhost:8000`) — no message broker needed |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                           │
│                                                                 │
│  Next.js App (port 3000)                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Pages/UI    │  │  NiiVue      │  │  SSE log stream    │    │
│  │  (React)     │  │  3D viewer   │  │  (EventSource)     │    │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘    │
│         │  REST + SSE                                           │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│                    FastAPI Backend (port 8000)                   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  REST API   │  │  SSE router  │  │  File serving         │  │
│  │  /cases/*   │  │  /jobs/*/log │  │  /files/{case_id}/... │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Job Manager                            │  │
│  │  SQLite job table  │  In-process FIFO queue               │  │
│  │  (persisted state) │  (one GPU job at a time)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────────┐  │
│  │             Inference Worker (separate process)           │  │
│  │                                                           │  │
│  │  Loads model checkpoint → runs MONAI sliding window →    │  │
│  │  writes prediction.nii.gz → streams log lines to SQLite  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  SQLite DB: cases + jobs + logs                                 │
│  Local FS:  data/cases/{case_id}/...                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key architectural decisions

**Why a separate process for inference?**
FastAPI runs on an asyncio event loop. PyTorch inference (especially with MONAI sliding window)
is CPU/GPU-bound and will block the event loop if run directly. Using `multiprocessing` or
`subprocess` isolates it completely — the FastAPI server stays responsive, the log stream
keeps working, and there are no GIL or CUDA context conflicts.

**Why SSE instead of WebSocket for logs?**
SSE is simpler to implement, stateless (auto-reconnects on network hiccup), and perfectly
suited for one-directional server→client streaming. If a user leaves the page and comes back,
`EventSource` reconnects and the backend replays logs from the last received ID.

**Why SQLite?**
Zero infrastructure. One file. Survives Docker restarts with a mounted volume. Simple enough
to inspect with any SQLite browser for debugging.

---

## 4. Filesystem Layout

### Repository structure (new `webapp/` directory added to existing repo)

```
integrative-project/
├── webapp/
│   ├── frontend/                    # Next.js application
│   │   ├── app/
│   │   │   ├── layout.tsx           # Root layout (theme provider, nav)
│   │   │   ├── page.tsx             # Dashboard
│   │   │   ├── cases/
│   │   │   │   ├── page.tsx         # Case library
│   │   │   │   ├── new/
│   │   │   │   │   └── page.tsx     # New case upload
│   │   │   │   └── [caseId]/
│   │   │   │       ├── page.tsx     # Case detail / viewer
│   │   │   │       └── export/
│   │   │   │           └── page.tsx # Export options
│   │   │   └── settings/
│   │   │       └── page.tsx         # App settings
│   │   ├── components/
│   │   │   ├── viewer/              # All NiiVue-related components
│   │   │   ├── upload/              # Drag-and-drop upload components
│   │   │   ├── jobs/                # Job status, log stream, progress
│   │   │   ├── cases/               # Case cards, list items, metadata form
│   │   │   └── ui/                  # Generic UI primitives
│   │   ├── lib/
│   │   │   ├── api.ts               # Typed fetch wrappers for all endpoints
│   │   │   ├── store.ts             # Zustand global state
│   │   │   └── types.ts             # Shared TypeScript types
│   │   ├── hooks/
│   │   │   ├── useCase.ts
│   │   │   ├── useJobStream.ts      # SSE connection hook
│   │   │   └── useModels.ts
│   │   ├── public/
│   │   ├── next.config.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── backend/                     # FastAPI application
│   │   ├── main.py                  # App entry point
│   │   ├── routers/
│   │   │   ├── cases.py             # Case CRUD
│   │   │   ├── jobs.py              # Job management + SSE log stream
│   │   │   ├── models.py            # Available model listing
│   │   │   ├── files.py             # NIfTI file serving
│   │   │   └── system.py            # Hardware info, health check
│   │   ├── services/
│   │   │   ├── job_manager.py       # Queue + state machine
│   │   │   ├── inference_worker.py  # Subprocess entry point (runs MONAI)
│   │   │   ├── file_manager.py      # Case directory operations
│   │   │   └── model_registry.py    # Discovers available checkpoints
│   │   ├── db/
│   │   │   ├── database.py          # aiosqlite connection + migrations
│   │   │   └── schema.sql           # DDL for cases, jobs, log_lines tables
│   │   ├── config.py                # Settings (paths, hardware, etc.)
│   │   └── requirements.txt
│   │
│   ├── data/                        # Runtime data (mounted as Docker volume)
│   │   ├── cases/                   # One directory per case
│   │   │   └── {case-uuid}/
│   │   │       ├── metadata.json    # Case name, notes, created_at, status
│   │   │       ├── inputs/
│   │   │       │   ├── flair.nii.gz (optional)
│   │   │       │   ├── t1.nii.gz    (optional)
│   │   │       │   ├── t1ce.nii.gz  (optional)
│   │   │       │   └── t2.nii.gz    (optional)
│   │   │       └── outputs/
│   │   │           ├── prediction_labels.nii.gz   # Integer label map (0,1,2,4)
│   │   │           ├── mask_wt.nii.gz             # Binary whole tumor
│   │   │           ├── mask_tc.nii.gz             # Binary tumor core
│   │   │           ├── mask_et.nii.gz             # Binary enhancing tumor
│   │   │           └── inference_log.txt          # Complete terminal output
│   │   └── app.db                   # SQLite database
│   │
│   ├── docker-compose.yml
│   ├── start.sh                     # One-command local dev launcher
│   └── README.md                    # User-facing setup guide
│
├── checkpoints/                     # Existing — model weights
├── src/                             # Existing — training code
└── ...                              # All other existing files
```

### Case directory: `data/cases/{uuid}/`

`metadata.json` schema:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "label": "Patient 42 — Pre-op",
  "notes": "Glioblastoma suspected. Referred by Dr. Smith.",
  "created_at": "2025-03-21T14:30:00Z",
  "modalities_uploaded": ["flair", "t1ce"],
  "model_used": "unet_flair_t1ce",
  "status": "completed",
  "job_id": "job-uuid",
  "inference_duration_seconds": 142
}
```

`status` values: `"uploading"` → `"ready"` → `"queued"` → `"running"` → `"completed"` | `"failed"`

---

## 5. Pages & UX Flows

### Page map

```
/                      Dashboard
/cases                 Case Library
/cases/new             New Case — Upload & Configure
/cases/[id]            Case Detail — Viewer & Results
/cases/[id]/export     Export Options
/settings              Application Settings
```

---

### Page 1: Dashboard (`/`)

**Purpose:** At-a-glance overview of the system and recent work.

**Sections:**

**A. System status bar** (top, always visible on dashboard)
- GPU detected: `NVIDIA GTX 1080 — CUDA 12.1` or `CPU only (no GPU detected)`
- Active job indicator: if a job is running, show its case label + elapsed time + a "View" button
- Disk usage: `data/ — 12.4 GB used`

**B. Quick stats row** — 4 stat cards:
- Total cases
- Cases with completed inference
- Cases pending / running
- Total inference runs performed

**C. Recent cases** — last 5 cases, each card showing:
- Case label
- Modalities uploaded (icon set: FLAIR, T1, T1ce, T2 — greyed out if not uploaded)
- Model used
- Status badge (`ready`, `running`, `completed`, `failed`)
- Elapsed time since creation
- Click → navigates to `/cases/[id]`

**D. Quick action** — prominent button: **"New Case"** → `/cases/new`

---

### Page 2: Case Library (`/cases`)

**Purpose:** Browse, search, and manage all saved cases.

**Layout:**

**A. Toolbar**
- Search input (filters by case label, notes — client-side)
- Filter dropdown: All | Completed | Running | Ready | Failed
- Sort: Newest first | Oldest first | Alphabetical
- View toggle: Grid | List
- Button: **"New Case"**

**B. Case grid/list**
Each case entry shows:
- Case label (editable in-place on double click)
- Modality icons (FLAIR / T1 / T1ce / T2) — coloured if uploaded, greyed if not
- Status badge
- Model name used
- Date created
- Actions (on hover or via `...` menu):
  - View results
  - Rename
  - Delete (with confirmation dialog — "This will permanently delete all files for this case.")
  - Export
  - Duplicate (creates a new `ready` case with same uploads, no outputs)

**C. Empty state**
If no cases: illustration + "No cases yet. Upload your first patient." + "New Case" button.

---

### Page 3: New Case (`/cases/new`)

**Purpose:** Upload NIfTI files, label the case, pick a model, start inference.

**This is a multi-step flow with clear visual step indicators.**

---

#### Step 1: Upload Files

**Upload zone behaviour:**
- Large drag-and-drop area accepting `.nii` and `.nii.gz` files only
- User can drop 1–4 files simultaneously, or click to open file picker
- Each dropped file is auto-classified by filename keyword matching:
  - Contains `flair` → FLAIR slot
  - Contains `t1ce` or `t1c` → T1ce slot (check BEFORE t1)
  - Contains `t1` (and not t1ce) → T1 slot
  - Contains `t2` → T2 slot
  - Unrecognized → ask user to manually assign (dropdown per file)
- Visual: 4 slots displayed (FLAIR, T1, T1ce, T2). Each slot shows:
  - Empty state: dashed border, modality name, "Drop here or click to upload"
  - Filled state: filename, file size, a "×" remove button
  - At least 1 slot must be filled to proceed
- File validation on drop:
  - Must be `.nii` or `.nii.gz` — else show inline error
  - Max file size: configurable (default 500 MB per file) — show error if exceeded
  - Files are uploaded to backend immediately on drop (chunked upload to `/api/cases/upload-temp`),
    showing a per-file progress indicator on each slot. The case UUID is reserved on first file drop.

**"Continue" button** → active only when ≥1 slot filled and all uploads complete.

---

#### Step 2: Case Details

**Fields:**
- **Case label** (required, text input) — default: `"Case YYYY-MM-DD HH:MM"`
- **Notes** (optional, textarea, max 1000 chars) — "Clinical notes, referral info, etc."
- **Tags** (optional, free-text tag input) — for filtering/grouping later (extensibility)

**"Continue" button** → active when label is filled.

---

#### Step 3: Model Selection

**Layout: model selection panel**

A list of available models, each showing:
- Model name (human-readable, from `experiments.yaml` `description` field)
- Architecture badge: `U-Net` or `CNN`
- Required modalities: icon row (e.g. "FLAIR + T1ce")
- Compatibility indicator:
  - ✅ "Compatible — you uploaded FLAIR + T1ce" (green)
  - ⚠️ "Partial match — model expects 4 channels, will zero-pad missing modalities" (amber)
  - ❌ "Incompatible — model requires modalities you didn't upload" (red, disabled)
- Checkpoint file size
- A "Recommended" badge if the model's modalities exactly match what the user uploaded

**Important:** Models that are ❌ incompatible are shown but disabled (greyed, non-selectable).
Models that are ⚠️ partial match are selectable with a clear warning tooltip.

Below the list: a collapsible "Advanced" section showing:
- Inference patch size override (default from model config)
- Sliding window overlap (default 0.5)
- These are optional; most users will never touch them.

**"Start Inference" button** → active when a model is selected.

---

#### Step 4: Inference Running (blocking modal, but dismissable)

**Behaviour:**
- On clicking "Start Inference", the case is submitted to the job queue.
- A full-screen modal (or dedicated page at `/cases/[id]`) opens automatically.
- The modal shows:

**A. Progress header**
- Case label
- Model name
- Status: `Queued (position 2 in queue)` | `Running` | `Completed` | `Failed`
- Elapsed time (live counter)
- Estimated time remaining (shown once inference starts, based on hardware benchmarks — optional/best-effort)

**B. Terminal log panel**
- Scrollable, monospace log output streamed live via SSE
- Auto-scrolls to bottom (with a "scroll to bottom" button that appears when user scrolls up)
- Log lines are colour-coded:
  - `[INFO]` → default text colour
  - `[WARN]` → amber
  - `[ERROR]` → red
- Line format preserved from existing training code: `[INFO] Epoch ...`

**C. Dismiss / background button**
- "Run in background" button — dismisses the modal, returns user to wherever they were
- A persistent, non-intrusive status indicator appears in the top navigation bar while any job is running:
  `● Case "Patient 42" — Running — 2m 14s` with a "View" link that re-opens the log panel
- This top-bar indicator is visible on ALL pages

**D. On completion:**
- Status changes to ✅ `Completed in 2m 41s`
- Button: "View Results" → navigates to `/cases/[id]`
- If user had minimized to background, a browser notification fires (if permission granted):
  `"Brain Tumor Seg — Inference complete for Patient 42"`

**E. On failure:**
- Status changes to ❌ `Failed`
- Full log is preserved and shown
- Error summary at top of log panel (last ERROR line highlighted)
- Button: "Retry" (re-queues the same job with same settings)

---

### Page 4: Case Detail / Viewer (`/cases/[id]`)

**Purpose:** The primary results page. Full-screen, rich interactive visualization.

**Layout: two-panel**

```
┌──────────────────────────────────────────────────────────┐
│  Nav bar (minimal — case label + breadcrumb)             │
├─────────────────┬────────────────────────────────────────┤
│                 │                                        │
│  LEFT SIDEBAR   │           VIEWER AREA                  │
│  (collapsible)  │                                        │
│                 │                                        │
│  Case info      │                                        │
│  Layer controls │                                        │
│  View controls  │                                        │
│  Export button  │                                        │
│                 │                                        │
└─────────────────┴────────────────────────────────────────┘
```

**If inference is not yet complete:** show status + log stream in place of viewer.

**If inference completed:** show viewer.

---

#### Left Sidebar

**A. Case info panel** (collapsible section)
- Case label (editable inline)
- Notes (editable inline, auto-save)
- Date created
- Model used
- Inference duration
- Modalities uploaded (icon badges)

**B. Layer controls panel**

The user sees the MRI + up to 3 mask overlays (WT, TC, ET):

For each layer:
- **MRI Background** (always present):
  - Modality selector: dropdown → which uploaded modality to use as background (FLAIR / T1 / T1ce / T2)
  - Opacity slider: 0–100%
  - Window/Level sliders (min intensity, max intensity) for brightness/contrast

- **Whole Tumor (WT)** overlay:
  - Visibility toggle (eye icon)
  - Colour picker (default: green)
  - Opacity slider: 0–100%

- **Tumor Core (TC)** overlay:
  - Visibility toggle
  - Colour picker (default: blue)
  - Opacity slider

- **Enhancing Tumor (ET)** overlay:
  - Visibility toggle
  - Colour picker (default: red)
  - Opacity slider

- Predefined colour presets row: "BraTS Standard" | "High Contrast" | "Custom"

**C. View controls panel**

Checkboxes/toggles for which panels to show in the viewer area:
- ☑ Axial
- ☑ Coronal
- ☑ Sagittal
- ☑ 3D Volume Render

View layout selector (radio):
- 2×2 grid (all 4 active)
- Single active panel (full width)
- 3+1 (3 slices small, 1 large)

Crosshair sync: toggle — when enabled, clicking a voxel in any slice panel moves
the crosshair in all other panels to the same anatomical point.

**D. Export button** → opens export panel or navigates to `/cases/[id]/export`

---

#### Viewer Area

**Implemented using NiiVue (`@niivue/niivue`).**

NiiVue is loaded as a React wrapper. Each panel is a NiiVue canvas.
The MRI volume is loaded as the base volume; masks are loaded as overlays.

**Slice panels (Axial / Coronal / Sagittal):**
- Slice index slider below each panel (or scroll wheel to navigate slices)
- Current slice index display: `Axial: 78 / 155`
- Click + drag to pan
- Scroll to change slice
- Double-click to enter "full panel" mode (maximises that panel, hides others)
- Right-click context menu: "Screenshot this view" (downloads a PNG of just this panel)

**3D Volume Render panel:**
- NiiVue's built-in volume render
- Click + drag to rotate
- Scroll to zoom
- Controls strip below the panel:
  - Render mode: `MIP (Maximum Intensity Projection)` | `Volume` | `Isosurface`
  - Clip plane toggle + slider (clip the volume to reveal interior)
- Masks rendered as coloured semi-transparent surfaces overlaid on MRI

**Global viewer controls (top bar of viewer area):**
- Reset view button (resets all panels to default orientation/zoom)
- Screenshot all button (composites all visible panels into one PNG, downloads)
- Fullscreen button (browser fullscreen API)

---

### Page 5: Export (`/cases/[id]/export`)

**Purpose:** Choose and download result files.

**Sections:**

**A. Export NIfTI masks** (first section, primary export)
- Checkboxes (all ticked by default):
  - ☑ `prediction_labels.nii.gz` — integer label map (0=background, 1=NCR/NET, 2=Edema, 4=ET)
  - ☑ `mask_wt.nii.gz` — whole tumor binary mask
  - ☑ `mask_tc.nii.gz` — tumor core binary mask
  - ☑ `mask_et.nii.gz` — enhancing tumor binary mask
- Download as ZIP toggle (default: on when >1 file selected)
- Button: **"Download Selected"**

**B. Export PNG screenshots**
- Thumbnail preview row: axial / coronal / sagittal / 3D render (generated server-side
  or client-side at export time from the NiiVue canvas — prefer client-side canvas capture)
- Per-thumbnail checkboxes
- Options:
  - Resolution selector: 1x | 2x | 4x (upscaling for presentation use)
  - Include colorbar/legend: toggle
  - Include case label as watermark: toggle
- Button: **"Export Selected PNGs"**

**C. Export log**
- "Download inference log (.txt)" — downloads `inference_log.txt`

**D. Disclaimer**
A non-dismissable text block (shown on this page only):
> "These results are generated by an experimental deep learning model trained on the
> BraTS dataset for research purposes only. They do not constitute a medical diagnosis
> and must not be used to guide clinical decisions without review by a qualified
> radiologist or neuro-oncologist."

---

### Page 6: Settings (`/settings`)

**Sections:**

**A. Hardware**
- Detected GPU: name, VRAM, CUDA version (or "No GPU detected — inference will use CPU")
- "Run hardware check now" button (re-runs detection)
- Inference device override: Auto | Force CPU | Force CUDA (for debugging)

**B. Model Registry**
- Table: all discovered models (from `checkpoints/` scan)
  - Name, architecture, modalities, checkpoint size, last modified date
  - Status: ✅ Available | ❌ Checkpoint missing
- "Refresh model list" button
- Path to checkpoints directory (read-only, from config)

**C. Storage**
- Path to `data/cases/` directory
- Total disk used by cases
- Per-case breakdown: sortable table (case label, size, date)
- "Clear all failed cases" (bulk delete cases with status `failed`)
- "Open data folder" (file manager shortcut — `shell: open` equivalent)

**D. Inference defaults**
- Default sliding window overlap: slider (0.25–0.75)
- Default SW batch size: number input (1–4)
- Max file size per upload: number input (MB)

**E. Application**
- Dark / Light / System theme selector
- Language (en only for now, but structure for i18n)
- "About" section: version, repo link, disclaimer

---

## 6. Component Inventory

Key reusable components the implementor must build:

### Viewer components (`components/viewer/`)

| Component | Purpose |
|---|---|
| `<NiiVuePanel>` | Single NiiVue canvas with controls (slice slider, zoom, right-click menu) |
| `<ViewerLayout>` | Manages 2×2 / single / 3+1 grid layout, renders correct panels |
| `<LayerControls>` | Sidebar panel: opacity/colour/visibility per layer |
| `<ViewControls>` | Panel toggles, layout selector, crosshair sync toggle |
| `<VolumeRenderPanel>` | 3D panel with render mode controls |
| `<ScreenshotButton>` | Canvas capture → PNG download |

### Upload components (`components/upload/`)

| Component | Purpose |
|---|---|
| `<UploadZone>` | 4-slot drag-and-drop area with auto-classification |
| `<ModalitySlot>` | Single slot (FLAIR/T1/T1ce/T2) with filled/empty states |
| `<FileClassifier>` | Logic: filename → modality mapping |
| `<UploadProgress>` | Per-file progress bar during chunk upload |

### Job components (`components/jobs/`)

| Component | Purpose |
|---|---|
| `<JobModal>` | Full-screen inference progress modal |
| `<LogStream>` | Scrollable, colour-coded SSE log terminal |
| `<GlobalJobBadge>` | Persistent top-nav active job indicator |
| `<JobStatusBadge>` | Coloured badge: queued/running/completed/failed |

### Case components (`components/cases/`)

| Component | Purpose |
|---|---|
| `<CaseCard>` | Grid/list item for case library |
| `<CaseMetaForm>` | Editable case label + notes + tags |
| `<ModelSelector>` | Model list with compatibility indicators |
| `<ModalityIcons>` | Row of 4 icons, coloured/greyed per upload status |

---

## 7. Backend API Specification

All routes prefixed with `/api`. All responses are JSON unless specified.

### Cases

```
GET    /api/cases                   List all cases (sorted by created_at desc)
POST   /api/cases                   Create new case (returns case with UUID)
GET    /api/cases/{id}              Get single case (includes job status)
PATCH  /api/cases/{id}             Update label/notes/tags
DELETE /api/cases/{id}             Delete case + all files

POST   /api/cases/{id}/upload/{modality}   Upload a NIfTI file for a modality
                                           modality: flair | t1 | t1ce | t2
                                           Content-Type: multipart/form-data
                                           Returns: { filename, size_bytes, modality }

DELETE /api/cases/{id}/upload/{modality}   Remove an uploaded modality file
```

### Jobs

```
POST   /api/cases/{id}/infer           Start inference job
                                        Body: { model_id, patch_size_override?, overlap? }
                                        Returns: { job_id, status: "queued", queue_position }

GET    /api/jobs/{job_id}              Get job status
                                        Returns: { job_id, case_id, status, started_at,
                                                   completed_at, error_message, queue_position }

DELETE /api/jobs/{job_id}              Cancel a queued job (cannot cancel running job)

GET    /api/jobs/{job_id}/log          SSE endpoint — streams log lines
                                        Each event: { id: line_number, data: log_line }
                                        Supports Last-Event-ID header for reconnection

POST   /api/jobs/{job_id}/retry        Re-queue a failed job with same settings
```

### Models

```
GET    /api/models                     List all available models
                                        Returns: [{ id, name, architecture, modalities,
                                                    checkpoint_path, checkpoint_size_mb,
                                                    available }]
```

### Files (for NiiVue to load directly)

```
GET    /api/files/{case_id}/inputs/{modality}.nii.gz     Serve input NIfTI
GET    /api/files/{case_id}/outputs/{filename}           Serve output NIfTI
                                                          e.g. prediction_labels.nii.gz
                                                               mask_wt.nii.gz
```

These endpoints must:
- Set `Content-Type: application/gzip`
- Set `Accept-Ranges: bytes` (for NiiVue streaming)
- Set CORS headers to allow `http://localhost:3000`

### System

```
GET    /api/system/hardware            Returns: { gpu_name, gpu_vram_gb, cuda_version,
                                                  cuda_available, device_used }

GET    /api/system/storage             Returns: { total_cases, total_size_bytes,
                                                  cases_dir_path }

GET    /api/health                     Simple health check: { status: "ok" }
```

### SQLite Schema

```sql
CREATE TABLE cases (
    id          TEXT PRIMARY KEY,          -- UUID
    label       TEXT NOT NULL,
    notes       TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',         -- JSON array of strings
    status      TEXT NOT NULL DEFAULT 'ready',
    modalities  TEXT DEFAULT '[]',         -- JSON array: ["flair", "t1ce"]
    model_used  TEXT,
    job_id      TEXT,
    created_at  TEXT NOT NULL,             -- ISO 8601
    updated_at  TEXT NOT NULL,
    inference_duration_seconds REAL
);

CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,      -- UUID
    case_id         TEXT NOT NULL REFERENCES cases(id),
    status          TEXT NOT NULL DEFAULT 'queued',
    model_id        TEXT NOT NULL,
    model_config    TEXT DEFAULT '{}',     -- JSON: patch_size, overlap etc.
    queue_position  INTEGER,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error_message   TEXT,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE log_lines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_log_lines_job_id ON log_lines(job_id, line_number);
CREATE INDEX idx_jobs_case_id ON jobs(case_id);
CREATE INDEX idx_cases_created_at ON cases(created_at);
```

---

## 8. Background Job System

### Requirements
- One GPU job at a time (GPU is a shared, exclusive resource)
- Multiple jobs can be queued (FIFO)
- The FastAPI server must remain responsive during inference
- Logs must persist if the user disconnects and reconnects
- No race conditions when multiple HTTP requests interact with job state

### Design

**Job lifecycle state machine:**
```
QUEUED → RUNNING → COMPLETED
                 → FAILED
       (DELETE)→ CANCELLED   (only from QUEUED state)
```

**JobManager class** (`services/job_manager.py`):
- Singleton, instantiated at FastAPI startup
- Internal asyncio `Queue` for pending jobs
- A single `asyncio.Task` runs the consumer loop
- Consumer loop:
  ```python
  while True:
      job_id = await queue.get()
      await run_job(job_id)          # blocks until subprocess exits
  ```
- `run_job()` uses `asyncio.create_subprocess_exec` to launch the inference worker
  as a child process, capturing stdout line-by-line and writing each line to the
  `log_lines` table in SQLite.

**Inference worker** (`services/inference_worker.py`):
- A standalone Python script, NOT imported by FastAPI at runtime
- Launched as: `python inference_worker.py --job_id X --case_id Y --model_id Z --config {...}`
- Writes all output to stdout (print statements)
- Exit code 0 = success, non-zero = failure
- Internally uses existing `src/inference/predict.py` logic from the training codebase

**SSE log streaming** (`routers/jobs.py`):
- `GET /api/jobs/{job_id}/log` returns an SSE response
- On connect: reads all existing `log_lines` rows from SQLite, sends them immediately
  (replay for reconnecting clients)
- Then polls SQLite for new lines every 250ms (or uses asyncio event / notify mechanism)
- Sends SSE `id:` field as `line_number` — client sends `Last-Event-ID` header on reconnect,
  server starts replay from that line number
- When job status becomes `completed` or `failed`, sends a special SSE event:
  `event: job_status\ndata: {"status": "completed"}\n\n`
  and closes the stream

**Race condition prevention:**
- All job state mutations go through `JobManager` methods which use asyncio locks
- SQLite WAL mode enabled (`PRAGMA journal_mode=WAL`) for concurrent reads during writes
- The inference subprocess is the ONLY writer to `log_lines` during a run
  (it writes via the parent process pipe, not directly to SQLite)
- On server restart: jobs in `running` state are set to `failed` with message
  "Server restarted during inference. Please retry."

**Queue position tracking:**
- `queue_position` column in jobs table is updated atomically when a job is enqueued/dequeued
- API response for queued jobs includes `"queue_position": 2` so UI can show "Position 2 in queue"

---

## 9. 3D Viewer Requirements

### Library: NiiVue

Use `@niivue/niivue` npm package. NiiVue is a WebGL-based neuroimaging viewer that natively
supports NIfTI volumes, MPR (multi-planar reconstruction), and volume rendering.

**Key NiiVue API usage:**

```typescript
// Initialize
const nv = new Niivue({ backColor: [0, 0, 0, 1] })
nv.attachToCanvas(canvasRef.current)

// Load MRI volume (background)
await nv.loadVolumes([{
  url: `/api/files/${caseId}/inputs/flair.nii.gz`,
  colormap: 'gray',
  opacity: 1.0,
}])

// Load mask overlays
await nv.addVolume({
  url: `/api/files/${caseId}/outputs/mask_wt.nii.gz`,
  colormap: 'green',
  opacity: 0.5,
})

// Set slice display mode
nv.setSliceType(nv.sliceTypeMultiplanar)  // axial + coronal + sagittal
nv.setSliceType(nv.sliceTypeRender)       // 3D volume render
nv.setSliceType(nv.sliceTypeAxial)        // single axial

// Volume render settings
nv.setVolumeRenderIllumination(0.6)
```

**Implementation notes:**

- Each NiiVue instance manages one canvas. For the 2×2 layout, create 4 canvas elements,
  each with its own NiiVue instance, all loaded with the same volumes.
  Crosshair sync is achieved by listening to `nv.onLocationChange` events and updating
  the scene location on all other instances.

- Alternatively, NiiVue supports a multi-planar layout natively in a single canvas
  (`sliceTypeMultiplanar`) — use this when showing axial/coronal/sagittal together.
  Use a separate canvas for the 3D render panel.

- Recommended architecture: **2 NiiVue instances**:
  1. MPR canvas (axial + coronal + sagittal via `sliceTypeMultiplanar`)
  2. 3D render canvas (via `sliceTypeRender`)
  Both share the same loaded volumes.

- NiiVue's `loadVolumes` accepts a URL — this is why the backend serves NIfTI files
  via `/api/files/...` with proper `Content-Type` and `Accept-Ranges` headers.

**Viewer state in Zustand:**

```typescript
interface ViewerState {
  backgroundModality: 'flair' | 't1' | 't1ce' | 't2'
  backgroundOpacity: number
  backgroundWindowMin: number
  backgroundWindowMax: number
  layers: {
    wt: { visible: boolean; color: string; opacity: number }
    tc: { visible: boolean; color: string; opacity: number }
    et: { visible: boolean; color: string; opacity: number }
  }
  activePanels: { axial: boolean; coronal: boolean; sagittal: boolean; volume3d: boolean }
  layout: '2x2' | 'single' | '3+1'
  crosshairSync: boolean
  selectedPanel: 'axial' | 'coronal' | 'sagittal' | 'volume3d' | null
}
```

When viewer state changes, the React component reconciles with NiiVue:
```typescript
useEffect(() => {
  nv.setOpacity(0, viewerState.backgroundOpacity)   // index 0 = background volume
  nv.setOpacity(1, viewerState.layers.wt.opacity)   // index 1 = WT mask
  // ...
  nv.drawScene()
}, [viewerState])
```

---

## 10. Export System

### NIfTI export

The output NIfTI files already exist on disk after inference. Export is simply a zip of
selected files served as a download:

```
GET /api/cases/{id}/export/nifti?files=prediction_labels,mask_wt,mask_tc,mask_et
```

Backend: streams a ZIP of requested files. No re-processing needed.

### PNG export

PNG screenshots are captured client-side from the NiiVue canvas elements:

```typescript
// For each visible panel canvas:
const canvas = document.getElementById('niivue-axial') as HTMLCanvasElement
const dataUrl = canvas.toDataURL('image/png')
// Bundle into a ZIP using jszip, trigger download
```

The resolution multiplier (`1x | 2x | 4x`) is applied by scaling the canvas before capture
(or by rendering at higher resolution — NiiVue supports `gl.drawingBufferWidth`).

**If implementing server-side PNG generation** (fallback for high-resolution exports):
- Backend route `POST /api/cases/{id}/export/png` triggers a headless NiiVue render
  using a headless WebGL context (e.g., `node-gl` or `headless-gl` in a Node.js subprocess)
- This is complex; prefer client-side canvas capture for MVP.

### Export ZIP assembly

Use `jszip` in the browser:
```typescript
const zip = new JSZip()
zip.file('mask_wt.nii.gz', await fetch(niftiUrl).then(r => r.blob()))
zip.file('axial.png', axialDataUrl.split(',')[1], { base64: true })
const blob = await zip.generateAsync({ type: 'blob' })
saveAs(blob, `${caseLabel}_export.zip`)
```

---

## 11. Hardware Auto-Detection

### Backend detection at startup (`config.py`):

```python
import torch

def detect_hardware():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
    else:
        device = "cpu"
        gpu_name = None
        vram = None
        cuda_version = None
    return {
        "device": device,
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(vram, 1) if vram else None,
        "cuda_version": cuda_version,
        "cuda_available": torch.cuda.is_available()
    }
```

The inference worker receives the device string via CLI arg and passes it to `torch.device()`.

The frontend shows the detected hardware in the Dashboard status bar and Settings page.

The settings page has an override option stored in `data/app_settings.json`:
```json
{ "device_override": "auto" }   // "auto" | "cpu" | "cuda"
```

If `device_override` is `"auto"`, the detected device is used. If the user forces `"cpu"`,
inference runs on CPU even if CUDA is available. This is useful for testing or when the GPU
is busy with another workload.

### Inference time estimation (optional, best-effort)

After the first inference run on the detected hardware, store the elapsed time per case in
`metadata.json`. On subsequent runs, show: `"Estimated time: ~2m 30s (based on previous runs)"`

---

## 12. Dark / Light Mode

- Frontend: implement via CSS custom properties (design tokens)
- Theme stored in `localStorage` as `"theme": "dark" | "light" | "system"`
- `"system"` reads `prefers-color-scheme` media query
- The NiiVue canvas background colour should adapt: dark mode → `[0,0,0,1]`, light mode → `[1,1,1,1]`
- All UI surfaces use theme tokens — no hardcoded colours anywhere in components

---

## 13. Containerization & Deployment

### Option A: Docker Compose (recommended, beginner-friendly)

**Prerequisites:** Docker Desktop installed (one installer, works on Windows/Mac/Linux).
No Python, Node, or any other tool required.

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../checkpoints:/app/checkpoints:ro   # model weights (read-only)
      - ../src:/app/src:ro                   # training source code (read-only)
      - ./data:/app/data                     # persistent case storage (read-write)
    environment:
      - DATA_DIR=/app/data
      - CHECKPOINTS_DIR=/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]   # comment out if no NVIDIA GPU

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
```

`backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version by default; override at build time for CUDA)
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For CUDA support, provide a separate `Dockerfile.cuda`:
```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# ... rest of install
```

And in `docker-compose.yml`, allow switching via `build: dockerfile: Dockerfile.cuda`.

`frontend/Dockerfile`:
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

**User-facing start command** (inside `webapp/`):
```bash
docker compose up
# then open http://localhost:3000
```

### Option B: Native dev mode (for developers / contributors)

`start.sh` (inside `webapp/`):
```bash
#!/bin/bash
set -e

echo "Starting BrainSeg webapp in dev mode..."

# Backend
cd backend
pip install -r requirements.txt -q
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend
cd ../frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ App running at http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers."

wait $BACKEND_PID $FRONTEND_PID
```

### User-facing `webapp/README.md` content outline

The README must be usable by a physician with no coding experience:

1. **Requirements** — just "Install Docker Desktop" with a download link
2. **Quick start** — 3 steps: clone, `cd webapp`, `docker compose up`
3. **First run** — will take a few minutes to download/build; subsequent starts are instant
4. **How to use** — screenshot-heavy walkthrough of the upload → inference → export flow
5. **Troubleshooting** — GPU not detected, port conflicts, disk space
6. **Stopping the app** — `Ctrl+C` or `docker compose down`
7. **Where is my data?** — explains `webapp/data/cases/`

---

## 14. ML Model Integration

### Model Registry (`services/model_registry.py`)

Scans `checkpoints/` for available models. Expected structure (from training codebase):
```
checkpoints/
  unet_4ch/
    best_model.pth
  unet_flair/
    best_model.pth
  cnn_4ch/
    best_model.pth
  ...
```

Reads `configs/experiments.yaml` to get human-readable names and modality requirements.
Maps checkpoint directories to experiment names.

Returns model objects:
```python
@dataclass
class ModelInfo:
    id: str                    # "unet_4ch"
    name: str                  # "Full 4-channel U-Net"
    architecture: str          # "UNet" | "CNN"
    modalities: list[str]      # ["flair", "t1", "t1ce", "t2"]
    checkpoint_path: str
    checkpoint_size_mb: float
    available: bool            # checkpoint file exists
```

### Inference worker integration

The inference worker (`inference_worker.py`) wraps existing code from `src/inference/predict.py`:

```python
# inference_worker.py (standalone script, not imported by FastAPI)
import sys
import json
import argparse
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # integrative-project root

from src.inference.predict import run_inference
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config_overrides", default="{}")
    args = parser.parse_args()

    cfg = load_config("full")
    # Override paths to use the case-specific directories
    cfg["paths"]["predictions_dir"] = str(Path(args.case_dir) / "outputs")
    # ... call run_inference with constructed sample dict

if __name__ == "__main__":
    main()
```

All `print()` statements from existing training code go to stdout, captured by the parent
process and written to the `log_lines` SQLite table.

### Modality mapping for inference

When the user uploads files and selects a model:
- The file paths in `data/cases/{id}/inputs/` are passed to the inference worker
- The worker builds a `sample` dict compatible with `src/data/dataset.discover_brats_samples`
- Missing modalities are handled by the existing `zero_pad_batch` logic from `run_cross_modality.py`

---

## 15. Error Handling & Edge Cases

| Scenario | Handling |
|---|---|
| User uploads non-NIfTI file | Frontend rejects before upload, shows inline error: "Only .nii and .nii.gz files are supported" |
| NIfTI file is corrupted | Backend validates with nibabel on receipt; returns 422 with message |
| GPU OOM during inference | Worker catches RuntimeError, logs "[ERROR] CUDA out of memory", exits with code 1. Job status → failed. Retry button visible. |
| Server restart mid-inference | On startup, any `running` job is transitioned to `failed` with message "Interrupted by server restart" |
| Multiple jobs submitted | Second job goes to queue. UI shows "Position 2 in queue." First job's log streams. |
| User deletes a case with a running job | Not allowed — "Cannot delete case while inference is running." Cancel job first. |
| Model checkpoint missing | ModelRegistry marks it `available: false`. UI shows it greyed out in model selector with tooltip "Checkpoint not found" |
| SSE connection drops | Browser `EventSource` automatically reconnects. Backend replays logs from `Last-Event-ID` |
| Disk full | Worker fails, logs error. Backend catches `OSError` from file writes, sets job to failed. Dashboard shows disk usage warning when >90% full. |
| Case data directory missing | Backend recreates directory structure on case read if outputs/ is missing |
| NiiVue fails to load a file | Viewer shows a per-panel error state: "Could not load [mask_wt.nii.gz]. File may be corrupt." |

---

## 16. Future Extensibility Notes

The implementor should structure the codebase to accommodate these likely extensions,
even if not building them now:

1. **Ground truth comparison** — physicians may want to upload a reference segmentation
   to compare against the model's output. The viewer should support loading an optional
   5th volume (ground truth) alongside the 3 prediction masks.

2. **Multi-case comparison** — a side-by-side viewer for two cases (pre/post treatment).

3. **Report generation** — a PDF report with thumbnails, metrics, and the disclaimer.
   Structure the export page so a "Generate PDF Report" button can be added.

4. **DICOM support** — physicians often work with DICOM series, not NIfTI.
   The upload pipeline should have a clear conversion hook (e.g., `dcm2niix` subprocess).

5. **Authentication** — if this ever becomes multi-user or semi-public, a simple
   session-based auth layer should be addable without restructuring. Keep all case data
   operations behind an API (no direct file access from frontend) to make this trivial.

6. **Metrics display** — if ground truth is available, show Dice/HD95 scores per region
   in a collapsible panel on the Case Detail page.

7. **Annotation / correction** — a future version may allow physicians to manually correct
   the predicted mask using brush/eraser tools (NiiVue supports drawing mode).

---

*End of specification.*
