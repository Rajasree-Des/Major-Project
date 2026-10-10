# Current ThermalSR project state

**As of:** 2026-10-10 (working tree; Git HEAD `0542cd2`).

## What the system is

A local research codebase that:

1. Ingests Landsat 8/9 Collection 2 Level-2 surface temperature and reflectance.
2. Builds 256×256 patch datasets with per-scene z-score normalization.
3. Synthesizes 128×128 thermal LR via blur + area downsample + 0.05 K noise.
4. Reconstructs 256×256 thermal with **official SwinIR-S ×2** weights adapted to one channel (not thermally fine-tuned on the QA dataset).
5. Optionally applies a **deterministic synthetic-forward consistency correction** (β = 1.0).
6. Renders three visualization colorizations that **do not change** the Kelvin array.
7. Serves a local FastAPI + Vite UI (ThermaScope) that can run the same pipeline on selected patches and display frozen evaluation JSON.

It is **not** a trained thermal-domain SwinIR, **not** a radiative-transfer inversion, and **not** a 15 m thermal accuracy demonstration.

## Architecture (data flow)

```text
dataset/raw/*.tif (ST_B10, SR_B2/3/4, QA_PIXEL)
        │
        ▼
preprocessing/pipeline_qa.py  →  dataset/processed_qa/<scene>/{thermal,rgb,metadata.json}
        │
        ▼
dataset_loader.ThermalSatelliteDataset   (z-score thermal + RGB + valid_mask, 256×256)
        │
        ▼
sr_data.ThermalSRDataset
   degrade_thermal_patch(HR Kelvin) → LR 128×128  (HR kept only as synthetic target / for eval)
        │
        ├─ bicubic upsample (torch, align_corners=False)
        ├─ SwinIR-S lightweight_x2 (1-ch, official DIV2K weights)
        └─ refine_hr_normalized (Kelvin residual vs noise-free forward model, then re-normalize)
                │
                ▼
        denormalize → Kelvin HR grid
                │
                ├─ evaluation/metric_core.py  vs HR reference (metrics only)
                └─ colorization.apply_all_methods  → PNG A/B/C (Kelvin unchanged)
```

Primary packages: `preprocessing/`, `dataset_loader/`, `sr_data/`, `models/swinir/`, `training/`, `evaluation/`, `colorization/`, `pipeline/`, `backend/`, `frontend/`.

## Frozen scientific configuration

Source: `pipeline/frozen_config.py` + `outputs/evaluation/physics_refinement_v1/METHOD.txt`.

| Parameter | Value |
|-----------|--------|
| Processed data | `dataset/processed_qa/` (`qa_v1`) |
| Split | `dataset/splits/scene_split_qa_v1.json` |
| Degradation | scale 2, σ 1.2 HR px, area downsample, 0.05 K noise, seed 42 |
| SwinIR | lightweight_x2, 904,744 params |
| Checkpoint | `pretrained/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` |
| Refinement β | 1.0 (frozen; tuned on Kerala validation only) |
| Colormap | inferno, 295–320 K |
| Guided filter | radius 8, ε 0.02, edge_mix 0.35 |
| Overlay | thermal_weight 0.55 |
| Device used in evals | CPU, PyTorch `2.14.1+cpu` |

## Inputs and outputs

**Inputs:** Landsat C2 L2 GeoTIFFs; or already-processed NumPy patches.  
**Learned model:** none trained to convergence on the QA split; inference uses adapted DIV2K weights.  
**Outputs (scientific):** `outputs/evaluation/**/results.json`, per-scene/per-patch CSV, report.txt.  
**Outputs (demo):** `outputs/colorization/`, `outputs/integrated_pipeline_v1/`, `outputs/ui_runs/`.

## How to run (existing, not new experiments)

```powershell
# API
py -m backend

# UI
cd frontend; npm run dev
# http://127.0.0.1:5173/

# Fast tests (do not overwrite frozen eval directories)
py -m pytest tests -q -m "not slow"
```

Locked-test evaluation directories already exist and **must not be overwritten** (`evaluate_multiscene.py` refuses non-empty protected outputs).

## Dependencies

`requirements.txt`: pystac-client, planetary-computer, rasterio, tifffile, numpy, matplotlib, torch≥2.0, fastapi, uvicorn, httpx. Frontend: Node 18+, Vite/React (see `frontend/package.json`).

## What is unfinished

- Converged thermal fine-tuning on the four train scenes.
- Locked-test eval of a fine-tuned checkpoint, with and without refinement.
- Dedicated inference latency/memory protocol (GPU memory impossible on the CPU wheel used).
- External native higher-resolution thermal ground truth.
- Git commit of the scientific tree.
- IEEE manuscript PDF not stored in the repo.
