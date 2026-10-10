# ThermalSR project history (evidence-based)

**Audit date:** 2026-10-10  
**Git HEAD (committed):** `0542cd2b13ae184d8eeb6487d00d71180f5b14d9` (2026-09-27, `chore: checkpoint before Claude development`)  
**Working tree:** most scientific code, datasets, evaluations, and the ThermaScope app are **untracked** relative to HEAD.  
**Manuscript PDF:** `ThermalSR_IEEE.pdf` was **not found** in the repository or typical local search paths. Paper-structure claims in this audit come from the paper title, the user’s table specification, and repository artifacts — not from a quoted PDF.

Legend: **FACT** = file/log/git evidence; **INFERENCE** = reasoned from timestamps, transcripts, or code; **UNKNOWN** = no evidence.

This audit did not modify frozen evaluation JSON, checkpoints, splits, or datasets.

---

## 0. What the project set out to build

**FACT (user/task language preserved across development sessions):** a “physics-aware deep learning framework for super-resolution and guided colorization of thermal satellite imagery,” later titled **ThermalSR**.

Intended pipeline (from successive task briefs, not from a committed design doc):

1. Acquire Landsat Collection 2 Level-2 scenes.  
2. Convert DN → Kelvin / reflectance, mask invalid pixels, extract 256×256 patches.  
3. Synthesize ×2 low-resolution thermal inputs.  
4. Super-resolve with SwinIR.  
5. Fine-tune SwinIR on thermal data.  
6. Apply a “physics-aware” refinement.  
7. Colorize for visualization (thermal-only and RGB-guided).  
8. Evaluate quantitatively and (later) against an external higher-resolution thermal benchmark.

**FACT:** the implemented scientific pipeline today is: Landsat C2 L2 → QA-aware patches → synthetic ×2 LR → **pretrained** SwinIR-S (1-channel adapted) → **synthetic forward-consistency** refinement → visualization colorization. Converged thermal fine-tuning and an external ASTER/Landsat 15 m benchmark were **not** completed.

---

## 1. Repository genesis (2026-07-30 to 2026-09-27) — committed Git

| Date | Commit | Evidence |
|------|--------|----------|
| 2026-07-30 | `f54fd4c` Initial project structure | Empty placeholders: `README.md`, `dataset/`, `docs`, `evaluation`, `models`, `preprocessing`, `training`, `scripts` |
| 2026-08-02 | `6129789` dataset downloader | STAC/Planetary Computer downloader introduced |
| 2026-08-05 | `ecde9b8` Initial updates | Frontend mockup + `scripts/preview_scene.py` + first scene inspector outputs |
| 2026-09-27 | `0542cd2` checkpoint before Claude development | Still essentially: download + inspect + React mock UI |

**FACT:** HEAD contains a **ThermaScope AI mock frontend** (`frontend/src/data/mockData.ts`, mock API) and a working scene inspector for `LC09_L2SP_143050_20240327_02_T1`. It does **not** contain the current `dataset_loader/`, `sr_data/`, `colorization/`, `pipeline/`, `backend/`, QA dataset, or evaluation trees.

**INFERENCE (agent transcript [ThermaScope early audit](afed94d1-6d29-4086-b816-2184b23576fc)):** as of an early October 2026 audit, the project was “real data ingestion + inspection” plus a UI mockup, with no ML training/inference and no frontend–backend integration.

---

## 2. Scene inspection of the reference Landsat 9 scene

**Goal:** verify a real Collection 2 Level-2 product could be read.

**Implemented:** `scripts/preview_scene.py`  
**Executed:** yes. Output: `outputs/previews/LC09_L2SP_143050_20240327_02_T1/report.json` (+ preview PNGs committed under `outputs/previews/`).

**FACT:** scene `LC09_L2SP_143050_20240327_02_T1`, CRS EPSG:32644, 30 m, bands ST_B10 / SR_B2 / SR_B3 / SR_B4, 7641×7781.

**Status:** complete for that scene.

---

## 3. Preprocessing v0 (NoData-only, single scene) — ~2026-10-05

**Goal:** radiometric conversion, validity mask, scene z-score, 256×256 patches.

**Implemented (untracked vs HEAD placeholders):** `preprocessing/radiometric.py`, `preprocessing/pipeline.py`, `preprocessing/patchify.py`, `preprocessing/export_numpy.py`.

**Executed:** yes. Output: `dataset/processed/LC09_L2SP_143050_20240327_02_T1/`  
- 2330 accepted patches, 1092 skipped  
- patch 256×256, stride 128, reject if invalid fraction > 10%  
- mask = finite and not NoData on ST_B10 and RGB (no QA_PIXEL flags)  
- thermal z-score mean 316.784 K, std 4.902 K  

**Tests:** later `tests/` cover related loaders; this original export predates `pytest.ini`.

**Status:** complete for one scene; later superseded for scientific eval by `processed_qa`.

---

## 4. PyTorch dataset + synthetic ×2 SR data

**Goal:** load patches and build LR 128×128 / HR 256×256 pairs.

**Implemented:** `dataset_loader/thermal_dataset.py`, `dataset_loader/physical.py`, `sr_data/dataset.py`, `sr_data/degradation.py`, `sr_data/metadata.py`.

**Degradation (FACT, `DegradationConfig`):** scale 2, Gaussian blur σ = 1.2 HR pixels, masked area downsample, 0.05 K Gaussian noise, `base_seed=42`. Applied in Kelvin, then remapped to the same per-scene z-score space as HR.

**Status:** complete and used by all later evaluations.

---

## 5. First SwinIR path: custom architecture (abandoned)

**Goal (early brief):** implement SwinIR thermal SR.

**INFERENCE (transcript 2026-10-05):** a custom SwinIR-like network was trained from (near) scratch on the single scene. A 2-epoch CPU run started; epoch 1 finished.

**FACT still on disk:** `outputs/models/swinir_custom_scratch/{best.pt,latest.pt,history.json}`  
- `best.pt` SHA-256 `7d2890a22e8f64fe20741e2ea5aabd3bec283ae015c5e77d98c86787ad87971a`, 11,568,013 bytes  
- history shows 2 epochs of training metrics on a development split  

**FACT (2026-10-07 workflow decision):** the project **explicitly abandoned** training from random initialization and adopted official pretrained SwinIR.

**Status:** experimental / superseded. **Not** used in locked-test evaluation.

---

## 6. Official pretrained SwinIR, 1-channel adaptation, preliminary “fine-tune”

**Goal:** load Jingyun Liang SwinIR-S ×2 DIV2K weights, adapt RGB→1 channel, fine-tune on Landsat thermal.

**Implemented:** vendored `models/swinir/network_swinir.py` (commit `6545850`, Apache 2.0), `models/swinir/pretrained.py` (strict `load_state_dict` after explicit tensor mapping), `training/train_swinir.py`.

**Executed:**

| Event | Evidence | What actually happened |
|-------|----------|------------------------|
| Checkpoint download | `pretrained/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` | Official DIV2K RGB weights; SHA-256 `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7`; 17,147,989 bytes |
| Adaptation | eval JSON `pretrained_checkpoint` | 327 tensors / 902,040 params copied; 3 tensors / 2,704 params adapted (`conv_first.weight` sum over RGB; `upsample.0` mean over RGB groups); 0 newly initialized; 36 buffers regenerated; **strict** load |
| Smoke / tiny train | `outputs/models/swinir/history.json`, `best.pt` | Epoch 0 val of adapted pretrained; then epochs 1–2 with `--freeze-epochs 2 --max-train-batches 3 --batch-size 2` on `dataset/processed/LC09_...` |
| CPU feasibility | `outputs/benchmarks/cpu_swinir/` | 10 optimizer steps; ~33 s/batch; **estimates** of full-epoch time |

**FACT:** `best.pt` note in `outputs/evaluation/sr_baseline/results.json`: “PRELIMINARY sanity-check checkpoint from a few CPU optimizer steps; not a converged fine-tune.” Freeze logic (`training/train_swinir.py`) sets warmup when `epoch <= freeze_epochs`, so both recorded training epochs only updated adapted `conv_first` / `upsample` layers.

**FACT:** a cloud-GPU guide exists (`docs/CLOUD_GPU_TRAINING.md`). **UNKNOWN / no artifact:** a completed cloud-GPU fine-tune checkpoint evaluated on the QA locked test.

**Status:** A/B/C (load, adapt, infer) complete. D/E (real thermal fine-tune + locked-test eval of that checkpoint) **not complete**.

---

## 7. Single-scene development evaluation (`sr_baseline`)

**Executed:** yes. `outputs/evaluation/sr_baseline/` timestamp `2026-10-09T06:03:39Z`.

- Dataset: original `dataset/processed/LC09_L2SP_143050_20240327_02_T1` (NoData-only, **not** `processed_qa`)  
- Split: random **patch** val split (233 patches) — labeled “DEVELOPMENT SPLIT — NOT FINAL SCIENTIFIC EVALUATION”  
- Methods: bicubic, pretrained SwinIR, preliminary fine-tune  
- Full-run wall-clock timings **invalid** (machine suspend ~11 h); separate `timing_check_10_batches.json` is the valid timing artifact  

**Status:** complete as a development check; **must not** be used as the paper’s test table.

---

## 8. Multi-scene expansion + QA masking (2026-10-09 morning)

**Goal:** more climates, Landsat 8 and 9, CFMask QA exclusion.

**Executed:** yes.

- Download log: `outputs/dataset_expansion/download_log.txt` — 6 additional scenes requested 2026-10-09 12:12 IST via Planetary Computer STAC (`ST_B10`, `SR_B2`, `SR_B3`, `SR_B4`, `QA_PIXEL`). The original LC09 143/050 scene was already present.  
- QA processing: `preprocessing/pipeline_qa.py`, `preprocessing/qa_mask.py` (qa_v1)  
- Report: `outputs/dataset_expansion/qa_dataset_report.txt` — **7 scenes, 15,985 patches**, 15.62 GiB processed  

QA bits excluded (USGS LSDS-1619 / STAC): fill, dilated cloud, cirrus, cloud, cloud shadow, snow. Combined with per-band NoData.

**Scene-level split:** `dataset/splits/scene_split_qa_v1.json` (SHA-256 `12bd1ff703b60732d23b7337c2d86d4eefabfd1542114d7c0ef33de2c6f40f7e`), written by `scripts/make_scene_split.py`. Manual diversity split, no random seed. Reference scene placed in **train** because the preliminary checkpoint had already seen it.

**Status:** complete. This is the dataset version used for all later scientific evaluations.

---

## 9. Multi-scene pretrained baseline (`multiscene_qa_v1`)

**Command (log):** `py -W ignore -m evaluation.evaluate_multiscene --roles validation test_locked`  
**Output:** `outputs/evaluation/multiscene_qa_v1/`  
**Timestamp (locked test):** `2026-10-09T08:24:14Z`  
**Training performed:** false  

Methods: bicubic ×2; SwinIR-S pretrained thermal-adapted. Non-overlapping 256-aligned patches.

Metric audit: `outputs/evaluation/metrics_audit_v1/` and `multiscene_qa_v1/audit/` — Kelvin pooling vs CSV **passed**.

**Status:** complete; frozen.

---

## 10. Consistency refinement (`physics_refinement_v1`)

**Implemented:** `sr_data/consistency_refinement.py` — **not** radiative transfer.

**β selection:** `tune_beta.json` — 137 Kerala validation patches (25%, seed 42); β ∈ {0, 0.25, 0.5, 0.75, 1.0}; **best_beta = 1.0**. Frozen in `METHOD.txt`. Locked test was not used for tuning (log: “Frozen beta=1.00 from METHOD.txt / CLI (no tuning on test)”).

**Validation:** 414 held-out Kerala patches.  
**Locked test:** command `py -W ignore -m evaluation.evaluate_physics_refinement --role test_locked` at 2026-10-09 18:40 IST; results `2026-10-09T14:22:49Z`; 1,165 patches; ~4357 s CPU.

Base model: **pretrained** SwinIR, not fine-tuned.

**Status:** complete on pretrained+refine; **missing** fine-tuned+refine.

---

## 11. Colorization prototype

**Executed:** `py -m scripts.run_guided_colorization_v1`  
**Output:** `outputs/colorization/guided_colorization_v1/` — 4 Kerala patches, methods A/B/C, 15.9 s CPU.  
**FACT:** colorization does not mutate Kelvin (`colorization/pipeline.py` asserts identity). No PSNR/SSIM for colorization (visualization only).

**Status:** prototype complete on 4 validation patches.

---

## 12. Integrated pipeline

**Executed:** `py -m scripts.run_integrated_pipeline_v1`  
**Output:** `outputs/integrated_pipeline_v1/` — same 4 patches, 17.4 s.  
**Integrity audit (2026-10-09):** `scripts/final_audit_v1.py` → standalone vs integrated Kelvin **exact match**; method PNGs exact match; legend PNGs differ (matplotlib titles).

**Status:** complete for the audited patches.

---

## 13. ThermaScope local web app (2026-10-09 evening)

**Implemented:** `backend/` FastAPI + rewritten `frontend/src` (no longer mock-only).  
**Executed:** UI jobs `outputs/ui_runs/20261009T180617_d954f91f/` and `20261009T181637_82c7872d/`.  
**Tests recorded:** `outputs/ui_integration_report_v1.txt` — 7 fast API tests passed; 1 CPU job test passed (~65 s); `npm run build` succeeded.

**Status:** functional local demo; one job at a time; not a substitute for locked-test metrics.

---

## 14. Final scientific-integrity audit (same day)

**Executed:** `py -m scripts.final_audit_v1`  
**Output:** `outputs/final_audit_v1/`  
**Recorded pytest (that audit, not this one):** 28 passed in ~90 s on listed test modules.

**This forensic audit (2026-10-10):** re-running the fast suite failed at **collection** (`WinError 4551` Application Control). Not a scientific-regression result.

**Status:** complete; frozen hashes recorded.

---

## 15. Current repository state (this audit)

- Scientific pipeline exists and has frozen locked-test numbers for **bicubic**, **pretrained SwinIR**, and **pretrained + consistency refinement**.  
- Almost all of that work is **uncommitted**.  
- No `ThermalSR_IEEE.pdf` in-repo.  
- No converged thermal fine-tune on `processed_qa` train scenes.  
- No external ASTER / native 15 m thermal benchmark artifacts found.

---

## Stage completion matrix

| Stage | Implemented | Executed with outputs | Scientific completeness |
|-------|-------------|----------------------|-------------------------|
| Landsat acquisition | yes | yes (7 scenes) | complete for this study |
| Radiometric conversion | yes | yes | complete |
| QA masking | yes (qa_v1) | yes | complete; original processed is NoData-only |
| Patch extraction | yes | yes | complete |
| Synthetic ×2 | yes | yes | complete; not real 15 m |
| Scene-level split | yes | yes | complete; locked test unused for selection |
| SwinIR official + adapt | yes | yes | complete |
| Thermal fine-tuning | code exists | only preliminary CPU warmup | **not a paper-grade experiment** |
| Bicubic / pretrained eval | yes | yes (QA locked test) | complete |
| Refinement | yes (consistency) | yes (pretrained base) | complete as consistency; not RT physics |
| Colorization | yes | 4-patch prototype | visualization only |
| Multi-scene tests | yes | val + locked test | 3 scenes scored; 4 train unused for training |
| Compute benchmark | partial | training micro-bench + eval `forward_s` | no dedicated inference memory protocol |
| Integrated pipeline | yes | 4 patches + UI jobs | complete for demo |
| Automated tests | yes | prior 28 passed; this audit re-ran fast suite | see integrity report |
| Website | yes | API + UI jobs | functional, not mock-only |
| External 15 m benchmark | no | no | **not done** |
| IEEE PDF in repo | no | — | **missing** |
