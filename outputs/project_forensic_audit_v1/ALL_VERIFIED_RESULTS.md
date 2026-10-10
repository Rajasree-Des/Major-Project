# All verified numerical results

Units: **Kelvin MAE/RMSE** unless `_z` is stated. PSNR in dB (per-patch data range = valid HR range in the space compute_psnr sees; invariant to affine z-score). SSIM uses C1=0.01², C2=0.03² in **z-units** (scene-std dependent).

**Do not mix** pooled Kelvin with macro-over-scenes RMSE. **Do not mix** `processed` v0 with `processed_qa`.

Recalculated patch-mean PSNR/SSIM over 1165 patches = (scene_mean × n_patches) summed / 1165. Original JSON stores per-scene means and equal-scene macro means, not this patch-weighted mean. Discrepancy vs macro is < 0.002 dB because 582 vs 583 patches.

---

## A. Locked test — publication-grade comparison set

**Artifact:** `outputs/evaluation/physics_refinement_v1/test_locked/results.json`  
**Identical bicubic/pretrained:** `outputs/evaluation/multiscene_qa_v1/test_locked/results.json`  
**Split:** `scene_split_qa_v1` test; 1165 non-overlapping patches; 76,196,944 valid pixels  
**Checkpoint:** official SwinIR-S ×2, SHA-256 `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7`  
**Model:** pretrained + 1-ch adaptation; `training_performed: false`

### A1. Per scene (from JSON; not recalculated)

**LC08 Deccan** (`LC08_L2SP_144048_20240427_02_T1`), 582 patches, 38,081,751 valid px, std 4.974 K

| Method | MAE_z | RMSE_z | MAE (K) | RMSE (K) | PSNR | SSIM |
|--------|------:|------:|--------:|---------:|-----:|-----:|
| bicubic | 0.05461749 | 0.07631568 | 0.27168218 | 0.37961484 | 36.277860 | 0.879682 |
| swinir_pretrained | 0.05340003 | 0.07391147 | 0.26562620 | 0.36765562 | 36.532613 | 0.884174 |
| swinir_consistency_refined | 0.03925606 | 0.05573300 | 0.19527028 | 0.27723100 | 39.033354 | 0.925454 |

**LC09 Assam** (`LC09_L2SP_137042_20230110_02_T1`), 583 patches, 38,115,193 valid px, std 2.868 K

| Method | MAE_z | RMSE_z | MAE (K) | RMSE (K) | PSNR | SSIM |
|--------|------:|------:|--------:|---------:|-----:|-----:|
| bicubic | 0.05355301 | 0.08386205 | 0.15360948 | 0.24054682 | 35.181712 | 0.858701 |
| swinir_pretrained | 0.05320983 | 0.08190829 | 0.15262513 | 0.23494272 | 35.314642 | 0.858932 |
| swinir_consistency_refined | 0.04287516 | 0.06865769 | 0.12298152 | 0.19693519 | 36.975444 | 0.895280 |

### A2. Aggregate — JSON original

| Method | macro RMSE (K) | **pooled MAE (K)** | **pooled RMSE (K)** | macro PSNR | macro SSIM |
|--------|---------------:|-------------------:|--------------------:|-----------:|-----------:|
| bicubic | 0.31008083 | 0.21261992 | 0.31775175 | 35.729786 | 0.869192 |
| swinir_pretrained | 0.30129917 | 0.20910087 | 0.30849120 | 35.923628 | 0.871553 |
| swinir_consistency_refined | 0.23708310 | 0.15911004 | 0.24044104 | 38.004399 | 0.910367 |

### A3. Recalculated patch-mean PSNR/SSIM (1165 patches)

| Method | PSNR patch-mean | SSIM patch-mean | vs JSON macro PSNR |
|--------|----------------:|----------------:|--------------------|
| bicubic | 35.729317 | 0.869354 | −0.00047 dB |
| swinir_pretrained | 35.922247 | 0.871714 | −0.00138 dB |
| swinir_consistency_refined | 38.003516 | 0.910353 | −0.00088 dB |

**Recommended table numbers (pooled Kelvin + patch-mean PSNR/SSIM):**

| Method | PSNR (dB) | SSIM | MAE (K) | RMSE (K) |
|--------|----------:|-----:|--------:|---------:|
| Bicubic | 35.729 | 0.8694 | 0.2126 | 0.3178 |
| Pretrained SwinIR | 35.922 | 0.8717 | 0.2091 | 0.3085 |
| Pretrained + consistency refine | 38.004 | 0.9104 | 0.1591 | 0.2404 |

Compatible improvement **refined vs pretrained** (same patches): PSNR **+2.081 dB**; SSIM **+0.0386**; MAE **−0.0500 K (−23.91%)**; RMSE **−0.0681 K (−22.06%)**.

Compatible improvement **pretrained vs bicubic**: PSNR **+0.193 dB**; SSIM **+0.0024**; MAE **−0.0035 K (−1.66%)**; RMSE **−0.0093 K (−2.92%)**.

Patch-distribution (refine vs pretrained RMSE K): mean improvement 0.06359 K; fraction improved 0.9983 (`patch_distribution` in locked JSON).

Fine-tuned SwinIR on this split: **EXPERIMENT REQUIRED**.

---

## B. Physics refinement validation (Kerala, 414 held-out patches)

**Artifact:** `outputs/evaluation/physics_refinement_v1/validation/results.json`  
**Do not treat as locked test.** β was chosen using a disjoint 137-patch tune set from the same scene.

| Method | MAE (K) | RMSE (K) | PSNR | SSIM |
|--------|--------:|---------:|-----:|-----:|
| bicubic | 0.184102 | 0.354247 | 34.772 | 0.9016 |
| swinir_pretrained | 0.181629 | 0.340562 | 34.750 | 0.9009 |
| swinir_consistency_refined | 0.139085 | 0.289679 | 36.391 | 0.9246 |

Note: pretrained PSNR is **slightly worse** than bicubic on this validation subset (−0.022 dB) while Kelvin RMSE improves. Do not hide this.

β grid (`tune_beta.json`, 137 patches): RMSE K = 0.32752 (β=0), 0.31025 (0.25), 0.29522 (0.5), 0.2828 (0.75), **0.27334 (1.0)**.

---

## C. Multiscene validation (Kerala, 551 patches, no refine)

**Artifact:** `outputs/evaluation/multiscene_qa_v1/validation/results.json`

| Method | MAE (K) | RMSE (K) | PSNR | SSIM |
|--------|--------:|---------:|-----:|-----:|
| bicubic | 0.185339 | 0.350678 | 34.904 | 0.9020 |
| swinir_pretrained | 0.182808 | 0.337361 | 34.899 | 0.9014 |

551 ≠ 414: different patch subsets (all non-overlap vs held-out after β-tune). **Not interchangeable.**

---

## D. Development split — NOT paper test

**Artifact:** `outputs/evaluation/sr_baseline/results.json`  
Dataset: `processed` v0, 233 random-patch val, scene now in **train** of qa_v1. Spatial leakage likely.

| Method | MAE_z | RMSE_z | MAE (K) | RMSE (K) | PSNR | SSIM | pooled PSNR global range |
|--------|------:|------:|--------:|---------:|-----:|-----:|-------------------------:|
| Bicubic | 0.044929 | 0.064547 | 0.220222 | 0.316382 | 36.269 | 0.8878 | 45.59 |
| Pretrained | 0.044043 | 0.062523 | 0.215878 | 0.306459 | 36.512 | 0.8927 | 45.87 |
| Prelim FT | 0.040123 | 0.057596 | 0.196665 | 0.282311 | 37.249 | 0.9056 | 46.58 |

---

## E. Compute

**Parameters:** 904,744 (`results.json` `model.parameters`).  
**Disk:** official pth 17,147,989 B = 16.35 MiB; `best.pt` 24,613,323 B includes optimizer (not inference size).

**Locked-test forward_s / 1165 patches** (batch size 4, no warmup protocol; physics_refinement JSON):

| Method | total forward_s | s/patch |
|--------|----------------:|--------:|
| bicubic | 1.58503 | 0.001361 |
| swinir_pretrained | 2084.497 | 1.789 |
| swinir_consistency_refined (includes a SwinIR pass + refine) | 2134.865 | 1.833 |

Hardware: CPU `Intel64 Family 6 Model 140 Stepping 1`, 4 torch threads, `torch 2.14.1+cpu`. GPU memory: **NOT FOUND**. Peak inference RSS: **NOT FOUND**. Training-bench peak RSS 7907.8 MB is **not** inference.

Clean 40-patch timing (`timing_check_10_batches.json`, different split): bicubic 0.816 ms/patch; pretrained 1.318 s/patch; prelim FT 1.327 s/patch.

---

## F. Colorization / pipeline (not SR metrics)

No PSNR/SSIM for A/B/C. Runtime 15.9 s (colorization) / 17.4 s (integrated) for 4 patches. Kelvin standalone vs integrated: max |Δ| = 0.0 on all four patches (`final_audit_v1`).

HR-separation audit (refined vs reference Kelvin, not a quality metric): mean |Δ| 0.10–0.21 K on the four patches (`integration_report.txt`).
