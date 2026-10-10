MULTI-SCENE QA_V1 BASELINE EVALUATION
============================================================
Synthetic x2 reconstruction against the 30 m Landsat thermal reference; not evidence of real 15 m thermal accuracy.

split_manifest.json  frozen scene-level split (copy of dataset/splits/scene_split_qa_v1.json)
audit/               read-only dataset audit
validation/          baseline on validation scene(s)
test_locked/         LOCKED - one-time baseline; do not use for model selection

--- validation/report.txt ---
MULTI-SCENE BASELINE EVALUATION - role: validation
==============================================================================
Synthetic x2 reconstruction against the 30 m Landsat thermal reference; not evidence of real 15 m thermal accuracy.
split manifest: C:\Users\HP\Downloads\ISRO project\Major-Project\dataset\splits\scene_split_qa_v1.json
scenes: LC08_L2SP_144053_20250225_02_T1
patch selection: non_overlapping (551 patches)

PER SCENE (Kelvin metrics use each scene's own mean/std)
LC08_L2SP_144053_20250225_02_T1: patches scored 551/551 (zero-valid excluded 0), valid px 35,917,778, thermal std 4.895 K
  bicubic            MAE 0.1853 K  RMSE 0.3507 K  PSNR 34.904 dB  SSIM 0.9020  (z: MAE 0.03786, RMSE 0.07164)
  swinir_pretrained  MAE 0.1828 K  RMSE 0.3374 K  PSNR 34.899 dB  SSIM 0.9014  (z: MAE 0.03735, RMSE 0.06892)
  swinir_pretrained - bicubic: MAE -0.0025 K, RMSE -0.0133 K, PSNR -0.004 dB, SSIM -0.0006

AGGREGATE (does not replace the per-scene table)
  bicubic            macro RMSE 0.3507 K [0.3507-0.3507], pooled RMSE 0.3507 K, pooled MAE 0.1853 K
  swinir_pretrained  macro RMSE 0.3374 K [0.3374-0.3374], pooled RMSE 0.3374 K, pooled MAE 0.1828 K, scenes better than bicubic (RMSE K): 1/1
  Macro means weight every scene equally; pooled Kelvin metrics weight scenes by valid pixels. z-space metrics are not pooled across scenes because each scene has its own std.

sanity checks passed: True
training performed: False

LIMITATIONS
  - Synthetic degradation (Gaussian blur sigma 1.2 HR px, area downsample, 0.05 K noise) against the 30 m reference; real 100 m TIRS acquisition physics and real 15 m truth are not represented.
  - Few scenes: 1 validation and 2 test scenes; per-scene results are case studies, not population estimates.
  - Degradation noise is keyed by seed and scene-local patch ID, so equal patch IDs in different scenes share a noise field (identical for every method).
  - Per-scene z-score normalization: z-space metrics scale with 1/std and SSIM constants are fixed in z-units, so only Kelvin metrics and per-patch PSNR are comparable across scenes.
  - Pretrained SwinIR weights are DIV2K RGB weights adapted to one channel; no thermal fine-tuning.
  - Raster reading used the tifffile fallback; rasterio-vs-tifffile equivalence was tested only on the reference scene (see audit).

--- test_locked/report.txt ---
MULTI-SCENE BASELINE EVALUATION - role: test
==============================================================================
Synthetic x2 reconstruction against the 30 m Landsat thermal reference; not evidence of real 15 m thermal accuracy.
LOCKED - one-time baseline; do not use for model selection
split manifest: C:\Users\HP\Downloads\ISRO project\Major-Project\dataset\splits\scene_split_qa_v1.json
scenes: LC08_L2SP_144048_20240427_02_T1, LC09_L2SP_137042_20230110_02_T1
patch selection: non_overlapping (1165 patches)

PER SCENE (Kelvin metrics use each scene's own mean/std)
LC08_L2SP_144048_20240427_02_T1: patches scored 582/582 (zero-valid excluded 0), valid px 38,081,751, thermal std 4.974 K
  bicubic            MAE 0.2717 K  RMSE 0.3796 K  PSNR 36.278 dB  SSIM 0.8797  (z: MAE 0.05462, RMSE 0.07632)
  swinir_pretrained  MAE 0.2656 K  RMSE 0.3677 K  PSNR 36.533 dB  SSIM 0.8842  (z: MAE 0.05340, RMSE 0.07391)
  swinir_pretrained - bicubic: MAE -0.0061 K, RMSE -0.0120 K, PSNR +0.255 dB, SSIM +0.0045
LC09_L2SP_137042_20230110_02_T1: patches scored 583/583 (zero-valid excluded 0), valid px 38,115,193, thermal std 2.868 K
  bicubic            MAE 0.1536 K  RMSE 0.2405 K  PSNR 35.182 dB  SSIM 0.8587  (z: MAE 0.05355, RMSE 0.08386)
  swinir_pretrained  MAE 0.1526 K  RMSE 0.2349 K  PSNR 35.315 dB  SSIM 0.8589  (z: MAE 0.05321, RMSE 0.08191)
  swinir_pretrained - bicubic: MAE -0.0010 K, RMSE -0.0056 K, PSNR +0.133 dB, SSIM +0.0002

AGGREGATE (does not replace the per-scene table)
  bicubic            macro RMSE 0.3101 K [0.2405-0.3796], pooled RMSE 0.3178 K, pooled MAE 0.2126 K
  swinir_pretrained  macro RMSE 0.3013 K [0.2349-0.3677], pooled RMSE 0.3085 K, pooled MAE 0.2091 K, scenes better than bicubic (RMSE K): 2/2
  Macro means weight every scene equally; pooled Kelvin metrics weight scenes by valid pixels. z-space metrics are not pooled across scenes because each scene has its own std.

sanity checks passed: True
training performed: False

LIMITATIONS
  - Synthetic degradation (Gaussian blur sigma 1.2 HR px, area downsample, 0.05 K noise) against the 30 m reference; real 100 m TIRS acquisition physics and real 15 m truth are not represented.
  - Few scenes: 1 validation and 2 test scenes; per-scene results are case studies, not population estimates.
  - Degradation noise is keyed by seed and scene-local patch ID, so equal patch IDs in different scenes share a noise field (identical for every method).
  - Per-scene z-score normalization: z-space metrics scale with 1/std and SSIM constants are fixed in z-units, so only Kelvin metrics and per-patch PSNR are comparable across scenes.
  - Pretrained SwinIR weights are DIV2K RGB weights adapted to one channel; no thermal fine-tuning.
  - Raster reading used the tifffile fallback; rasterio-vs-tifffile equivalence was tested only on the reference scene (see audit).
