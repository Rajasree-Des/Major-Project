# Dataset and splits (verified)

## Product

**FACT:** Landsat Collection 2 Level-2 Science Products (Surface Temperature + Surface Reflectance), Landsat 8 and Landsat 9, downloaded from Microsoft Planetary Computer STAC (`landsat-c2-l2`).

Assets used: `ST_B10` (LST), `SR_B2` (blue), `SR_B3` (green), `SR_B4` (red), `QA_PIXEL` (qa_v1 only).

Grid: 30 m, UTM (scene CRS). Thermal and RGB are exported on the **same 30 m grid**. There is no native 15 m thermal band in this dataset.

## Radiometric conversion

Source: `preprocessing/radiometric.py` and per-scene `metadata.json`.

```
T_kelvin = DN_ST_B10 * 0.00341802 + 149.0
ρ       = DN_SR     * 0.0000275  + (-0.2)
```

Units: Kelvin (LST); unitless surface reflectance. Output dtype float32.

**FACT (QA audit):** reconstructed patch Kelvin vs raw DN conversion agrees to ~1.6×10⁻⁵ K (`outputs/evaluation/multiscene_qa_v1/audit/audit_report.txt`).

## Masking

### Original `dataset/processed/` (one scene)

Valid = finite and not NoData (DN 0) in ST_B10 and RGB. **QA_PIXEL flags were not applied.**

### QA-filtered `dataset/processed_qa/` (`processing_version: qa_v1`)

Valid = original rule **AND** none of QA_PIXEL bits 0–5 set:

| Bit | Flag | Role |
|-----|------|------|
| 0 | fill | no data |
| 1 | dilated_cloud | CFMask buffer |
| 2 | cirrus | OLI cirrus |
| 3 | cloud | cloud |
| 4 | cloud_shadow | shadow |
| 5 | snow | snow/ice excluded from this dataset |

Bits 6–7 (clear, water) and 8–15 (confidence pairs) are **not** exclusion rules. Water is kept. Source: `preprocessing/qa_mask.py`, USGS LSDS-1619; STAC bitfields verified in processing logs.

Invalid-pixel handling in SR: invalid LR pixels zero-filled (`prepare_lr`); metrics use `hr_valid_mask ∧ finite(HR)`.

## Normalization

Per-scene z-score on valid pixels. **Never pooled across scenes or splits.**

```
z = (x − mean_scene) / std_scene
x = z * std_scene + mean_scene
```

## Patch geometry

| Setting | Value | Source |
|---------|-------|--------|
| Patch size | 256×256 | `preprocessing/constants.py` |
| Stride (dataset build) | 128 | same (50% overlap) |
| Reject window | invalid fraction > 10% | metadata `max_invalid_percent` |
| SR HR / LR | 256 / 128 | scale 2 |
| Eval patch selection | **non-overlapping** (row, col both multiples of 256) | `evaluation/evaluate_multiscene.py` |

## Two dataset versions — do not mix

| Version | Path | Scenes | Accepted patches | Used by |
|---------|------|--------|------------------|---------|
| v0 NoData-only | `dataset/processed/LC09_L2SP_143050_20240327_02_T1` | 1 | **2330** (skipped 1092) | `sr_baseline`, preliminary SwinIR train, CPU benchmark |
| qa_v1 | `dataset/processed_qa/` | **7** | **15985** | split, multiscene, refinement, colorization, integrated, UI |

Same Landsat scene in both versions is **not** identical:

| | processed v0 | processed_qa |
|--|--------------|--------------|
| Patches | 2330 | 2318 |
| Valid pixels (scene) | 40,662,001 (68.39%) | 40,455,868 (68.05%) |
| Thermal mean / std | 316.784 / 4.902 K | 316.855 / 4.809 K |
| Extra exclusions | — | 206,133 pixels by QA |

## Seven QA scenes

Source: `dataset/splits/scene_split_qa_v1.json` + `outputs/dataset_expansion/qa_dataset_report.txt`.

| Scene ID | Sensor | Date | Region (split file) | Role | QA patches |
|----------|--------|------|---------------------|------|------------|
| LC09_L2SP_143050_20240327_02_T1 | L9 | 2024-03-27 | Southern Andhra Pradesh (reference) | train | 2318 |
| LC09_L2SP_150041_20250416_02_T1 | L9 | 2025-04-16 | Thar Desert, Rajasthan | train | 2327 |
| LC09_L2SP_144041_20231213_02_T1 | L9 | 2023-12-13 | Indo-Gangetic Plain, UP | train | 2325 |
| LC08_L2SP_146039_20241119_02_T1 | L8 | 2024-11-19 | Himalayan foothills, Uttarakhand | train | 2161 |
| LC08_L2SP_144053_20250225_02_T1 | L8 | 2025-02-25 | Kerala coast / Western Ghats | validation | 2202 |
| LC08_L2SP_144048_20240427_02_T1 | L8 | 2024-04-27 | Deccan Plateau, Telangana | test (locked) | 2322 |
| LC09_L2SP_137042_20230110_02_T1 | L9 | 2023-01-10 | Brahmaputra valley, Assam | test (locked) | 2330 |

**Role totals (all overlapping patches):** train 9131, validation 2202, test 4652, **sum 15985**.

Split SHA-256: `12bd1ff703b60732d23b7337c2d86d4eefabfd1542114d7c0ef33de2c6f40f7e`.

**Rules in the split file:** all patches of a scene share one role; test never used for selection; normalization is per scene; random patch splits must not be used for generalization claims.

## Synthetic degradation

`sr_data/metadata.py` `DegradationConfig`:

- Scale factor **2**
- Masked Gaussian blur, σ = **1.2 HR pixels**, in Kelvin
- Area downsample
- Additive Gaussian noise **0.05 K**
- `base_seed=42` (noise keyed by seed + scene-local patch id)

This defines a **synthetic reconstruction problem** against the 30 m Landsat thermal field. It does **not** emulate the native ~100 m TIRS IFOV or prove 15 m accuracy.

## Evaluation subsets (non-overlapping)

| Experiment | Role | Scenes | Patches scored |
|------------|------|--------|----------------|
| multiscene_qa_v1 validation | val | Kerala | **551** |
| physics_refinement_v1 β-tune | val (25%) | Kerala | **137** |
| physics_refinement_v1 validation | val held-out | Kerala | **414** (551−137) |
| multiscene + physics locked test | test | Deccan 582 + Assam 583 | **1165** |
| colorization / integrated | val samples | Kerala | **4** (ids 000096, 000448, 001006, 001116) |
| sr_baseline | random patch val | LC09 143/050 v0 | **233** |

Patch-key SHA-256 locked test (both evals): `b62dc6c4a55b1499139556fd5e19bc5e0358e20793817cee2ad2dac6e2f1d134`.

## What was acquired vs processed vs evaluated

- Acquired: 7 C2 L2 scenes (1 pre-existing + 6 in `download_log.txt`).  
- Processed qa_v1: 7/7.  
- Used as SwinIR **training data on QA split:** 0 scenes (no QA-split fine-tune).  
- Used in published-style locked test: 2 scenes.  
- Validation scoring: 1 scene.

No ASTER or other external HR thermal product was found in `dataset/`.
