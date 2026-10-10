# Refinement and colorization audit

## 1. Refinement — what the code does

**Name in code:** `synthetic_forward_consistency_refinement` (`sr_data/consistency_refinement.py`, `METHOD.txt`).

**Correct scientific description:** a deterministic post-process that reduces disagreement between the SwinIR HR estimate and the **project’s own synthetic LR operator** (noise-free). It is **not** a TIRS PSF model, atmospheric correction, or radiative-transfer retrieval.

Module docstring: “Does not read reference HR pixels.” Signature: `refine_hr_normalized(swinir_z, lr_z, lr_valid, stats, config)` — no HR argument.

### Mathematics (Kelvin)

Let \(T_{\mathrm{SR}}\) be SwinIR output in Kelvin, \(T_{\mathrm{LR}}\) the observed synthetic LR, \(\beta\) the step size, \(\sigma=1.2\) the HR-pixel Gaussian sigma, \(s=2\) the scale.

1. Convert z-score tensors to Kelvin with the **scene** mean/std.  
2. Forward (noise-free): masked Gaussian blur of \(T_{\mathrm{SR}}\) then **area downsample** → \(\hat{T}_{\mathrm{LR}}\).  
3. Residual on valid LR pixels: \(r = T_{\mathrm{LR}} - \hat{T}_{\mathrm{LR}}\).  
4. Upsample \(r\) by **block replication** (adjoint of area downsample).  
5. Smooth with the same masked Gaussian.  
6. \(T_{\mathrm{out}} = T_{\mathrm{SR}} + \beta \cdot r_{\mathrm{smooth}}\) on finite pixels.  
7. Convert back to z-score for metrics / later stages.

β = 0 returns SwinIR unchanged. Frozen β = **1.0**.

### Hyperparameter selection

`tune_beta.json`: β ∈ {0, 0.25, 0.5, 0.75, 1.0} on **137** Kerala validation patches (fraction 0.25, seed 42). Lowest tune RMSE K at β=1.0 (0.27334). Locked test **not** used (log line: “no tuning on test”).

Limitation: single validation scene; 0.05 K degradation noise makes exact consistency impossible (`METHOD.txt`).

### What it was applied to

**Pretrained** 1-channel SwinIR, not a fine-tuned thermal checkpoint. `training_performed: false` in both physics_refinement result files.

It **does** change the numerical temperature array (that is the point). Locked-test pooled RMSE: 0.3085 K → 0.2404 K.

### Before / after (locked test, pooled Kelvin + patch-mean PSNR/SSIM)

See `ALL_VERIFIED_RESULTS.md` section A. Headline compatible numbers: PSNR 35.922 → 38.004 dB; RMSE 0.3085 → 0.2404 K.

### Limitations (code-backed)

- Matches the **synthetic** training degradation, not real TIRS (~100 m) physics.  
- No atmosphere, emissivity, or sensor MTF.  
- Correction lives in the blur’s low-pass band; it does not invent new high frequency from physics.  
- Cannot be renamed “validated physical model” without new evidence.

---

## 2. Colorization — what the code does

**Version:** `guided_colorization_v1` (`colorization/types.py`).  
**Orchestration:** `colorization/pipeline.py` `apply_all_methods` copies the Kelvin array and **raises** if it mutates.

Thermal input: `colorization/thermal_predict.py` — SwinIR(LR) → refine → denormalize. HR reference is not the colorization source.

RGB: SR_B2/B3/B4 surface reflectance, inverse z-score, same 30 m patch grid. Display scale: clip by `rgb_reflectance_display_max=0.25` (not sRGB).

Invalid pixels: RGBA (200,200,200,0) — transparent.

### Method A — thermal-only colormap

`method_standard_colormap`: matplotlib **inferno**, **fixed** 295–320 K (not per-image stretch). Visualization only.

### Method B — RGB-guided

Guided filter (He et al. box-filter implementation in `colorization/_boxfilter.py`) of colormap RGB using RGB **luminance** as guide. radius=8, ε=0.02. Mix weight `edge_mix * edge_rgb * (1 - edge_t)` capped at 0.35 so RGB edges are used more where thermal is already smooth. **Does not change Kelvin.**

### Method C — overlay

`0.55 * thermal_colormap + 0.45 * RGB_display`.

### Outputs

256×256 PNG per method + legends; Kelvin `.npy` stored separately. Prototype: 4 Kerala patches (`outputs/colorization/guided_colorization_v1/`). Integrated copies: `outputs/integrated_pipeline_v1/patches/...`.

### Metrics

Colorization is **excluded** from PSNR/SSIM/MAE/RMSE. Those scores are on thermal arrays only.

### Risks (stated in manifests)

- Optical edges need not be thermal boundaries.  
- RGB is 30 m reflectance, not 15 m thermal truth.  
- Fixed 295–320 K can clip or wash out scenes whose LST lies outside that window (Deccan mean ~324 K is **above** 320 K).  
- Appearance quality ≠ reconstruction accuracy.

---

## 3. Integrated path vs standalone

`pipeline/integrated.py` uses the same predict → colorize chain. `outputs/final_audit_v1`: Kelvin exact match; method PNG exact match; legend PNG mismatch due to title strings only.
