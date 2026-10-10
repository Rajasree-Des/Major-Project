# Publication update guide (evidence-only)

The IEEE PDF was not in the repository. Use this as a section-by-section rewrite checklist. Replacement sentences are **only** what the artifacts support.

## Title / abstract

**Currently implied:** a physics-aware deep model that super-resolves and colorizes thermal satellite imagery, with fine-tuning and a four-way ablation.

**Evidence:** adapted DIV2K SwinIR-S, synthetic ×2 vs 30 m LST, consistency correction, visualization colorization.

**Change:** make the synthetic protocol and the pretrained (not fine-tuned) backbone explicit.

**Suggested wording:**

> We study ×2 thermal super-resolution as reconstruction of Landsat Collection 2 Level-2 30 m surface temperature from synthetically degraded 128×128 patches. A lightweight SwinIR model pretrained on DIV2K is adapted to one channel and evaluated without thermal-domain convergence training. A deterministic correction enforces consistency with the same blur-and-area-downsample operator used to form the low-resolution input. On a locked two-scene test (1,165 non-overlapping patches), the pretrained network reduces pooled RMSE from 0.3178 K (bicubic) to 0.3085 K; adding the consistency correction further reduces RMSE to 0.2404 K (PSNR 35.73 / 35.92 / 38.00 dB). RGB-guided colorization is a visualization overlay and does not alter the Kelvin field. Results are not a demonstration of native 15 m thermal accuracy.

**Safe numbers:** those three RMSE/PSNR triples. **Unsafe:** any fine-tune Δ, 15 m, ASTER, GPU-hours.

---

## Introduction and contributions

**Change:** list contributions that exist: QA Landsat dataset + scene split; official SwinIR adaptation; locked-test bicubic vs pretrained; consistency refinement ablation; visualization methods; reproducibility artifacts.

**Do not list:** converged thermal FT; four learned configurations; external HR benchmark; operational 15 m product.

---

## Related work

No local numbers. Keep SwinIR citation (Liang et al., 2021) consistent with vendored commit `6545850`.

---

## Proposed system

**Evidence:** `pipeline/AUDIT.md` flow.  
**Change:** draw Fig. 1 from that flow; label refinement “synthetic forward consistency.”  
**Safe figure inputs:** none required beyond a schematic.

---

## Dataset and preprocessing

**Use:** 7 C2 L2 scenes; formulas `T=DN×0.00341802+149`, `ρ=DN×2.75e-5−0.2`; QA bits 0–5; 256/128 patches; 15,985 qa_v1 patches; split table in `DATASET_AND_SPLITS.md`.

**Do not** cite 2,330 as the multi-scene N (that is v0, one scene).

---

## Synthetic degradation

**Keep:** σ=1.2 HR px, area downsample, 0.05 K noise, seed 42, scale 2.  
**Must say:** HR is 30 m Landsat LST, not sub-30 m truth.

---

## Transfer learning and fine-tuning

**What to claim:**

> Official SwinIR-S ×2 DIV2K weights (file `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`) are loaded strictly after summing RGB input channels of `conv_first` and averaging PixelShuffle RGB groups in the upsampler (2,704 adapted parameters; 902,040 copied). All reported test metrics use this adapted pretrained network.

**What to delete:** “we fine-tune SwinIR on Landsat thermal until convergence.”  
**Optional appendix only:** 6 CPU warmup steps on a leaked single-scene split, checkpoint `outputs/models/swinir/best.pt`, explicitly non-comparable.

---

## Refinement

**Replace “physics-aware / radiative transfer” with:**

> After SwinIR inference, we convert the estimate to Kelvin, apply the noise-free synthetic forward operator (masked Gaussian blur with σ=1.2 followed by area 2× downsampling), form the LR residual, upsample by replication, smooth with the same Gaussian, and add the residual with β=1.0. β was selected on 137 Kerala validation patches and frozen before the locked test.

---

## Colorization

State methods A/B/C, inferno 295–320 K, guided radius 8 / ε 0.02 / edge_mix 0.35, overlay 0.55. State Kelvin unchanged. State RGB-edge risk. Do not assign PSNR to colorized images.

**Safe figure files:**

- `outputs/colorization/guided_colorization_v1/patches/patch_000096/A_standard_colormap.png` (and B, C)  
- same for 000448, 001006, 001116  
- integrated copies under `outputs/integrated_pipeline_v1/patches/LC08_L2SP_144053_20250225_02_T1/`

These are **validation** patches, not locked test.

---

## External benchmark

**Remove** until an ASTER (or similar) experiment exists. Future work is the honest location.

---

## Computing setup

> Experiments were run on CPU (Intel 11th-generation Core, CPUID Family 6 Model 140) with PyTorch 2.14.1 CPU build, four intra-op threads, batch size 4. No CUDA device was used for the frozen evaluations. Mean forward time on the 1,165 locked-test patches was 1.36 ms (bicubic), 1.789 s (SwinIR), and 1.833 s (SwinIR + consistency), including the extra SwinIR pass inside the refine context, excluding data loading.

Do not claim GPU training time. Do not invent memory. Model size: 16.35 MB disk for the official `.pth`; 904,744 parameters. Bicubic: N/A size/params.

---

## Tables

### Table I (architecture) — can complete now

Use the JSON `model.config` block and 904,744 params.

### Table II — complete only two rows, or retitle

| Method | PSNR (dB) | SSIM | MAE (K) | RMSE (K) |
|--------|----------:|-----:|--------:|---------:|
| Bicubic interpolation | 35.729 | 0.8694 | 0.2126 | 0.3178 |
| Pretrained SwinIR (DIV2K, 1-ch adapted) | 35.922 | 0.8717 | 0.2091 | 0.3085 |
| Fine-tuned SwinIR | — | — | — | — |
| Fine-tuned + refinement | — | — | — | — |

Footnote: MAE/RMSE pooled Kelvin; PSNR/SSIM mean over 1,165 patches; synthetic ×2; locked Deccan+Assam.

**If the paper must show three methods:** replace rows 3–4 with a single row “Pretrained SwinIR + consistency refinement | 38.004 | 0.9104 | 0.1591 | 0.2404” and **delete** the fine-tuned labels.

### Table III — cannot complete

Do not fill from `sr_baseline`. Better to omit until a QA-split fine-tune exists.

### Table IV — complete if relabeled

| Method | PSNR | SSIM | MAE (K) | RMSE (K) |
|--------|-----:|-----:|--------:|---------:|
| Pretrained SwinIR | 35.922 | 0.8717 | 0.2091 | 0.3085 |
| + consistency refinement (β=1.0) | 38.004 | 0.9104 | 0.1591 | 0.2404 |
| Improvement | +2.081 dB | +0.0386 | −23.91% (−0.0500 K) | −22.06% (−0.0681 K) |

### Table V — partial

| Configuration | Model size | Parameters | Inference | Memory |
|---------------|------------|------------|-----------|--------|
| Bicubic | N/A | N/A | 1.36 ms/patch | NOT MEASURED |
| Pretrained SwinIR | 16.35 MB (disk) | 904,744 | 1.789 s/patch | NOT MEASURED (CPU-only) |
| Fine-tuned | EXPERIMENT REQUIRED | (same arch.) | EXPERIMENT REQUIRED | NOT MEASURED |
| + refinement | 0 extra learned params | 904,744 | 1.833 s/patch (pretrained+refine) | NOT MEASURED |

---

## Results / discussion / conclusion

Report per-scene RMSE as well as pooled (Deccan 0.368→0.277 K; Assam 0.235→0.197 K for pretrained→refine). Admit pretrained PSNR gain is small and can reverse on validation. Limitations paragraph can be copied almost verbatim from locked `results.json` `limitations`.

**Conclusion that is allowed:** consistency correction substantially improves synthetic reconstruction relative to adapted SwinIR; the learned model alone is close to bicubic in Kelvin RMSE.

**Conclusion that is not allowed:** ThermalSR achieves 15 m thermal SR; fine-tuning is shown to help; RGB guidance improves thermal accuracy.

---

## Missing experiments before stronger claims (priority)

1. Fine-tune official SwinIR on the four **train** qa_v1 scenes (val early-stopping only).  
2. Locked-test eval of that checkpoint ± frozen β=1.0 (new output dir).  
3. Dedicated latency/memory bench.  
4. Optional: real 15 m / ASTER protocol — only then change the accuracy claim.
