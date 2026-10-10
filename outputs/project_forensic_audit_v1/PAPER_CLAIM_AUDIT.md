# IEEE manuscript claim audit

## Manuscript file

**`ThermalSR_IEEE.pdf` was not found** in `Major-Project`, parent `ISRO project`, or a filename search under the user Downloads tree. This matrix therefore cannot quote Abstract/Introduction sentences. It audits:

1. The stated title.  
2. The table/row structure the author specified for Tables II–V in the prior publication-data request.  
3. The section list and ten “pay particular attention” claims in the forensic-audit brief.  
4. What the repository can actually support.

Status key: `SUPPORTED` | `PARTIALLY SUPPORTED` | `UNSUPPORTED` | `CONTRADICTED BY EVIDENCE` | `NOT VERIFIABLE` (missing PDF wording).

| Paper section or claim | What is currently asserted (from title, table spec, or typical draft claims) | Evidence in the project | Status | Required correction |
|---|---|---|---|---|
| Title: “Physics-Aware Deep Learning Framework” | Physics-aware DL for SR + guided colorization | DL = adapted SwinIR (pretrained). “Physics” module is synthetic forward **consistency**, explicitly not RT (`METHOD.txt`). | `PARTIALLY SUPPORTED` | Qualify: data-driven SR + **synthetic degradation consistency**; do not imply sensor-physics inversion. |
| Title / abstract: thermal SR of satellite imagery | Super-resolution of Landsat thermal | Yes, synthetic ×2 vs 30 m ST_B10. | `PARTIALLY SUPPORTED` | State synthetic ×2 / 30 m reference; forbid “15 m accuracy”. |
| Abstract headline result | Unknown exact sentence | Best honest headline from locked test: consistency refine vs pretrained, RMSE 0.3085→0.2404 K, PSNR 35.92→38.00 dB, **pretrained** base. Pretrained vs bicubic is only +0.19 dB PSNR. | `NOT VERIFIABLE` (wording) / likely `CONTRADICTED` if it cites fine-tuning gains | Rewrite headline from locked JSON; do not cite fine-tune Δ. |
| Introduction / contributions: four-method ablation | Bicubic, pretrained, fine-tuned, fine-tuned+refine | Only first two + pretrained+refine exist on locked test. | `CONTRADICTED BY EVIDENCE` | Drop or relabel rows; or run missing FT experiment. |
| Related work | Unknown | n/a | `NOT VERIFIABLE` | Keep literature claims independent of local numbers. |
| Proposed system: SwinIR backbone | SwinIR-S ×2 lightweight | Vendored official net + 1-ch wrap. | `SUPPORTED` | Name official checkpoint and adaptation. |
| Dataset: Landsat C2 L2 | L8/L9 ST + RGB | 7 scenes documented. | `SUPPORTED` | Use qa_v1 counts (15985), not v0 2330 as the multi-scene N. |
| Preprocessing: QA + Kelvin | DN→K, masks | `radiometric.py`, `qa_mask.py`, audit vs raw. | `SUPPORTED` | Distinguish v0 NoData-only vs qa_v1. |
| Synthetic degradation | ×2 blur/downsample | `DegradationConfig` as specified. | `SUPPORTED` | Keep σ=1.2, 0.05 K, seed 42. |
| Transfer learning | Load DIV2K RGB weights, adapt 1-ch | `pretrained.py` strict load. | `SUPPORTED` | Describe sum/mean tensor map. |
| Fine-tuning completed | Thermal-domain FT | Eval JSON: no FT; `best.pt` is 6 warmup steps on v0 leaked split. | `CONTRADICTED BY EVIDENCE` | Say adapted **zero-shot** pretrained; optional mention of preliminary non-converged run in appendix only. |
| Refinement as physics / RT | Physics-aware residual | Consistency with synthetic forward only. | `CONTRADICTED BY EVIDENCE` if claimed as RT | Use “synthetic forward-consistency correction (β=1.0)”. |
| Colorization | Guided + thermal-only | A/B/C implemented; Kelvin unchanged. | `SUPPORTED` as visualization | Do not treat as accuracy contribution. |
| External Landsat/ASTER benchmark | Often planned in such papers | **No** ASTER/GT dataset or eval folder. | `UNSUPPORTED` | Remove or mark future work. |
| Computing setup: local CPU + cloud GPU | Guide exists | All `results.json` device **cpu**, torch **+cpu**. Cloud FT **not** evidenced. | `PARTIALLY SUPPORTED` | Report Intel Family 6 Model 140, CPU-only evals; call Colab guide **unexecuted** unless a GPU ckpt is produced. |
| Architecture table (Table I typical) | SwinIR-S params | 904,744 params, config in JSON. | `SUPPORTED` | Use those hyperparameters. |
| Experimental setup: scene split, locked test | qa_v1 split | Manifest + SHA recorded; test unused for β. | `SUPPORTED` | Cite scene IDs and 1165 patches. |
| Table II all four rows | See prior table spec | Rows 1–2 yes; 3–4 no. | `PARTIALLY SUPPORTED` | Complete only bicubic + pretrained; or retitle last row to pretrained+refine (then it is Table IV duplicated). |
| Table III FT effect | Pretrained vs FT | No compatible pair. | `UNSUPPORTED` | Delete or EXPERIMENT REQUIRED. |
| Table IV refine effect on **fine-tuned** model | FT vs FT+refine | Measured pair is **pretrained** vs pretrained+refine. | `CONTRADICTED BY EVIDENCE` if labeled FT | Relabel base model. |
| Table V compute | size, params, time, memory | Size/params/time (eval forward_s) yes; GPU mem no; bicubic size N/A. | `PARTIALLY SUPPORTED` | Memory column EXPERIMENT REQUIRED or omit; do not use 7.9 GB training RSS. |
| Fig. 1–3 system/dataset | Unknown files | Pipeline diagrams can be redrawn from `pipeline/AUDIT.md`. | `NOT VERIFIABLE` | Use real module names. |
| Fig. 4 experimental outputs | “actual experimental outputs rather than illustrations” | Colorization PNGs exist for 4 val patches; locked-test `results.json` lists figure filenames but this audit’s glob found **0 PNG** under `outputs/evaluation/` (paths listed, files not confirmed present). UI/integrated PNGs exist for demos. | `PARTIALLY SUPPORTED` | Use existing `outputs/colorization/.../A|B|C_*.png` and any eval figures if restored; do not use mock UI screenshots as science. |
| Results: generalization | Multi-scene | 2 test scenes only; pretrained gain small and scene-dependent. | `PARTIALLY SUPPORTED` | Call them case studies. |
| Results: pretrained ≫ bicubic | Possible draft claim | Locked test +0.19 dB PSNR, −2.9% RMSE; val PSNR can be slightly **worse**. | `PARTIALLY SUPPORTED` | Report Kelvin RMSE + both PSNR directions. |
| Discussion/limitations | If they match code notes | Code already documents synthetic ×2, few scenes, no FT, consistency ≠ RT. | `SUPPORTED` **if** the PDF matches those notes | Copy limitations from frozen `results.json` `limitations` arrays. |
| Conclusion: framework validated at 15 m | If claimed | Contradicted by every eval `label`. | `CONTRADICTED BY EVIDENCE` if present | Restrict to synthetic 30 m reconstruction. |
| Website as contribution | If claimed as deployed system | Local FastAPI+Vite, real jobs, frozen JSON charts. | `PARTIALLY SUPPORTED` | “Local demonstration UI”, not operational ISRO service. |

### Ten requested checkpoints

1. **Thermal fine-tuning completed:** **No** (`CONTRADICTED BY EVIDENCE`).  
2. **Four-configuration ablation:** **No** as written.  
3. **Tables II–IV completable now:** II rows 1–2 yes; III no; IV only if relabeled to pretrained.  
4. **Table V measurements:** params + disk size + CPU s/patch yes; memory **no**.  
5. **Refinement described accurately:** only if called synthetic consistency.  
6. **Abstract headline:** only valid if it uses pretrained+refine on synthetic ×2, not FT or 15 m.  
7. **Fig. 4 real outputs:** colorization/UI PNGs real; eval-set figure files not confirmed on disk.  
8. **External Landsat/ASTER benchmark:** **not evaluated**.  
9. **Computing setup CPU/GPU:** CPU evals evidenced; cloud GPU **not**.  
10. **Dataset version / split:** paper must use `processed_qa` + `scene_split_qa_v1`, not v0 233-patch val.
