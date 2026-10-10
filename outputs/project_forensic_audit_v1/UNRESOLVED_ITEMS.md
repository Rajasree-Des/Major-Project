# Unresolved items (ranked for publication)

## P0 — will mislead reviewers if unfixed in the PDF

1. **Fine-tuning is not a completed experiment.** Locked evals say `training_performed: false`. `best.pt` is a few CPU warmup steps on a leaked v0 patch split. **Action:** rewrite paper language; or run a real train on qa_v1 train scenes and a new locked eval (do not overwrite frozen JSON).  
2. **Tables II–IV as specified (four configs / FT ablation) cannot be filled honestly.** **Action:** relabel to pretrained+refine or omit FT rows.  
3. **“Physics-aware” vs synthetic consistency.** `METHOD.txt` forbids RT interpretation. **Action:** rename in title/abstract/method if the IEEE draft uses physical-sensor language.  
4. **15 m / ASTER / external GT.** No dataset, no metrics. **Action:** delete or move to future work.  
5. **`ThermalSR_IEEE.pdf` not in the repo.** Section-level wording could not be quoted. **Action:** place the PDF in the project and re-run a sentence-level audit.

## P1 — completeness / reproducibility

6. **Almost all scientific code and outputs are untracked** (git status vs `0542cd2`). Reviewers cannot clone the experiment from origin/main. **Action:** commit source (not raw TIFFs / 15 GB patches if too large); record checkpoint SHA.  
7. **`pretrained/` is gitignored.** Official pth must be re-downloadable; SHA is recorded — keep that SHA in the paper.  
8. **Table V memory.** No inference peak RSS or CUDA mem. Do not use training-bench 7907.8 MB. **Action:** short new bench in a new folder, or drop the column.  
9. **Eval figure PNGs.** `results.json` lists locked-test figure paths; a recursive `*.png` search under `outputs/evaluation/` returned **0 files**. **Action:** confirm whether figures were never written or live elsewhere before using Fig. 4.  
10. **Rasterio vs tifffile.** Evals used tifffile fallback; equivalence tested only on the reference scene (limitations arrays).

## P2 — scientific narrowness (honest, but weakens claims)

11. **Two locked-test scenes / one val scene.** Per-scene results are case studies.  
12. **Pretrained SwinIR ≈ bicubic.** PSNR +0.19 dB on test; **negative** PSNR delta on some validation subsets. Do not market a large learned-SR gain.  
13. **Colormap 295–320 K** vs Deccan mean ~324 K: visualization clipping on hot scenes.  
14. **Snow excluded** (QA bit 5) while a Himalaya train scene still contains mountain terrain — document the mask choice.  
15. **Train scenes unused for training.** The split reserves 9,131 train patches that were never used to update SwinIR in the frozen evals.

## P3 — software / demo (not paper-critical)

16. **UI:** one job at a time; no cancel; Scene Explorer shows reference previews not SR; UI metrics ≠ locked JSON (`THERMASCOPE_UI.md`).  
17. **Legend PNG regression** (title strings) — cosmetic.  
18. **Git index** still shows deleted placeholders (`evaluation`, `training`, …) vs live directories — cleanup when committing.  
19. **Cloud GPU guide unexecuted** in this working tree.

## Explicit EXPERIMENT REQUIRED cells

- Table II: Fine-tuned SwinIR (all four metrics)  
- Table II: Fine-tuned SwinIR + refinement (all four)  
- Table III: Fine-tuned row and Improvement  
- Table IV: if labels remain “Fine-tuned …”  
- Table V: fine-tuned size (weights-only), fine-tuned latency, all Memory entries, GPU identity  

## Integrity notes from this audit

- Frozen hashes: `outputs/final_audit_v1/frozen_hashes_evidence.json` (no drift vs baseline snapshot in that audit).  
- This forensic audit **did not** rewrite those files.  
- **This session’s pytest:** `py -m pytest` on listed unit modules (`-m "not slow"`) **did not complete**. Six files **ERROR during collection** with `OSError: [WinError 4551] An Application Control policy has blocked this file` (typical Windows blocking of a Torch/native DLL). That is an environment block, not an assertion failure. Do **not** treat this session as a green test run.  
- Prior audit (`outputs/final_audit_v1/FINAL_AUDIT_REPORT.md`) recorded `28 passed in ~90s` on 2026-10-09; that is **historical**, not re-confirmed here.  
- UI report `outputs/ui_integration_report_v1.txt` recorded 7 fast API tests + 1 CPU job test passed the same day.
