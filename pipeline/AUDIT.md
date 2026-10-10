# Integration audit (Phase 1)

Code-backed interface summary for the integrated pipeline. See also
`outputs/integrated_pipeline_v1/integration_report.txt` after a validation run.

## Data flow

1. **ThermalSatelliteDataset** — normalized thermal + RGB + combined `valid_mask` (256×256, same grid).
2. **ThermalSRDataset** — synthetic LR via `degrade_thermal_patch` (scale 2, seed 42, σ=1.2, 0.05 K noise).
3. **SwinIR-S** — pretrained, thermal-adapted (`evaluation.evaluate_multiscene.build_methods`).
4. **Consistency refinement** — `sr_data.consistency_refinement`, β from `physics_refinement_v1/METHOD.txt`.
5. **Kelvin** — inverse z-score; saved as `thermal_kelvin_refined.npy` (separate from PNGs).
6. **Colorization** — `colorization.pipeline.apply_all_methods` on refined Kelvin + RGB reflectance.

HR arrays inside `ThermalSRDataset` are used only to **build synthetic LR** during degradation, not as the colorization thermal input (`colorization/thermal_predict.py`).

## Frozen parameters

- β = 1.0, degradation `base_seed=42`, colormap 295–320 K (matches colorization v1 validation).

## Known duplication (not removed in this task)

- `scripts/run_guided_colorization_v1.py` still orchestrates the same steps; use `pipeline.integrated` for E2E.

## Not verified by automated tests

- Cross-run byte identity with `outputs/colorization/guided_colorization_v1/` (same seed/patches should match; not asserted in CI).
