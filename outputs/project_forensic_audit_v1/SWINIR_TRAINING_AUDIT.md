# SwinIR training audit

This file separates five events that are easy to conflate.

| ID | Event | Occurred? |
|----|--------|-----------|
| A | Load official pretrained RGB SwinIR weights | **Yes** |
| B | Adapt architecture / tensors to 1 thermal channel | **Yes** |
| C | Run inference with the adapted network | **Yes** (all frozen evals) |
| D | Fine-tune on thermal data | **Only a few CPU warmup steps** on the old single-scene set; **not** on `processed_qa` train scenes |
| E | Evaluate a saved fine-tuned checkpoint on the locked test | **No** |

---

## Implementation and source

- Vendored official network: `models/swinir/network_swinir.py`  
  Header cites Jingyun Liang SwinIR, arXiv 2108.10257, GitHub blob `6545850fbf8df298df73d81f3e8cba638787c8bd`, Apache 2.0.  
  Local change: `timm` imports redirected to `models/swinir/_timm_compat.py`.
- Thermal wrapper: `models/swinir/model.py` (`ThermalSwinIR`, `in_channels=1`, `out_channels=1`).
- Checkpoint mapping: `models/swinir/pretrained.py` — every tensor accounted; **final `load_state_dict` is strict**.

Variant used everywhere in evals: **lightweight_x2** (SwinIR-S).

```
upscale=2, img_size=64, window_size=8, depths=[6,6,6,6], embed_dim=60,
num_heads=[6,6,6,6], mlp_ratio=2, upsampler=pixelshuffledirect, resi_connection=1conv
```

---

## A. Pretrained checkpoint

| Item | Value |
|------|--------|
| Filename | `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` |
| Path | `pretrained/swinir/` (gitignored) |
| URL | `https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` |
| Original domain | DIV2K **RGB** lightweight ×2 SR |
| SHA-256 | `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7` |
| Size | 17,147,989 bytes |
| State key | `params` |

Loading succeeded in every eval log examined (matched 327 tensors, 0 unexpected, 0 newly initialized).

---

## B. Channel adaptation

| Tensor | Mapping |
|--------|---------|
| `conv_first.weight` | (60, 3, 3, 3) → (60, 1, 3, 3) by **summing RGB input channels** (grey-as-RGB equivalent) |
| `upsample.0.weight` | (12, 60, 3, 3) → (4, 60, 3, 3) **mean over RGB output groups** (PixelShuffle order preserved) |
| `upsample.0.bias` | (12,) → (4,) same mean |

Adapted values: **2,704**. Copied unchanged: **902,040**. Total instantiated parameters: **904,744**. Attention masks regenerated (36 buffers).

---

## C. Inference with adapted pretrained weights

Frozen evals set `"training_performed": false` and limitations text: “Pretrained SwinIR weights are DIV2K RGB weights adapted to one channel; **no thermal fine-tuning**.”

This is the model behind `swinir_pretrained` and, with the consistency module, `swinir_consistency_refined`.

---

## D. What “fine-tuning” actually ran

### D1. Official-SwinIR preliminary run (the file people might call “fine-tuned”)

- Code: `training/train_swinir.py`  
- Checkpoint: `outputs/models/swinir/best.pt`  
  SHA-256 `f97a5fff48268bd129c5291274b06f13d240b2c612272221351f190b8dba6e1d`, 24,613,323 bytes (includes **optimizer_state_dict** per `training/checkpoint.py`).  
- `latest.pt`: SHA `0ed5c87a1d29b7f9222596db495ead6f7b95e4321f8131817cd0b3d3dee9d9ff`, 24,616,175 bytes.

Recorded in `sr_baseline/results.json` `finetuned_checkpoint`:

- epoch 2, batch_size 2, **max_train_batches 3**, **freeze_epochs 2**  
- train_loss 0.04064, val_loss 0.04251  
- data: `dataset/processed/LC09_L2SP_143050_20240327_02_T1` (v0, **not** qa_v1)  
- loss L1, default lr 1e-4, AdamW (script defaults), **CPU**

`set_warmup_freeze`: if `epoch <= freeze_epochs`, only `conv_first` / `upsample` are trainable. History has epochs 0 (eval only), 1, 2 with freeze_epochs=2 ⇒ **backbone never unfrozen in this run**. Upper bound on optimizer steps: 2 epochs × 3 batches = **6 steps** on adapted layers.

`history.json` epoch-0 (adapted pretrained, no FT): val MAE 0.04659 vs bicubic 0.04765 (z). After the tiny run, val MAE 0.04251. These numbers are **not** locked-test results.

### D2. CPU feasibility (10 steps)

`outputs/benchmarks/cpu_swinir/benchmark.json`: 10 optimizer steps, lr 1e-4, L1, batch 4, train_samples 1864 / val 233 (v0 random split). Checkpoints_untouched: true. This is **not** a saved fine-tune.

### D3. Custom/from-scratch (abandoned)

`outputs/models/swinir_custom_scratch/`. Two epochs in history. Workflow decision 2026-10-07: do not train from random init. **Not** used in qa_v1 evals.

### D4. Cloud GPU

`docs/CLOUD_GPU_TRAINING.md` describes Colab/Kaggle. **No** returned GPU checkpoint, no GPU eval JSON, no CUDA device in any `results.json` (`device: cpu`, `2.14.1+cpu`).

### D5. QA-split fine-tune

**Not found.** Four train scenes (9,131 patches) were never used to train SwinIR according to eval metadata (`training_performed: false`).

---

## E. Locked-test evaluation of a fine-tuned model

**EXPERIMENT REQUIRED.** No `results.json` scores `outputs/models/swinir/best.pt` on Deccan+Assam qa_v1.

Using D1 as “fine-tuned SwinIR” in Tables II–III would be incorrect because: (1) not converged, (2) backbone frozen, (3) wrong dataset version, (4) leaked patch split of a scene that is now **train**, (5) not the locked test.

---

## Training logs

There is no full-epoch training log for a QA-split fine-tune. Available: `history.json` (3 entries), `benchmark_log.txt` (10 lines), eval `run.log` files (inference only).
