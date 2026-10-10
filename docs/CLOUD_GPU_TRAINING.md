# Fine-tuning thermal SwinIR on a free cloud GPU (Colab / Kaggle)

This guide runs the **official pretrained SwinIR (SwinIR-S, x2, DIV2K), adapted and fine-tuned
for Landsat thermal imagery** on a free GPU. No local NVIDIA GPU is required; the same commands run
on CPU, only slower.

The model consumes the existing processed Landsat patches and the existing synthetic x2 degradation
(`sr_data.ThermalSRDataset`). Nothing here downloads or builds a different dataset.

---

## 1. Prepare the processed dataset (on your own machine)

The processed patches are not in git. Zip one scene directory, including `metadata.json`,
`thermal/` and `rgb/`. The dataset indexes thermal/RGB pairs, so both folders are needed even though
SR uses only thermal. About 2.3 GB for `LC09_L2SP_143050_20240327_02_T1`.

```powershell
Compress-Archive -Path dataset\processed\LC09_L2SP_143050_20240327_02_T1 -DestinationPath processed_LC09_143050.zip
```

- **Colab:** upload the zip to Google Drive (e.g. `MyDrive/thermal/processed_LC09_143050.zip`).
- **Kaggle:** create a private Kaggle Dataset from the zip (Kaggle extracts it automatically).

---

## 2. Start the runtime

- **Colab:** Runtime → Change runtime type → GPU (T4 is fine).
- **Kaggle:** Notebook settings → Accelerator → GPU T4 / P100, Internet → On.

---

## 3. Install

```bash
git clone https://github.com/Rajasree-Des/Major-Project.git
cd Major-Project
pip install -r requirements.txt
```

The preinstalled PyTorch on Colab/Kaggle satisfies `torch>=2.0.0` and is kept. The SwinIR network
is vendored in `models/swinir/network_swinir.py`; `timm` is not required.

## 4. Verify PyTorch and CUDA

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"{p.total_memory / 1024**3:.1f} GiB")
```

The training script prints the same information at startup.

---

## 5. Make the dataset available

**Colab**

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
mkdir -p dataset/processed
unzip -q /content/drive/MyDrive/thermal/processed_LC09_143050.zip -d dataset/processed/
ls dataset/processed/LC09_L2SP_143050_20240327_02_T1   # metadata.json  rgb/  thermal/
```

Unzip to local disk (`/content`); reading 2330 small `.npy` files directly from Drive is slow.

**Kaggle**: the dataset is mounted read-only under `/kaggle/input/<dataset-name>/`; point
`--data-dir` at the directory containing `metadata.json`, e.g.
`/kaggle/input/processed-lc09-143050/LC09_L2SP_143050_20240327_02_T1`.

---

## 6. Pretrained checkpoint

`--pretrained-checkpoint auto` (the default) downloads
`002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth` (16 MB) from the
[official SwinIR release](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) into
`pretrained/swinir/`. Without internet, download it manually and pass its path:

```bash
--pretrained-checkpoint /path/to/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
```

At startup the log reports exactly how the RGB checkpoint was mapped to 1 channel:

```
Pretrained parameters matched:  327 tensors / 902040 values (copied unchanged)
Adapted parameters:             3 tensors / 2704 values
    conv_first.weight  (60, 3, 3, 3) -> (60, 1, 3, 3)  [sum over RGB input channels ...]
    upsample.0.weight  (12, 60, 3, 3) -> (4, 60, 3, 3) [mean over RGB output groups ...]
    upsample.0.bias  (12,) -> (4,)                     [mean over RGB output groups ...]
Newly initialized parameters:   0 tensors / 0 values
Unexpected checkpoint tensors:  0 (ignored)
```

---

## 7. Smoke test, then a short experiment

Always run the smoke test first (forward/backward, finite gradients, optimizer step, checkpoint
round-trip):

```bash
python -m training.train_swinir --smoke-test-only \
  --data-dir dataset/processed/LC09_L2SP_143050_20240327_02_T1
```

Short fine-tuning run (a few epochs, full train/val split, checkpoints to Drive so they survive a
disconnect):

```bash
python -m training.train_swinir \
  --data-dir dataset/processed/LC09_L2SP_143050_20240327_02_T1 \
  --epochs 5 --batch-size 8 --learning-rate 1e-4 --num-workers 2 \
  --checkpoint-dir /content/drive/MyDrive/thermal/swinir_run1
```

Useful options:

| Option | Purpose |
|--------|---------|
| `--freeze-epochs N` | First N epochs train only the adapted `conv_first` / `upsample` layers, then full fine-tuning |
| `--max-train-batches`, `--max-val-batches` | Cap batches per epoch for quick checks |
| `--device cpu` / `cuda` | Override auto selection |
| `--no-save-viz` | Skip validation figures |
| `--seed` | Seed for Python/NumPy/PyTorch, the patch split and the degradation |

If a T4 runs out of memory, lower `--batch-size` (4 is safe).

Each run logs a pre-fine-tuning evaluation (epoch 0), then per epoch: masked L1 train/val loss,
and SwinIR vs bicubic MAE / RMSE / PSNR / SSIM plus Kelvin MAE / RMSE on the same validation
patches and masks.

---

## 8. Resume after a disconnect

```bash
python -m training.train_swinir \
  --data-dir dataset/processed/LC09_L2SP_143050_20240327_02_T1 \
  --epochs 10 --batch-size 8 \
  --checkpoint-dir /content/drive/MyDrive/thermal/swinir_run1 \
  --resume /content/drive/MyDrive/thermal/swinir_run1/latest.pt
```

`--epochs` is the total target; training continues from the epoch after the one stored in
`latest.pt`, restoring model, optimizer and best validation loss. `best.pt` is only replaced when
validation loss improves.

---

## 9. Retrieve results

Outputs in `--checkpoint-dir`:

- `best.pt`, `latest.pt`: model and optimizer state, epoch, losses, model/training/degradation
  config, seed, and pretrained-checkpoint provenance (URL, sha256, adaptation summary)
- `history.json`: per-epoch losses and SwinIR/bicubic metrics
- `figures/epoch_XXX.png`: LR, bicubic, SwinIR, HR reference, absolute error

With Drive as the checkpoint dir, they sync automatically. Otherwise:

```python
from google.colab import files
files.download("outputs/models/swinir/best.pt")
```

On Kaggle, files written under `/kaggle/working/` appear in the notebook's Output tab.

Load locally:

```python
from models.swinir import ThermalSwinIR, ThermalSwinIRConfig
from training.checkpoint import load_checkpoint
import torch

payload = torch.load("best.pt", map_location="cpu", weights_only=False)
model = ThermalSwinIR(ThermalSwinIRConfig.from_dict(payload["model_config"]))
load_checkpoint("best.pt", model)
```

---

## 10. Limitations: report these with any results

- The HR target is the **30 m Landsat thermal reference**; the LR input is produced by
  **synthetic degradation** of it. Metrics are **synthetic SR reconstruction metrics**, not
  evidence of true 15 m thermal accuracy.
- With one scene, the split is a **random patch split within the scene**
  (`WARNING: SINGLE-SCENE DEVELOPMENT SPLIT — NOT FINAL SCIENTIFIC EVALUATION`); neighbouring
  patches overlap, so validation is not independent.
- Final evaluation needs multiple Landsat scenes with scene-level train/validation/test
  separation and diverse conditions.
- The pretrained weights come from a generic RGB image-SR model (DIV2K). The model is a
  pretrained SwinIR *adapted and fine-tuned* for thermal imagery, not a pretrained thermal model.
