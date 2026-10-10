"""Fine-tune official pretrained SwinIR (adapted to 1 thermal channel) on synthetic x2 SR patches."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset_loader.split import (
    DEVELOPMENT_SPLIT_LABEL,
    SINGLE_SCENE_SPLIT_WARNING,
    development_patch_split,
    list_processed_scene_ids,
    require_single_scene_for_patch_split,
)
from models.swinir.model import ThermalSwinIR, config_for_variant
from models.swinir.pretrained import (
    VARIANT_CHOICES,
    download_official_checkpoint,
    load_pretrained_swinir,
    official_checkpoint_url,
)
from models.swinir.utils import count_parameters, count_trainable_parameters
from sr_data.dataset import ThermalSRDataset
from sr_data.metadata import SCIENTIFIC_LIMITATIONS, DegradationConfig
from training.checkpoint import load_checkpoint, save_checkpoint, save_training_history
from training.losses import build_loss
from training.metrics import MetricBundle, aggregate_metrics, bicubic_upsample, evaluate_batch
from training.repro import set_seed
from training.visualize import save_validation_panel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "models" / "swinir"

SINGLE_SCENE_WARNING_BANNER = "WARNING: SINGLE-SCENE DEVELOPMENT SPLIT — NOT FINAL SCIENTIFIC EVALUATION"
METRICS_DISCLAIMER = (
    "Reported metrics are synthetic SR reconstruction metrics against the 30 m Landsat thermal "
    "reference after synthetic degradation. They do not demonstrate true 15 m thermal accuracy."
)
WARMUP_TRAINABLE_PREFIXES = ("net.conv_first.", "net.upsample.")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def collate_sr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lr": torch.stack([b["lr"] for b in batch]),
        "hr": torch.stack([b["hr"] for b in batch]),
        "lr_valid_mask": torch.stack([b["lr_valid_mask"] for b in batch]),
        "hr_valid_mask": torch.stack([b["hr_valid_mask"] for b in batch]),
        "patch_id": [b["patch_id"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return device


def log_device(device: torch.device) -> None:
    logger.info("Device: %s", device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        logger.info("GPU: %s | memory %.1f GiB", props.name, props.total_memory / 1024**3)
    else:
        logger.info("CUDA available: %s (running on CPU)", torch.cuda.is_available())


def prepare_lr(lr: torch.Tensor) -> torch.Tensor:
    """Invalid LR pixels are NaN in the dataset; the network needs finite input."""
    return torch.nan_to_num(lr, nan=0.0)


def set_warmup_freeze(model: ThermalSwinIR, frozen: bool) -> None:
    """Freeze everything except the RGB->thermal adapted layers (``frozen=True``), or unfreeze all."""
    for name, param in model.named_parameters():
        param.requires_grad = (not frozen) or name.startswith(WARMUP_TRAINABLE_PREFIXES)


def smoke_test(model: ThermalSwinIR, device: torch.device, scale: int, batch_size: int = 2) -> None:
    """Forward/backward/step/checkpoint round-trip on a copy of the model (original untouched)."""
    probe = copy.deepcopy(model).to(device)
    probe.train()
    x = torch.randn(batch_size, 1, 128, 128, device=device)
    y = probe(x)
    expected = (batch_size, 1, 128 * scale, 128 * scale)
    if tuple(y.shape) != expected:
        raise RuntimeError(f"Smoke test: output shape {tuple(y.shape)} != {expected}")
    if not torch.isfinite(y).all():
        raise RuntimeError("Smoke test: output contains NaN/Inf")

    target = torch.randn_like(y)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, :, : y.shape[-2] // 2] = False
    loss = build_loss("l1")(y, target, mask)
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke test: masked L1 loss is not finite")
    empty_loss = build_loss("l1")(y, target, torch.zeros_like(mask))
    if not torch.isfinite(empty_loss):
        raise RuntimeError("Smoke test: masked L1 with zero valid pixels is not finite")

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in probe.parameters() if p.requires_grad]
    if not grads or any(g is None for g in grads):
        raise RuntimeError("Smoke test: some trainable parameters received no gradient")
    if not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Smoke test: non-finite gradients")
    optimizer.step()

    probe.eval()
    with torch.no_grad():
        ref = probe(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        save_checkpoint(
            path,
            {
                "model_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 0,
                "train_loss": loss.item(),
                "val_loss": float("nan"),
                "model_config": probe.config.to_dict(),
                "training_config": {},
                "seed": None,
                "pretrained": None,
            },
        )
        reloaded = ThermalSwinIR(copy.deepcopy(probe.config)).to(device)
        load_checkpoint(path, reloaded, map_location=device)
        reloaded.eval()
        with torch.no_grad():
            if not torch.allclose(ref, reloaded(x), atol=1e-6):
                raise RuntimeError("Smoke test: reloaded checkpoint gives different output")
    logger.info(
        "Smoke test passed: %s -> %s, finite output, masked L1=%.4f (empty-mask L1=%.1f), "
        "finite gradients, optimizer step, checkpoint save/load round-trip",
        tuple(x.shape),
        tuple(y.shape),
        loss.item(),
        abs(empty_loss.item()),
    )


@torch.no_grad()
def run_validation(
    model: ThermalSwinIR,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    thermal_stats,
    scale: int,
    *,
    max_batches: int | None,
    save_viz_path: Path | None = None,
) -> tuple[float, MetricBundle, MetricBundle]:
    model.eval()
    losses: list[float] = []
    swinir_metrics: list[MetricBundle] = []
    bicubic_metrics: list[MetricBundle] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        lr = prepare_lr(batch["lr"].to(device))
        hr = batch["hr"].to(device)
        mask = batch["hr_valid_mask"].to(device)

        pred = model(lr)
        loss = loss_fn(pred, hr, mask)
        if torch.isfinite(loss):
            losses.append(loss.item())

        swinir_metrics.append(evaluate_batch(pred, hr, mask, thermal_stats))
        bic = bicubic_upsample(lr, scale)
        bicubic_metrics.append(evaluate_batch(bic, hr, mask, thermal_stats))

        if batch_idx == 0 and save_viz_path is not None:
            save_validation_panel(
                lr[0, 0],
                pred[0, 0],
                hr[0, 0],
                mask[0, 0],
                save_viz_path,
                title=f"{batch['scene_id'][0]} / {batch['patch_id'][0]} (synthetic x{scale} SR, development split)",
            )

    val_loss = sum(losses) / len(losses) if losses else float("nan")
    return val_loss, aggregate_metrics(swinir_metrics), aggregate_metrics(bicubic_metrics)


def log_metrics(label: str, m: MetricBundle) -> None:
    logger.info(
        "Val %-8s MAE=%.5f RMSE=%.5f PSNR=%.2f SSIM=%.4f | MAE_K=%.4f RMSE_K=%.4f",
        label,
        m.mae,
        m.rmse,
        m.psnr,
        m.ssim,
        m.mae_kelvin if m.mae_kelvin is not None else float("nan"),
        m.rmse_kelvin if m.rmse_kelvin is not None else float("nan"),
    )


def _jsonable(args: argparse.Namespace) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def resolve_pretrained(model: ThermalSwinIR, spec: str, variant: str) -> dict[str, Any] | None:
    """Initialise ``model`` from the official checkpoint according to ``spec`` ('auto' | 'none' | path)."""
    if spec.lower() == "none":
        logger.warning("--pretrained-checkpoint none: random initialisation (ablation only, not the project baseline)")
        return None
    if spec.lower() == "auto":
        path = download_official_checkpoint(variant)
        url: str | None = official_checkpoint_url(variant)
    else:
        path = Path(spec)
        url = None
    report = load_pretrained_swinir(model, path, source_url=url)
    report.log()
    return {"filename": Path(report.checkpoint_path).name, **report.summary()}


def build_datasets(args: argparse.Namespace, degradation: DegradationConfig):
    data_dir = args.data_dir.resolve()
    full_ds = ThermalSRDataset(data_dir, degradation=degradation)
    require_single_scene_for_patch_split(list(full_ds.base_dataset.metadata_by_scene))
    split = development_patch_split(len(full_ds), seed=args.seed)
    train_ds = ThermalSRDataset(data_dir, degradation=degradation, patch_indices=split.train_indices)
    val_ds = ThermalSRDataset(data_dir, degradation=degradation, patch_indices=split.validation_indices)
    return full_ds, train_ds, val_ds


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    log_device(device)

    scene_count = len(list_processed_scene_ids(args.data_dir)) or 1
    if scene_count <= 1:
        logger.warning(SINGLE_SCENE_WARNING_BANNER)
        logger.warning(SINGLE_SCENE_SPLIT_WARNING)
    logger.warning(DEVELOPMENT_SPLIT_LABEL)
    logger.info("Scientific context: %s", SCIENTIFIC_LIMITATIONS)
    logger.info(METRICS_DISCLAIMER)

    model_cfg = config_for_variant(args.variant)
    if model_cfg.upscale != args.scale:
        raise ValueError(f"Variant '{args.variant}' is x{model_cfg.upscale}, but --scale={args.scale}")
    model = ThermalSwinIR(model_cfg)

    resume_payload: dict[str, Any] | None = None
    if args.resume is not None:
        resume_payload = load_checkpoint(args.resume, model, map_location="cpu")
        pretrained_info = resume_payload.get("pretrained")
        logger.info("Resuming from %s (epoch %s); skipping pretrained initialisation", args.resume, resume_payload.get("epoch"))
    else:
        pretrained_info = resolve_pretrained(model, args.pretrained_checkpoint, args.variant)
    model.to(device)
    logger.info("Model: official SwinIR (%s), %d parameters", args.variant, count_parameters(model))

    smoke_test(model, device, args.scale, batch_size=min(2, args.batch_size))
    if args.smoke_test_only:
        return {"smoke_test": "passed", "device": str(device), "parameters": count_parameters(model)}

    degradation = DegradationConfig(scale=args.scale, base_seed=args.seed)
    full_ds, train_ds, val_ds = build_datasets(args, degradation)
    logger.info("Dataset: %d patch pairs | train %d | val %d", len(full_ds), len(train_ds), len(val_ds))

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "collate_fn": collate_sr,
                     "pin_memory": device.type == "cuda"}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = build_loss(args.loss)

    scene_metas = full_ds.base_dataset.metadata_by_scene
    if len(scene_metas) > 1:
        logger.warning("Kelvin metrics use the first scene's normalization stats; per-scene stats not yet applied.")
    thermal_stats = next(iter(scene_metas.values())).thermal_norm

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    viz_dir = ckpt_dir / "figures"
    history_path = ckpt_dir / "history.json"
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    start_epoch = 1

    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_val = float(resume_payload.get("best_val_loss", resume_payload.get("val_loss", float("inf"))))
        if history_path.is_file():
            history = [r for r in json.loads(history_path.read_text(encoding="utf-8")) if r["epoch"] < start_epoch]

    if start_epoch == 1:
        bl, bs, bb = run_validation(model, val_loader, device, loss_fn, thermal_stats, args.scale,
                                    max_batches=args.max_val_batches)
        logger.info("Before fine-tuning (pretrained, adapted, epoch 0): val_loss=%.6f", bl)
        log_metrics("SwinIR", bs)
        log_metrics("Bicubic", bb)
        history.append({"epoch": 0, "train_loss": None, "val_loss": bl,
                        "swinir_metrics": bs.__dict__, "bicubic_metrics": bb.__dict__})

    for epoch in range(start_epoch, args.epochs + 1):
        warmup = epoch <= args.freeze_epochs
        set_warmup_freeze(model, frozen=warmup)
        logger.info(
            "Epoch %d/%d: %s | trainable %d / %d parameters",
            epoch,
            args.epochs,
            "warmup (only adapted conv_first/upsample trainable)" if warmup else "full fine-tuning",
            count_trainable_parameters(model),
            count_parameters(model),
        )

        model.train()
        train_losses: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            lr = prepare_lr(batch["lr"].to(device, non_blocking=True))
            hr = batch["hr"].to(device, non_blocking=True)
            mask = batch["hr_valid_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(lr), hr, mask)
            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite loss at batch %d", batch_idx)
                continue
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % args.log_every == 0:
                logger.info("  batch %d loss=%.6f", batch_idx, loss.item())

        train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")
        val_loss = float("nan")
        swinir_m = bic_m = None

        if epoch % args.val_every == 0 or epoch == args.epochs:
            viz_path = viz_dir / f"epoch_{epoch:03d}.png" if args.save_viz else None
            val_loss, swinir_m, bic_m = run_validation(
                model, val_loader, device, loss_fn, thermal_stats, args.scale,
                max_batches=args.max_val_batches, save_viz_path=viz_path,
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "model_config": model_cfg.to_dict(),
                "training_config": _jsonable(args),
                "degradation_config": degradation.to_dict(),
                "seed": args.seed,
                "split_label": DEVELOPMENT_SPLIT_LABEL,
                "pretrained": pretrained_info,
            }
            save_checkpoint(ckpt_dir / "latest.pt", payload)
            if improved:
                save_checkpoint(ckpt_dir / "best.pt", payload)
                logger.info("New best val_loss=%.6f -> %s", val_loss, ckpt_dir / "best.pt")

        logger.info("Epoch %d/%d train_loss=%.6f val_loss=%.6f", epoch, args.epochs, train_loss, val_loss)
        if swinir_m and bic_m:
            log_metrics("SwinIR", swinir_m)
            log_metrics("Bicubic", bic_m)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "swinir_metrics": swinir_m.__dict__ if swinir_m else None,
            "bicubic_metrics": bic_m.__dict__ if bic_m else None,
        })
        save_training_history(history_path, history)

    return {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "device": str(device),
        "parameters": count_parameters(model),
        "checkpoints": str(ckpt_dir),
        "scene_count": scene_count,
        "best_val_loss": best_val,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune official pretrained SwinIR, adapted to 1-channel thermal, on synthetic x2 SR patches."
    )
    p.add_argument("--data-dir", "--processed-root", dest="data_dir", type=Path, default=Path("dataset/processed"),
                   help="Processed scene directory, or dataset/processed for all scenes")
    p.add_argument("--pretrained-checkpoint", default="auto",
                   help="'auto' downloads the official checkpoint, a path loads a local file, 'none' = random init")
    p.add_argument("--variant", default="lightweight_x2", choices=VARIANT_CHOICES)
    p.add_argument("--resume", type=Path, default=None, help="Resume from a latest.pt/best.pt training checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="Warmup epochs training only the adapted conv_first/upsample layers (0 = full fine-tuning)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="'auto' (CUDA if available, else CPU), 'cpu', 'cuda', 'cuda:0'")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--loss", default="l1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--save-viz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke-test-only", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        summary = train(args)
        print("\n=== Training Summary ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
