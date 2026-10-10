"""Masked reconstruction metrics for thermal SR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dataset_loader.metadata import ChannelNormStats


@dataclass
class MetricBundle:
    mae: float
    rmse: float
    psnr: float
    ssim: float
    mae_kelvin: float | None = None
    rmse_kelvin: float | None = None


def _valid_values(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    return pred[valid], target[valid]


def compute_mae_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    p, t = _valid_values(pred, target, mask)
    if p.numel() == 0:
        return float("nan"), float("nan")
    mae = torch.mean(torch.abs(p - t)).item()
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    return mae, rmse


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _valid_values(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    mse = torch.mean((p - t) ** 2).item()
    if mse == 0:
        return float("inf")
    data_range = (t.max() - t.min()).item()
    if data_range <= 0:
        data_range = 1.0
    return float(10.0 * torch.log10(torch.tensor((data_range**2) / mse)).item())


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError("SSIM expects (B,1,H,W)")
    b = pred.shape[0]
    window = _gaussian_window(11, 1.5, pred.device)
    c1, c2 = 0.01**2, 0.03**2
    values: list[float] = []
    for i in range(b):
        p = pred[i : i + 1]
        t = target[i : i + 1]
        m = mask[i : i + 1].float()
        denom = F.conv2d(m, window, padding=5) + 1e-8
        mu_p = F.conv2d(p * m, window, padding=5) / denom
        mu_t = F.conv2d(t * m, window, padding=5) / denom
        sigma_p = F.conv2d(p * p * m, window, padding=5) / denom - mu_p**2
        sigma_t = F.conv2d(t * t * m, window, padding=5) / denom - mu_t**2
        sigma_pt = F.conv2d(p * t * m, window, padding=5) / denom - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
            (mu_p**2 + mu_t**2 + c1) * (sigma_p + sigma_t + c2) + 1e-8
        )
        valid = m.bool() & torch.isfinite(ssim_map)
        if valid.any():
            values.append(ssim_map[valid].mean().item())
    return float(sum(values) / len(values)) if values else float("nan")


def denormalize_kelvin(normalized: torch.Tensor, stats: ChannelNormStats) -> torch.Tensor:
    return normalized * stats.std + stats.mean


def compute_physical_kelvin_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    stats: ChannelNormStats,
) -> tuple[float, float]:
    pred_k = denormalize_kelvin(pred, stats)
    target_k = denormalize_kelvin(target, stats)
    return compute_mae_rmse(pred_k, target_k, mask)


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thermal_stats: ChannelNormStats | None = None,
) -> MetricBundle:
    mae, rmse = compute_mae_rmse(pred, target, mask)
    psnr = compute_psnr(pred, target, mask)
    ssim = compute_ssim(pred, target, mask)
    mae_k = rmse_k = None
    if thermal_stats is not None:
        mae_k, rmse_k = compute_physical_kelvin_errors(pred, target, mask, thermal_stats)
    return MetricBundle(mae=mae, rmse=rmse, psnr=psnr, ssim=ssim, mae_kelvin=mae_k, rmse_kelvin=rmse_k)


def aggregate_metrics(bundles: list[MetricBundle]) -> MetricBundle:
    def nanmean(vals: list[float | None]) -> float:
        clean = [v for v in vals if v is not None and v == v and abs(v) != float("inf")]
        return sum(clean) / len(clean) if clean else float("nan")

    return MetricBundle(
        mae=nanmean([b.mae for b in bundles]),
        rmse=nanmean([b.rmse for b in bundles]),
        psnr=nanmean([b.psnr for b in bundles]),
        ssim=nanmean([b.ssim for b in bundles]),
        mae_kelvin=nanmean([b.mae_kelvin for b in bundles]),
        rmse_kelvin=nanmean([b.rmse_kelvin for b in bundles]),
    )


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
