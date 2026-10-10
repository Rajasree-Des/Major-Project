"""CPU-only feasibility benchmark: 10 optimizer steps of thermal SwinIR-S x2 fine-tuning.

Run from the project root:  py -m outputs.benchmarks.cpu_swinir.run_benchmark  (or: py outputs/benchmarks/cpu_swinir/run_benchmark.py)
Writes only to outputs/benchmarks/cpu_swinir/. Epoch times are extrapolated ESTIMATES.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models.swinir.model import ThermalSwinIR, config_for_variant  # noqa: E402
from models.swinir.utils import count_parameters  # noqa: E402
from sr_data.metadata import DegradationConfig  # noqa: E402
from training.losses import build_loss  # noqa: E402
from training.metrics import bicubic_upsample, evaluate_batch  # noqa: E402
from training.repro import set_seed  # noqa: E402
from training.train_swinir import build_datasets, collate_sr, parse_args, prepare_lr, resolve_pretrained  # noqa: E402

OUT = ROOT / "outputs" / "benchmarks" / "cpu_swinir"
CKPTS = [ROOT / "outputs/models/swinir/best.pt", ROOT / "outputs/models/swinir/latest.pt"]
N_BATCHES, N_VAL_BATCHES = 10, 2
BATCH_LIMIT_S, TOTAL_LIMIT_S = 180.0, 1200.0


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def process_ram_mb() -> tuple[float | None, float | None]:
    """(current, peak) working set in MiB; None on non-Windows."""
    if os.name != "nt":
        return None, None
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20


def system_ram_gb() -> tuple[float | None, float | None]:
    if os.name != "nt":
        return None, None
    ms = _MS()
    ms.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullTotalPhys / 2**30, ms.ullAvailPhys / 2**30


def ckpt_state() -> dict[str, tuple[int, int] | None]:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in CKPTS}


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    before = ckpt_state()
    ram_total, ram_avail = system_ram_gb()

    args = parse_args(["--data-dir", "dataset/processed/LC09_L2SP_143050_20240327_02_T1", "--device", "cpu"])
    set_seed(args.seed)
    model = ThermalSwinIR(config_for_variant(args.variant))
    resolve_pretrained(model, args.pretrained_checkpoint, args.variant)
    degradation = DegradationConfig(scale=args.scale, base_seed=args.seed)
    full_ds, train_ds, val_ds = build_datasets(args, degradation)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_sr, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = build_loss(args.loss)
    print(f"config: batch={args.batch_size} lr={args.learning_rate} wd={args.weight_decay} workers={args.num_workers} "
          f"loss={args.loss} seed={args.seed} params={count_parameters(model)} train={len(train_ds)} "
          f"val={len(val_ds)} threads={torch.get_num_threads()}", flush=True)

    records: list[dict] = []
    status = "complete"
    model.train()
    t0 = time.perf_counter()
    batches = iter(loader)
    for i in range(N_BATCHES):
        t_data = time.perf_counter()
        batch = next(batches)
        t_compute = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(prepare_lr(batch["lr"])), batch["hr"], batch["hr_valid_mask"])
        loss.backward()
        optimizer.step()
        t_end = time.perf_counter()
        rss, _ = process_ram_mb()
        rec = {"batch": i + 1, "loss": loss.item(), "data_s": t_compute - t_data,
               "compute_s": t_end - t_compute, "total_s": t_end - t_data, "rss_mb": rss}
        records.append(rec)
        print(f"batch {i + 1:2d} loss={rec['loss']:.6f} data={rec['data_s']:.2f}s compute={rec['compute_s']:.2f}s "
              f"total={rec['total_s']:.2f}s rss={rss or float('nan'):.0f}MB", flush=True)
        if rec["total_s"] > BATCH_LIMIT_S or t_end - t0 > TOTAL_LIMIT_S:
            status = f"stopped early after {i + 1} batches (safety limit)"
            break
    wall = time.perf_counter() - t0

    model.eval()
    val_times: list[float] = []
    vloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_sr)
    stats = next(iter(full_ds.base_dataset.metadata_by_scene.values())).thermal_norm
    with torch.no_grad():
        for j, vb in enumerate(vloader):
            if j >= N_VAL_BATCHES:
                break
            tv = time.perf_counter()
            vlr = prepare_lr(vb["lr"])
            evaluate_batch(model(vlr), vb["hr"], vb["hr_valid_mask"], stats)
            evaluate_batch(bicubic_upsample(vlr, args.scale), vb["hr"], vb["hr_valid_mask"], stats)
            val_times.append(time.perf_counter() - tv)

    n = len(records)
    steady = records[1:] if n > 1 else records
    avg_steady = sum(r["total_s"] for r in steady) / len(steady)
    train_bpe = -(-len(train_ds) // args.batch_size)
    val_bpe = -(-len(val_ds) // args.batch_size)
    avg_val = sum(val_times) / len(val_times) if val_times else float("nan")
    epoch_train_s = train_bpe * avg_steady
    epoch_s = epoch_train_s + val_bpe * avg_val
    _, peak = process_ram_mb()

    result = {
        "label": "CPU-ONLY FEASIBILITY BENCHMARK (10 optimizer steps); epoch times are ESTIMATES",
        "status": status,
        "device": "cpu",
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "system_ram_gb": ram_total,
        "system_ram_available_gb_at_start": ram_avail,
        "config": {"batch_size": args.batch_size, "learning_rate": args.learning_rate,
                   "weight_decay": args.weight_decay, "num_workers": args.num_workers, "loss": args.loss,
                   "seed": args.seed, "variant": args.variant, "pretrained": args.pretrained_checkpoint,
                   "train_samples": len(train_ds), "val_samples": len(val_ds)},
        "batches_run": n,
        "wall_clock_s": wall,
        "avg_s_per_batch": wall / n,
        "avg_s_per_batch_excl_first": avg_steady,
        "batches_per_s": n / wall,
        "samples_per_s": n * args.batch_size / wall,
        "avg_data_s": sum(r["data_s"] for r in records) / n,
        "avg_compute_s": sum(r["compute_s"] for r in records) / n,
        "initial_loss": records[0]["loss"],
        "final_loss": records[-1]["loss"],
        "peak_process_ram_mb": peak,
        "final_process_ram_mb": records[-1]["rss_mb"],
        "val_batches_timed": len(val_times),
        "avg_val_batch_s": avg_val,
        "estimates": {
            "note": "ESTIMATES extrapolated from steady-state batch time (excl. first batch) and 2 timed val batches",
            "train_batches_per_epoch": train_bpe,
            "val_batches_per_epoch": val_bpe,
            "epoch_train_only_min": epoch_train_s / 60,
            "epoch_with_val_min": epoch_s / 60,
            **{f"{e}_epochs_h": e * epoch_s / 3600 for e in (1, 5, 10, 20)},
        },
        "checkpoints_untouched": before == ckpt_state(),
        "per_batch": records,
    }
    (OUT / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_batch"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
