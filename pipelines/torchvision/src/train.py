from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .models import configure_stage
from .utils import atomic_write_csv, atomic_write_json, now_utc_iso, sha256_file


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best = math.inf
        self.bad = 0
        self.best_epoch = -1

    def update(self, value: float, epoch: int) -> bool:
        if value < self.best - 1e-12:
            self.best = float(value)
            self.bad = 0
            self.best_epoch = int(epoch)
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), target.reshape(-1))


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = _bce_loss(logits, y)
            total += float(loss.detach().cpu()) * len(y)
            count += len(y)
    return total / max(count, 1)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    effective_batch_size: int,
    use_amp: bool,
) -> float:
    """Train one epoch while preserving the configured effective batch size.

    Gradients are accumulated as *summed* per-sample BCE contributions and then
    normalized by the actual number of samples in each optimizer step. This is
    exact even when the final accumulation group is smaller than the nominal
    effective batch size.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total = 0.0
    count = 0
    group_samples = 0
    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n = int(len(y))
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            # Sum reduction lets us normalize each optimizer step by the true
            # accumulated sample count, including a short final group.
            loss_sum = nn.functional.binary_cross_entropy_with_logits(
                logits.reshape(-1), y.reshape(-1), reduction="sum"
            )
        scaler.scale(loss_sum).backward()
        group_samples += n
        total += float(loss_sum.detach().cpu())
        count += n

        is_last = (step + 1) == len(loader)
        if group_samples >= effective_batch_size or is_last:
            # DataLoader micro-batches never exceed the effective batch size in
            # official configs; therefore a group cannot overshoot except by a
            # configuration error.
            if group_samples > effective_batch_size and not is_last:
                raise RuntimeError(
                    f"Accumulated {group_samples} samples before optimizer step; "
                    f"expected <= effective batch {effective_batch_size}."
                )
            inv = 1.0 / max(group_samples, 1)
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(inv)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    return total / max(count, 1)


def train_stage(
    model: nn.Module,
    architecture: str,
    stage: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    max_epochs: int,
    lr: float,
    patience: int,
    effective_batch_size: int,
    micro_batch_size: int,
    use_amp: bool,
) -> Dict[str, object]:
    if effective_batch_size % micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    accumulation_steps = effective_batch_size // micro_batch_size
    stage_info = configure_stage(model, architecture, stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=lr)
    stopper = EarlyStopper(patience)
    best_path = run_dir / f"stage{stage}_best.pt"
    history = []
    start = time.time()
    for epoch in range(int(max_epochs)):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch + stage * 1000)
        tr = _train_epoch(model, train_loader, optimizer, device, effective_batch_size, use_amp)
        va = evaluate_loss(model, val_loader, device, use_amp)
        improved = stopper.update(va, epoch)
        elapsed_epoch = time.time() - start
        print(
            f"[TRAIN] {architecture} stage={stage} epoch={epoch+1}/{max_epochs} "
            f"train_loss={tr:.6f} val_loss={va:.6f} "
            f"{'BEST' if improved else ''} elapsed_stage={elapsed_epoch/60:.1f} min",
            flush=True,
        )
        history.append(
            {
                "stage": stage,
                "epoch": epoch + 1,
                "train_loss": tr,
                "val_loss": va,
                "improved": improved,
                "learning_rate": lr,
                "micro_batch_size": micro_batch_size,
                "effective_batch_size": effective_batch_size,
                "accumulation_steps": accumulation_steps,
            }
        )
        if improved:
            _atomic_torch_save(
                {
                    "state_dict": model.state_dict(),
                    "stage": stage,
                    "epoch": epoch + 1,
                    "val_loss": va,
                    "architecture": architecture,
                    "stage_info": stage_info,
                },
                best_path,
            )
        if stopper.should_stop:
            break
    if not best_path.exists():
        raise RuntimeError(f"Stage {stage} produced no checkpoint")
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    result = {
        "stage": stage,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_val_loss": float(ckpt["val_loss"]),
        "selected_epoch": int(ckpt["epoch"]),
        "epochs_executed": len(history),
        "duration_seconds": time.time() - start,
        "stage_info": stage_info,
        "history": history,
    }
    atomic_write_json(run_dir / f"stage{stage}_summary.json", result)
    atomic_write_csv(run_dir / f"stage{stage}_history.csv", pd.DataFrame(history))
    return result


def select_best_stage(stage1: Dict[str, object], stage2: Dict[str, object]) -> Dict[str, object]:
    if float(stage1["best_val_loss"]) <= float(stage2["best_val_loss"]):
        return {"selected_stage": 1, **stage1}
    return {"selected_stage": 2, **stage2}


def train_two_stage(
    model: nn.Module,
    architecture: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str | Path,
    head_epochs: int = 8,
    finetune_epochs: int = 20,
    patience: int = 4,
    head_lr: float = 1e-4,
    finetune_lr: float = 1e-5,
    effective_batch_size: int = 64,
    micro_batch_size: int = 64,
    use_amp: bool = True,
) -> Dict[str, object]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = now_utc_iso()
    stage1 = train_stage(
        model, architecture, 1, train_loader, val_loader, device, run_dir,
        head_epochs, head_lr, patience, effective_batch_size, micro_batch_size, use_amp,
    )
    # Stage 2 explicitly starts from the best Stage-1 checkpoint.
    ckpt1 = torch.load(stage1["best_checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt1["state_dict"])
    stage2 = train_stage(
        model, architecture, 2, train_loader, val_loader, device, run_dir,
        finetune_epochs, finetune_lr, patience, effective_batch_size, micro_batch_size, use_amp,
    )
    best = select_best_stage(stage1, stage2)
    selected_path = Path(best["best_checkpoint"])
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    summary = {
        "started_utc": started,
        "finished_utc": now_utc_iso(),
        "stage1": stage1,
        "stage2": stage2,
        "selected_stage": int(best["selected_stage"]),
        "selected_epoch": int(best["selected_epoch"]),
        "selected_val_loss": float(best["best_val_loss"]),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
    }
    atomic_write_json(run_dir / "training_summary.json", summary)
    return summary


def make_loader(dataset, micro_batch_size: int, shuffle: bool, num_workers: int, collate_fn, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(micro_batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # ensures dataset epoch updates reach freshly spawned workers
        generator=g,
        collate_fn=collate_fn,
    )


def suggested_micro_batch_sizes(effective_batch_size: int = 64) -> list[int]:
    return [b for b in [64, 32, 16, 8, 4, 2, 1] if effective_batch_size % b == 0 and b <= effective_batch_size]
