"""
Train RestorationUNet on i4C paired NoisyLR (128) -> GT (256).

Examples:
  python train.py --sanity_check
  python train.py --epochs 30 --batch_size 8
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.dataset import build_datasets, paired_filenames
from src.losses import CombinedL1SSIMLoss
from src.metrics import compute_all, clip01
from src.model import RestorationUNet, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path | None) -> dict:
    defaults = {
        "seed": 42,
        "data_root": ".",
        "val_ratio": 0.1,
        "model": {"base_ch": 32},
        "train": {
            "epochs": 30,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "lambda_ssim": 0.5,
            "log_csv": "results/train_log.csv",
            "checkpoint": "weights/best.pt",
        },
        "sanity": {"n_pairs": 2, "steps": 200, "lr": 1e-3},
    }
    if path is None or not path.is_file():
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    # shallow-merge top-level dict sections
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(defaults.get(k), dict):
            defaults[k] = {**defaults[k], **v}
        else:
            defaults[k] = v
    return defaults


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    with_lpips: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        pred = clip01(model(noisy))
        gt_c = clip01(gt)
        # per-image metrics then average
        bsz = pred.shape[0]
        for i in range(bsz):
            m = compute_all(pred[i], gt_c[i], device=device, with_lpips=with_lpips)
            sums["psnr"] += m["psnr"]
            sums["ssim"] += m["ssim"]
            if with_lpips and "lpips" in m and m["lpips"] == m["lpips"]:
                sums["lpips"] += m["lpips"]
            n += 1
    if n == 0:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def bicubic_baseline(
    loader: DataLoader,
    device: torch.device,
    with_lpips: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Bicubic upsample NoisyLR to 256, clip to [0,1], compare to GT (no denoise)."""
    sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        noisy = batch["noisy"].to(device)
        gt = clip01(batch["gt"].to(device))
        up = F.interpolate(noisy, size=(256, 256), mode="bicubic", align_corners=False)
        up = clip01(up)
        bsz = up.shape[0]
        for i in range(bsz):
            m = compute_all(up[i], gt[i], device=device, with_lpips=with_lpips)
            sums["psnr"] += m["psnr"]
            sums["ssim"] += m["ssim"]
            if with_lpips and "lpips" in m and m["lpips"] == m["lpips"]:
                sums["lpips"] += m["lpips"]
            n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def run_sanity_check(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    data_root = Path(args.data_root)
    all_names = paired_filenames(data_root / "train" / "GT", data_root / "train" / "NoisyLR")
    n_pairs = int(cfg["sanity"]["n_pairs"])
    names = all_names[:n_pairs]
    train_ds, _, _, _ = build_datasets(data_root, sanity_names=names)
    loader = DataLoader(train_ds, batch_size=n_pairs, shuffle=True, num_workers=0)

    model = RestorationUNet(base_ch=cfg["model"]["base_ch"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["sanity"]["lr"]))
    criterion = CombinedL1SSIMLoss(lambda_ssim=float(cfg["train"]["lambda_ssim"]))

    steps = int(cfg["sanity"]["steps"])
    print(f"[sanity] overfitting on {names} for {steps} steps | device={device}")
    print(f"[sanity] params={count_parameters(model):,}")
    model.train()
    losses = []
    t0 = time.perf_counter()
    step = 0
    while step < steps:
        for batch in loader:
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(noisy)
            out = criterion(pred, gt)
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"].item()))
            if step % 20 == 0 or step == steps - 1:
                print(
                    f"  step {step:4d}/{steps}  loss={losses[-1]:.6f}  "
                    f"l1={float(out['l1']):.6f}  ssim={float(out['ssim']):.6f}"
                )
            step += 1
            if step >= steps:
                break
    dt = time.perf_counter() - t0
    print(f"[sanity] done in {dt:.1f}s | first_loss={losses[0]:.6f} last_loss={losses[-1]:.6f}")
    if losses[-1] < losses[0] * 0.5:
        print("[sanity] PASS: loss dropped substantially — pipeline looks healthy.")
    else:
        print("[sanity] WARN: loss did not drop much — check learning rate / data / model.")


def train_full(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    data_root = Path(args.data_root)
    train_ds, val_ds, train_names, val_names = build_datasets(
        data_root, val_ratio=float(args.val_ratio), seed=int(args.seed)
    )
    print(f"train={len(train_names)} val={len(val_names)} device={device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = RestorationUNet(base_ch=cfg["model"]["base_ch"]).to(device)
    print(f"model params={count_parameters(model):,}")
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    criterion = CombinedL1SSIMLoss(lambda_ssim=float(args.lambda_ssim))

    log_path = Path(args.log_csv)
    ckpt_path = Path(args.checkpoint)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Baseline once at start
    print("Computing bicubic upsample baseline on val set...")
    baseline = bicubic_baseline(val_loader, device, with_lpips=not args.no_lpips)
    print(
        f"BASELINE bicubic | PSNR={baseline['psnr']:.4f}  "
        f"SSIM={baseline['ssim']:.4f}  LPIPS={baseline['lpips']:.4f}"
    )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_psnr",
                "val_ssim",
                "val_lpips",
                "baseline_psnr",
                "baseline_ssim",
                "baseline_lpips",
                "lr",
                "seconds",
            ]
        )
        writer.writerow(
            [
                0,
                "",
                "",
                "",
                "",
                f"{baseline['psnr']:.6f}",
                f"{baseline['ssim']:.6f}",
                f"{baseline['lpips']:.6f}",
                args.lr,
                "",
            ]
        )

    best_psnr = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(noisy)
            out = criterion(pred, gt)
            out["loss"].backward()
            opt.step()
            running += float(out["loss"].item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        val_m = evaluate(model, val_loader, device, with_lpips=not args.no_lpips)
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.6f}  "
            f"PSNR={val_m['psnr']:.4f}  SSIM={val_m['ssim']:.4f}  "
            f"LPIPS={val_m['lpips']:.4f}  ({dt:.1f}s)"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_m['psnr']:.6f}",
                    f"{val_m['ssim']:.6f}",
                    f"{val_m['lpips']:.6f}",
                    f"{baseline['psnr']:.6f}",
                    f"{baseline['ssim']:.6f}",
                    f"{baseline['lpips']:.6f}",
                    args.lr,
                    f"{dt:.2f}",
                ]
            )

        if val_m["psnr"] > best_psnr:
            best_psnr = val_m["psnr"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_psnr": best_psnr,
                    "val_ssim": val_m["ssim"],
                    "val_lpips": val_m["lpips"],
                    "baseline": baseline,
                    "config": {
                        "base_ch": cfg["model"]["base_ch"],
                        "lambda_ssim": args.lambda_ssim,
                        "seed": args.seed,
                    },
                },
                ckpt_path,
            )
            print(f"  saved best checkpoint -> {ckpt_path} (PSNR={best_psnr:.4f})")

    print(f"Training complete. Best val PSNR={best_psnr:.4f} | log={log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train i4C restoration U-Net")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data_root", type=str, default=None, help="Override config data_root")
    p.add_argument("--sanity_check", action="store_true", help="Overfit 2 pairs then exit")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda_ssim", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_csv", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS (faster / no lpips pkg)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)

    # CLI overrides
    if args.data_root is None:
        args.data_root = cfg["data_root"]
    if args.epochs is None:
        args.epochs = cfg["train"]["epochs"]
    if args.batch_size is None:
        args.batch_size = cfg["train"]["batch_size"]
    if args.lr is None:
        args.lr = cfg["train"]["lr"]
    if args.lambda_ssim is None:
        args.lambda_ssim = cfg["train"]["lambda_ssim"]
    if args.val_ratio is None:
        args.val_ratio = cfg["val_ratio"]
    if args.seed is None:
        args.seed = cfg["seed"]
    if args.log_csv is None:
        args.log_csv = cfg["train"]["log_csv"]
    if args.checkpoint is None:
        args.checkpoint = cfg["train"]["checkpoint"]

    set_seed(int(args.seed))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.sanity_check:
        run_sanity_check(args, cfg, device)
    else:
        train_full(args, cfg, device)


if __name__ == "__main__":
    main()
