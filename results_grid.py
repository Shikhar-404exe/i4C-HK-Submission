"""
Val-set qualitative grid: best/worst PSNR cases vs GT.

Test has no GT, so this scores the same 90/10 val split used in training,
runs weights/best.pt, and saves:
  NoisyLR (NN-upscaled for view) | Model output | GT
with per-image PSNR/SSIM labels.

Example:
  python results_grid.py
  python results_grid.py --n_best 4 --n_worst 4 --out results/val_comparison_grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import build_datasets
from src.metrics import clip01, psnr, ssim
from src.model import RestorationUNet


def load_model(checkpoint: Path, device: torch.device) -> RestorationUNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    base_ch = 32
    if isinstance(ckpt, dict) and "config" in ckpt:
        base_ch = int(ckpt["config"].get("base_ch", 32))
    model = RestorationUNet(in_ch=1, out_ch=1, base_ch=base_ch).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def to_display2d(arr: np.ndarray) -> np.ndarray:
    """Map array to [0,1] for visualization only."""
    x = arr.astype(np.float64)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x)
    return np.clip(x, 0.0, 1.0)


@torch.no_grad()
def score_val(
    model: RestorationUNet,
    loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    """Per-image metrics + tensors needed for plotting."""
    rows: list[dict] = []
    model.eval()
    for batch in loader:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        names = batch["name"]
        pred = clip01(model(noisy))
        gt_c = clip01(gt)

        # NN upsample NoisyLR to 256 for side-by-side viewing
        noisy_up = F.interpolate(noisy, size=(256, 256), mode="nearest")

        for i in range(noisy.shape[0]):
            p = pred[i, 0].cpu().numpy()
            g = gt_c[i, 0].cpu().numpy()
            n_up = noisy_up[i, 0].cpu().numpy()
            rows.append(
                {
                    "name": names[i],
                    "psnr": psnr(p, g),
                    "ssim": ssim(p, g),
                    "noisy_up": n_up,
                    "pred": p,
                    "gt": g,
                }
            )
    return rows


def pick_extremes(rows: list[dict], n_best: int, n_worst: int) -> list[dict]:
    ranked = sorted(rows, key=lambda r: r["psnr"], reverse=True)
    best = ranked[:n_best]
    worst = list(reversed(ranked[-n_worst:]))  # worst first within bad block
    # Avoid duplicates if val set is tiny
    seen = set()
    picked: list[dict] = []
    for r in best + worst:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        picked.append(r)
    return picked


def save_grid(examples: list[dict], n_best: int, out_path: Path) -> None:
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n), squeeze=False)

    for row, ex in enumerate(examples):
        tag = "BEST" if row < n_best else "WORST"
        panels = [
            (to_display2d(ex["noisy_up"]), "NoisyLR (upscaled)"),
            (np.clip(ex["pred"], 0, 1), "Model"),
            (np.clip(ex["gt"], 0, 1), "GT"),
        ]
        for col, (img, title) in enumerate(panels):
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
        axes[row, 0].set_ylabel(
            f"{tag}\n{ex['name']}\nPSNR {ex['psnr']:.2f}\nSSIM {ex['ssim']:.3f}",
            fontsize=8,
            rotation=0,
            labelpad=48,
            va="center",
            ha="right",
        )

    fig.suptitle(
        "Val comparison — best / worst PSNR cases\n"
        "NoisyLR (NN×2 for view) | Model | GT",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.08, 0.01, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid -> {out_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Val best/worst PSNR comparison grid")
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--checkpoint", type=str, default="weights/best.pt")
    p.add_argument("--out", type=str, default="results/val_comparison_grid.png")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_best", type=int, default=4)
    p.add_argument("--n_worst", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    _, val_ds, _, val_names = build_datasets(
        args.data_root, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"val size={len(val_names)} device={device} checkpoint={ckpt_path}")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model(ckpt_path, device)

    print("Scoring all val images...")
    rows = score_val(model, loader, device)
    rows_sorted = sorted(rows, key=lambda r: r["psnr"], reverse=True)
    print(
        f"val PSNR  min={rows_sorted[-1]['psnr']:.2f} ({rows_sorted[-1]['name']})  "
        f"max={rows_sorted[0]['psnr']:.2f} ({rows_sorted[0]['name']})  "
        f"mean={np.mean([r['psnr'] for r in rows]):.2f}"
    )

    picked = pick_extremes(rows, n_best=args.n_best, n_worst=args.n_worst)
    print("Selected examples:")
    for i, ex in enumerate(picked):
        tag = "BEST" if i < args.n_best else "WORST"
        print(f"  [{tag}] {ex['name']}  PSNR={ex['psnr']:.2f}  SSIM={ex['ssim']:.3f}")

    save_grid(picked, n_best=args.n_best, out_path=Path(args.out))


if __name__ == "__main__":
    main()
