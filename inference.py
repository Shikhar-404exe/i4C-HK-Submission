"""
Run restoration on Test_NoisyLR (or any folder of 128x128 .npy files).

Example:
  python inference.py --input_dir Test_NoisyLR/NoisyLR --output_dir results/test_restored
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.metrics import clip01
from src.model import RestorationUNet


def list_npy_recursive(root: Path) -> list[Path]:
    files = [
        p
        for p in root.rglob("*.npy")
        if p.is_file() and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: str(p).lower())


def load_checkpoint(path: Path, device: torch.device) -> tuple[RestorationUNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_ch = 32
    if isinstance(ckpt, dict) and "config" in ckpt:
        base_ch = int(ckpt["config"].get("base_ch", 32))
    model = RestorationUNet(in_ch=1, out_ch=1, base_ch=base_ch).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def restore_file(model: RestorationUNet, path: Path, device: torch.device) -> np.ndarray:
    arr = np.load(path)  # do not clip input
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got {arr.shape}")
    if arr.shape != (128, 128):
        raise ValueError(f"{path}: expected (128,128), got {arr.shape}")
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0).unsqueeze(0).to(device)
    y = clip01(model(x))
    out = y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="i4C restoration inference")
    p.add_argument(
        "--input_dir",
        type=str,
        default="Test_NoisyLR/NoisyLR",
        help="Folder of degraded .npy (searched recursively)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/test_restored",
        help="Where to write restored .npy files",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best.pt",
        help="Path to trained checkpoint",
    )
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            "Train first: python train.py  (or python train.py --sanity_check)"
        )

    files = list_npy_recursive(input_dir)
    if not files:
        raise FileNotFoundError(f"No .npy files under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, meta = load_checkpoint(ckpt_path, device)
    print(f"device={device}  files={len(files)}  checkpoint={ckpt_path}")
    if "val_psnr" in meta:
        print(f"checkpoint val_psnr={meta['val_psnr']:.4f} (epoch {meta.get('epoch', '?')})")

    for i, path in enumerate(files, 1):
        restored = restore_file(model, path, device)
        # Preserve relative path under input_dir when nested
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, restored)
        if i == 1 or i == len(files) or i % 50 == 0:
            print(f"  [{i}/{len(files)}] {rel} -> {out_path} shape={restored.shape}")

    elapsed = time.perf_counter() - t_start
    print(f"Done. Wrote {len(files)} files to {output_dir.resolve()}")
    print(f"Total end-to-end wall-clock runtime: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
