"""Paired GT / NoisyLR dataset for 2x restoration (128 -> 256, grayscale .npy)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Confirmed layout: GT (256,256) float32 in [0,1]; NoisyLR (128,128) float32, may be outside [0,1].
IMAGE_EXT = ".npy"


def list_npy_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == IMAGE_EXT and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name)


def paired_filenames(gt_dir: Path, noisy_dir: Path) -> list[str]:
    """Return sorted filenames present in both GT and NoisyLR (1:1 match)."""
    gt_names = {p.name for p in list_npy_files(gt_dir)}
    noisy_names = {p.name for p in list_npy_files(noisy_dir)}
    only_gt = gt_names - noisy_names
    only_noisy = noisy_names - gt_names
    if only_gt or only_noisy:
        raise RuntimeError(
            f"Filename mismatch: {len(only_gt)} GT-only, {len(only_noisy)} NoisyLR-only "
            f"(examples GT={sorted(only_gt)[:5]}, NoisyLR={sorted(only_noisy)[:5]})"
        )
    return sorted(gt_names)


def train_val_split(
    names: Sequence[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Fixed-seed 90/10 split with no leakage (disjoint name lists)."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    names = list(names)
    rng = random.Random(seed)
    shuffled = names.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)  # keep at least one train sample
    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])
    assert set(train_names).isdisjoint(set(val_names))
    return train_names, val_names


def _to_chw1(arr: np.ndarray) -> torch.Tensor:
    """(H,W) float array -> float32 tensor (1,H,W). Does not clip."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {arr.shape}")
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return t.unsqueeze(0)


class RestorationDataset(Dataset):
    """
    Loads paired NoisyLR (1,128,128) and GT (1,256,256).

    Augmentation (train only): horizontal flip, vertical flip, k*90 rotation.
    Applied consistently to both tensors (GT rotated/flipped at its own resolution).
    No random crop — fixed sizes; cropping would break LR/HR spatial pairing.
    NoisyLR is never clipped on load.
    """

    def __init__(
        self,
        data_root: str | Path,
        filenames: Sequence[str],
        augment: bool = False,
        gt_subdir: str = "train/GT",
        noisy_subdir: str = "train/NoisyLR",
    ) -> None:
        self.data_root = Path(data_root)
        self.gt_dir = self.data_root / gt_subdir
        self.noisy_dir = self.data_root / noisy_subdir
        self.filenames = list(filenames)
        self.augment = augment

        if not self.filenames:
            raise ValueError("RestorationDataset received empty filename list")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.filenames[idx]
        noisy = np.load(self.noisy_dir / name)  # do NOT clip
        gt = np.load(self.gt_dir / name)

        if noisy.shape != (128, 128):
            raise ValueError(f"{name}: expected NoisyLR (128,128), got {noisy.shape}")
        if gt.shape != (256, 256):
            raise ValueError(f"{name}: expected GT (256,256), got {gt.shape}")

        noisy_t = _to_chw1(noisy)
        gt_t = _to_chw1(gt)

        if self.augment:
            noisy_t, gt_t = self._augment_pair(noisy_t, gt_t)

        return {"noisy": noisy_t, "gt": gt_t, "name": name}

    @staticmethod
    def _augment_pair(
        noisy: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Horizontal flip
        if random.random() < 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
        # Vertical flip
        if random.random() < 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            gt = torch.flip(gt, dims=[-2])
        # Random 90-degree rotation (0,1,2,3) — both square, same aspect
        k = random.randint(0, 3)
        if k:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            gt = torch.rot90(gt, k, dims=[-2, -1])
        return noisy, gt


def build_datasets(
    data_root: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    sanity_names: Optional[Sequence[str]] = None,
) -> tuple[RestorationDataset, RestorationDataset, list[str], list[str]]:
    """
    Build train/val datasets with a fixed split.

    If sanity_names is provided, both train and val use those names (overfit check).
    """
    data_root = Path(data_root)
    all_names = paired_filenames(data_root / "train" / "GT", data_root / "train" / "NoisyLR")

    if sanity_names is not None:
        names = list(sanity_names)
        train_ds = RestorationDataset(data_root, names, augment=True)
        val_ds = RestorationDataset(data_root, names, augment=False)
        return train_ds, val_ds, names, names

    train_names, val_names = train_val_split(all_names, val_ratio=val_ratio, seed=seed)
    train_ds = RestorationDataset(data_root, train_names, augment=True)
    val_ds = RestorationDataset(data_root, val_names, augment=False)
    return train_ds, val_ds, train_names, val_names
