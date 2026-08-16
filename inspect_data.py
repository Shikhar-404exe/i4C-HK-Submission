"""
Dataset inspection script for i4C image restoration MVP.
Run before any model code to verify folder layout, pairing, shapes, and value ranges.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name)


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    # Fallback for standard image formats if present later
    return plt.imread(path)


def summarize_array(name: str, arr: np.ndarray) -> None:
    finite = np.isfinite(arr)
    print(
        f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, finite={finite.all()} "
        f"(nonfinite={(~finite).sum()})"
    )


def to_display(arr: np.ndarray) -> np.ndarray:
    """Map array to [0,1] for visualization only (does not alter source data)."""
    x = arr.astype(np.float64)
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        pass  # HWC
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        pass
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x)
    return np.clip(x, 0.0, 1.0)


def check_train_pairs(gt_dir: Path, noisy_dir: Path) -> list[str]:
    print("=" * 72)
    print("TRAIN PAIRING CHECK")
    print("=" * 72)

    gt_files = list_image_files(gt_dir)
    noisy_files = list_image_files(noisy_dir)
    gt_names = {p.name for p in gt_files}
    noisy_names = {p.name for p in noisy_files}

    print(f"train/GT      : {len(gt_files)} files  (exts={Counter(p.suffix.lower() for p in gt_files)})")
    print(f"train/NoisyLR : {len(noisy_files)} files  (exts={Counter(p.suffix.lower() for p in noisy_files)})")

    only_gt = sorted(gt_names - noisy_names)
    only_noisy = sorted(noisy_names - gt_names)
    matched = sorted(gt_names & noisy_names)

    print(f"Matched 1:1   : {len(matched)}")
    print(f"Orphans in GT only      : {len(only_gt)}")
    if only_gt:
        print(f"  examples: {only_gt[:10]}")
    print(f"Orphans in NoisyLR only : {len(only_noisy)}")
    if only_noisy:
        print(f"  examples: {only_noisy[:10]}")

    if len(only_gt) == 0 and len(only_noisy) == 0 and len(matched) == len(gt_files):
        print("PASS: filenames match 1:1 between GT and NoisyLR.")
    else:
        print("FAIL: pairing mismatch — see orphans above.")

    return matched


def sample_and_compare(
    gt_dir: Path,
    noisy_dir: Path,
    matched_names: list[str],
    n_samples: int,
    out_grid: Path,
) -> None:
    print()
    print("=" * 72)
    print("SAMPLE PAIR STATS + RESOLUTION CHECK")
    print("=" * 72)

    if not matched_names:
        print("No matched pairs to inspect.")
        return

    # Stratify a bit: first, middle, last + a few evenly spaced
    idxs = np.linspace(0, len(matched_names) - 1, num=min(n_samples, len(matched_names)), dtype=int)
    idxs = sorted(set(idxs.tolist()))
    names = [matched_names[i] for i in idxs]

    shape_pairs: Counter[tuple] = Counter()
    same_res = 0
    smaller_noisy = 0
    larger_noisy = 0

    # Broader resolution census over more files (fast: load headers via full load for .npy)
    census_n = min(64, len(matched_names))
    census_idxs = np.linspace(0, len(matched_names) - 1, num=census_n, dtype=int)
    census_names = [matched_names[i] for i in sorted(set(census_idxs.tolist()))]

    print(f"Resolution census over {len(census_names)} pairs:")
    for name in census_names:
        gt = load_array(gt_dir / name)
        noisy = load_array(noisy_dir / name)
        gh, gw = gt.shape[:2]
        nh, nw = noisy.shape[:2]
        shape_pairs[(gt.shape, noisy.shape)] += 1
        if (gh, gw) == (nh, nw):
            same_res += 1
        elif nh * nw < gh * gw:
            smaller_noisy += 1
        else:
            larger_noisy += 1

    for (gshape, nshape), cnt in sorted(shape_pairs.items(), key=lambda x: -x[1]):
        print(f"  GT {gshape}  <->  NoisyLR {nshape}   ({cnt} pairs in census)")

    print()
    print(
        f"Resolution summary (census): same={same_res}, "
        f"NoisyLR_smaller={smaller_noisy}, NoisyLR_larger={larger_noisy}"
    )
    if smaller_noisy > 0 and same_res == 0:
        print(
            "CONCLUSION: Downsampling is baked into pixel dimensions "
            "(NoisyLR is spatially smaller than GT). Model must upsample."
        )
    elif same_res > 0 and smaller_noisy == 0:
        print(
            "CONCLUSION: Same spatial resolution — restore in-place "
            "(quality degradation without smaller H/W)."
        )
    else:
        print("CONCLUSION: Mixed resolution relationship — handle both cases in the pipeline.")

    print()
    print(f"Detailed stats for {len(names)} display samples:")
    display_pairs = []
    for name in names:
        gt = load_array(gt_dir / name)
        noisy = load_array(noisy_dir / name)
        print(f"- {name}")
        summarize_array("GT     ", gt)
        summarize_array("NoisyLR", noisy)
        ratio_h = gt.shape[0] / max(noisy.shape[0], 1)
        ratio_w = gt.shape[1] / max(noisy.shape[1], 1)
        print(f"  scale GT/NoisyLR ~ ({ratio_h:.3f}x H, {ratio_w:.3f}x W)")
        display_pairs.append((name, gt, noisy))

    # Save side-by-side grid (NoisyLR upscaled for visual comparison only)
    n = len(display_pairs)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, (name, gt, noisy) in enumerate(display_pairs):
        gt_d = to_display(gt)
        noisy_d = to_display(noisy)

        # Nearest-neighbor upsample NoisyLR to GT size for side-by-side viewing
        if noisy_d.shape[:2] != gt_d.shape[:2]:
            # Simple repeat upsample (integer factors) or matplotlib imshow handles size;
            # for grid consistency, use np.kron-like via indexing
            zy = gt_d.shape[0] / noisy_d.shape[0]
            zx = gt_d.shape[1] / noisy_d.shape[1]
            if abs(zy - round(zy)) < 1e-6 and abs(zx - round(zx)) < 1e-6:
                zy_i, zx_i = int(round(zy)), int(round(zx))
                noisy_up = np.repeat(np.repeat(noisy_d, zy_i, axis=0), zx_i, axis=1)
            else:
                yy = (np.linspace(0, noisy_d.shape[0] - 1, gt_d.shape[0])).astype(int)
                xx = (np.linspace(0, noisy_d.shape[1] - 1, gt_d.shape[1])).astype(int)
                noisy_up = noisy_d[np.ix_(yy, xx)] if noisy_d.ndim == 2 else noisy_d[yy][:, xx]
        else:
            noisy_up = noisy_d

        axes[row, 0].imshow(noisy_d, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"NoisyLR\n{name}\n{noisy.shape}")
        axes[row, 1].imshow(noisy_up, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"NoisyLR upscaled\n(for view only)\n{noisy_up.shape[:2]}")
        axes[row, 2].imshow(gt_d, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"GT\n{gt.shape}")
        for c in range(3):
            axes[row, c].axis("off")

    fig.suptitle("Train pairs: NoisyLR | NoisyLR upscaled (viz) | GT", fontsize=12)
    fig.tight_layout()
    out_grid.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Saved visual sanity-check grid -> {out_grid.resolve()}")


def check_test(test_dir: Path) -> None:
    print()
    print("=" * 72)
    print("TEST SET CHECK")
    print("=" * 72)

    # Prefer nested NoisyLR if present (documented quirk)
    nested = test_dir / "NoisyLR"
    search_root = nested if nested.is_dir() else test_dir
    print(f"Listing images under: {search_root}")

    # Recursive listing in case of deeper nesting
    files = sorted(
        [
            p
            for p in search_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
        ],
        key=lambda p: str(p.relative_to(search_root)),
    )
    print(f"Test image count: {len(files)}")
    print(f"Extensions: {Counter(p.suffix.lower() for p in files)}")

    if not files:
        print("WARNING: no test images found.")
        return

    shapes = Counter()
    print(f"Sample stats ({min(8, len(files))} files):")
    sample_idxs = np.linspace(0, len(files) - 1, num=min(8, len(files)), dtype=int)
    for i in sorted(set(sample_idxs.tolist())):
        arr = load_array(files[i])
        shapes[arr.shape] += 1
        rel = files[i].relative_to(search_root)
        print(f"- {rel}")
        summarize_array("Test   ", arr)

    # Broader shape census
    census_n = min(64, len(files))
    census_idxs = np.linspace(0, len(files) - 1, num=census_n, dtype=int)
    shape_census: Counter[tuple] = Counter()
    for i in sorted(set(census_idxs.tolist())):
        arr = load_array(files[i])
        shape_census[arr.shape] += 1
    print(f"Shape census over {census_n} test files:")
    for shape, cnt in sorted(shape_census.items(), key=lambda x: -x[1]):
        print(f"  {shape}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect i4C restoration dataset")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing train/ and Test_NoisyLR/",
    )
    parser.add_argument("--n_samples", type=int, default=6, help="Pairs to print/plot")
    parser.add_argument(
        "--out_grid",
        type=Path,
        default=Path("inspection_grid.png"),
        help="Output PNG path for side-by-side pairs",
    )
    args = parser.parse_args()

    root = args.data_root
    gt_dir = root / "train" / "GT"
    noisy_dir = root / "train" / "NoisyLR"
    test_dir = root / "Test_NoisyLR"

    print(f"data_root = {root.resolve()}")
    print(f"exists train/GT={gt_dir.is_dir()}, train/NoisyLR={noisy_dir.is_dir()}, "
          f"Test_NoisyLR={test_dir.is_dir()}")

    matched = check_train_pairs(gt_dir, noisy_dir)
    sample_and_compare(gt_dir, noisy_dir, matched, args.n_samples, args.out_grid)
    check_test(test_dir)
    print()
    print("Inspection complete. Review printouts + inspection_grid.png before model work.")


if __name__ == "__main__":
    main()
