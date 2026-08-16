# i4C — AI Image Restoration MVP

Restore clean ground-truth (GT) grayscale images from degraded NoisyLR inputs
(speckle noise + additive Gaussian noise + **2× downsampling**).

Hackathon baseline: small U-Net with PixelShuffle upsampling. Prioritizes a
clean, reproducible pipeline over novel architecture.

---

## Environment setup

- **Python:** 3.10+ recommended (developed/verified on **Python 3.14.2**)
- **OS:** Windows / Linux / macOS
- Create a venv and install pinned deps:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

GPU (CUDA) is used automatically when available; otherwise training and
inference fall back to CPU.

---

## Dataset layout (confirmed)

| Path | Contents |
|------|----------|
| `train/GT/*.npy` | 3200 GT images, `(256, 256)` float32 in `[0, 1]` |
| `train/NoisyLR/*.npy` | 3200 paired LR inputs, `(128, 128)` float32 (may be outside `[0, 1]`) |
| `Test_NoisyLR/NoisyLR/*.npy` | 400 test LR images (no GT), same shape/format as train NoisyLR |

Filenames match 1:1 between `train/GT` and `train/NoisyLR`. Downsampling is
baked into pixel size (exact 2×). **Do not clip NoisyLR on load.**

Optional inspection:

```bash
python inspect_data.py --data_root . --out_grid inspection_grid.png
```

---

## Reproduce (exact commands)

From the project root (`i4C/`):

```bash
# 1) Sanity-check the pipeline (overfit 2 pairs, ~200 steps)
python train.py --sanity_check

# 2) Full training (90/10 val split, seed 42)
python train.py --epochs 30 --batch_size 8

# 3) Restore the official test set
python inference.py --input_dir Test_NoisyLR/NoisyLR --output_dir results/test_restored

# 4) Qualitative val grid (best / worst PSNR vs GT)
python results_grid.py --out results/val_comparison_grid.png
```

Defaults also live in `configs/default.yaml` (`lr=1e-3`, `lambda_ssim=0.5`,
checkpoint `weights/best.pt`, log `results/train_log.csv`).

Useful flags: `--cpu`, `--no_lpips`, `--data_root <path>`, `--checkpoint <path>`.

---

## Input / output contract

| | Train / Val | Test inference |
|--|-------------|----------------|
| **Format** | `.npy` float32, single-channel grayscale | same |
| **Input** | NoisyLR `(128, 128)` — values may leave `[0, 1]` | same |
| **Target / output** | GT `(256, 256)` in `[0, 1]` | restored `(256, 256)` **clipped to `[0, 1]`** |
| **Loading** | `np.load` only (no PIL for data I/O) | recursive `.npy` under `--input_dir` |
| **Saving** | checkpoints / CSV / PNG grids | `.npy` with original filenames under `--output_dir` |

---

## Model architecture summary

**RestorationUNet** (`src/model.py`), ~**518K** trainable parameters (`base_ch=32`):

1. **Encoder** at LR scale: conv blocks with downsampling `128 → 64 → 32`
2. **Decoder** with skip connections back to `128`
3. **PixelShuffle(2)** head: conv → 4× channels → rearrange → **256×256**, then final 1×1 conv to 1 channel
4. **No** final sigmoid/tanh during training (loss + inference clip handle range)

**Loss:** `L1 + 0.5 × (1 − SSIM)`  
**Metrics:** PSNR, SSIM, LPIPS (Alex) on `[0, 1]`-clipped images  
**Baseline:** bicubic upsample of NoisyLR to 256×256 (no denoise), scored once on val

---

## Final results

Logged in `results/train_log.csv`. Best checkpoint selected by **val PSNR**
→ `weights/best.pt` (**epoch 22**).

| Method | Val PSNR ↑ | Val SSIM ↑ | Val LPIPS ↓ |
|--------|------------|------------|-------------|
| Bicubic upsample (baseline) | 23.28 | 0.562 | 0.431 |
| **U-Net + PixelShuffle (best epoch 22)** | **27.44** | **0.757** | **0.333** |

Δ vs bicubic: **+4.16 dB PSNR**, **+0.195 SSIM**, **−0.098 LPIPS**.

Training: 30 epochs, batch size 8, lr `1e-3`, Adam, seed 42, 2880 train / 320 val.
Qualitative best/worst cases: `results/val_comparison_grid.png`.
Test outputs (400 files): `results/test_restored/`.

---

## Hardware

| Setting | This run |
|---------|----------|
| Device | **CPU** (PyTorch CPU build) |
| Wall time | ~7 min/epoch ≈ **~3.5–4 h** for 30 epochs (+ LPIPS eval each epoch) |
| Note | An **H100** (or any modern CUDA GPU) would cut this to minutes, not hours — same commands; CUDA is auto-detected |

---

## Project layout

```
i4C/
  train.py / inference.py / results_grid.py / inspect_data.py
  configs/default.yaml
  requirements.txt
  src/{dataset,model,losses,metrics}.py
  weights/best.pt
  results/{train_log.csv,test_restored/,val_comparison_grid.png}
  train/{GT,NoisyLR}/
  Test_NoisyLR/NoisyLR/
```

---

## Known limitations / next steps

- **LR scheduling:** fixed `1e-3` for all 30 epochs; add cosine / ReduceLROnPlateau once val plateaus (~epoch 22+).
- **Model capacity:** ~518K params is intentionally small; raise `base_ch` or deepen the U-Net if GPU memory allows.
- **Perceptual loss:** eval uses LPIPS, but training does not — try a small LPIPS/VGG term alongside L1+SSIM for sharper textures on hard cases.
- **Hard failure modes:** worst-val examples (fine grain / noise-like textures) still hallucinate structure — see `results/val_comparison_grid.png`.
- **Classical baseline:** bicubic only; a bicubic + mild denoise filter would be a fairer non-learned baseline.
- **Algorithm unrolling:** explore unrolled iterative denoisers / restoration networks aligned with KLA reference approaches for more interpretable, physics-aware pipelines.
- **TODO (code):** SwinIR-lite backbone; mixed precision on GPU; stronger augmentations if capacity grows.

---

## License / notes

Hackathon MVP — reproducible baseline, not SOTA. Paths are relative to the
repo root; override with `--data_root` for portability.
