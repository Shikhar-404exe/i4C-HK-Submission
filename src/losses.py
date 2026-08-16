"""Training losses: L1 + lambda * (1 - SSIM).

# TODO: try perceptual/LPIPS loss term for sharper textures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d


def ssim_loss_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable SSIM (mean over batch/spatial). Returns scalar in [0, 1] approximately.
    Expects (N, C, H, W). Uses a fixed Gaussian window (no learnable params).
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=c)
    mu_y = F.conv2d(target, window, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    ssim_map = numerator / denominator
    return ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """loss = L1(pred, gt) + lambda_ssim * (1 - SSIM(pred, gt))."""

    def __init__(self, lambda_ssim: float = 0.5) -> None:
        super().__init__()
        self.lambda_ssim = float(lambda_ssim)
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = self.l1(pred, target)
        ssim = ssim_loss_map(pred, target)
        total = l1 + self.lambda_ssim * (1.0 - ssim)
        return {"loss": total, "l1": l1.detach(), "ssim": ssim.detach()}
