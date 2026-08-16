"""Evaluation metrics: PSNR, SSIM, LPIPS on [0,1]-clipped images."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]

_lpips_model = None


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _as_tensor(x: ArrayLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach()
    else:
        t = torch.from_numpy(np.asarray(x))
    t = t.float()
    if device is not None:
        t = t.to(device)
    return t


def clip01(x: ArrayLike) -> ArrayLike:
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, 0.0, 1.0)
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def psnr(pred: ArrayLike, target: ArrayLike, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB). Inputs clipped to [0,1] first."""
    pred_t = clip01(_as_tensor(pred))
    target_t = clip01(_as_tensor(target))
    mse = torch.mean((pred_t - target_t) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian_window_np(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(pred: ArrayLike, target: ArrayLike, data_range: float = 1.0) -> float:
    """Mean SSIM; inputs clipped to [0,1]. Works on (H,W), (1,H,W), or (N,1,H,W)."""
    pred_t = clip01(_as_tensor(pred))
    target_t = clip01(_as_tensor(target))

    if pred_t.ndim == 2:
        pred_t = pred_t.unsqueeze(0).unsqueeze(0)
        target_t = target_t.unsqueeze(0).unsqueeze(0)
    elif pred_t.ndim == 3:
        pred_t = pred_t.unsqueeze(0)
        target_t = target_t.unsqueeze(0)

    c = pred_t.shape[1]
    window_size = 11
    window = _gaussian_window_np(window_size).to(pred_t.device)
    window = window.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = F.conv2d(pred_t, window, padding=pad, groups=c)
    mu_y = F.conv2d(target_t, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(pred_t * pred_t, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target_t * target_t, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred_t * target_t, window, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return float(ssim_map.mean().item())


def _get_lpips(device: torch.device):
    global _lpips_model
    if _lpips_model is None:
        import lpips  # deferred import

        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    else:
        _lpips_model = _lpips_model.to(device)
    return _lpips_model


def lpips_distance(pred: ArrayLike, target: ArrayLike, device: Optional[torch.device] = None) -> float:
    """
    LPIPS (AlexNet). Lower is better.
    Grayscale inputs are repeated to 3 channels; values mapped from [0,1] to [-1,1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_t = clip01(_as_tensor(pred, device))
    target_t = clip01(_as_tensor(target, device))

    def to_nchw3(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        return t * 2.0 - 1.0

    model = _get_lpips(device)
    with torch.no_grad():
        d = model(to_nchw3(pred_t), to_nchw3(target_t))
    return float(d.mean().item())


def compute_all(
    pred: ArrayLike,
    target: ArrayLike,
    device: Optional[torch.device] = None,
    with_lpips: bool = True,
) -> dict[str, float]:
    out = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
    if with_lpips:
        try:
            out["lpips"] = lpips_distance(pred, target, device=device)
        except Exception as exc:  # noqa: BLE001 — keep training usable if lpips missing
            out["lpips"] = float("nan")
            out["lpips_error"] = str(exc)
    return out
