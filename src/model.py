"""Lightweight U-Net: 1x128x128 NoisyLR -> 1x256x256 restored GT."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers at constant spatial size."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool then ConvBlock -> half spatial size."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample 2x, concat skip, ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pads if needed (should not for power-of-two sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class RestorationUNet(nn.Module):
    """
    Encoder-decoder at LR scale, then PixelShuffle 2x head to HR.

    Spatial path (default base_ch=32):
      enc0: 128x128
      enc1:  64x64
      enc2:  32x32  (bottleneck)
      dec1:  64x64  (+ skip enc1)
      dec0: 128x128 (+ skip enc0)
      head: 256x256 via conv -> 4*out_ch, PixelShuffle(2), final 1x1 conv

    No final sigmoid/tanh — loss + inference clipping handle range.
    # TODO: try SwinIR-lite backbone for stronger restoration.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        c = base_ch
        self.enc0 = ConvBlock(in_ch, c)       # 128, c
        self.enc1 = Down(c, c * 2)            # 64,  2c
        self.enc2 = Down(c * 2, c * 4)        # 32,  4c

        self.dec1 = Up(c * 4, c * 2, c * 2)   # 64,  2c
        self.dec0 = Up(c * 2, c, c)           # 128, c

        # PixelShuffle 2x: channels -> 4 * mid, then rearrange to 2x spatial
        mid = c
        self.pre_ps = nn.Conv2d(c, mid * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)          # 256, mid
        self.head = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2, e1)
        d0 = self.dec0(d1, e0)
        y = self.ps(self.pre_ps(d0))
        return self.head(y)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(base_ch: int = 32) -> RestorationUNet:
    return RestorationUNet(in_ch=1, out_ch=1, base_ch=base_ch)


if __name__ == "__main__":
    model = build_model(base_ch=32)
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"params: {count_parameters(model):,}")
    assert y.shape == (1, 1, 256, 256), f"Unexpected output shape {y.shape}"
    print("OK: output shape is (1, 1, 256, 256)")
