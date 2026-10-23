"""
Building blocks for Dual-Stream U-Net architecture.
Updated for EA_Delta with Residual Connections:
  - ResEncoderBlock: Residual shortcut (Identity or 1x1 Conv+BN) with scSE attention.
  - ResDecoderBlock: Residual shortcut across concatenated skip+upsample features with scSE attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .scse import SCSEBlock
except (ImportError, ValueError):
    from scse import SCSEBlock


class ResEncoderBlock(nn.Module):
    r"""
    Residual Encoder Block with scSE Attention.
    
    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN ------------------(+) -> ReLU -> scSE -> output
          \                                                             /
           -> Shortcut: [Conv1x1 -> BN] (if in_ch != out_ch) or Identity
    """
    def __init__(self, in_channels: int, out_channels: int, use_scse: bool = True, reduction: int = 16, use_sd: bool = True):
        super().__init__()
        # Main convolution branch: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Residual shortcut branch:
        # Case A: in_channels == out_channels -> Identity (S(x) = x)
        # Case B: in_channels != out_channels -> Conv 1x1 + BN projection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        # Attention module (SD-scSE or standard scSE)
        self.scse = SCSEBlock(out_channels, reduction, use_sd=use_sd) if use_scse else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.relu(out + residual)
        return self.scse(out)


class ResDecoderBlock(nn.Module):
    r"""
    Residual Decoder Block with Upsampling, Skip Connection, and scSE Attention.

    Structure:
        x -> TransposeConv -> Pad -> Concat(skip) -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -(+) -> ReLU -> scSE
                                                 \                                             /
                                                  -> Shortcut: [Conv1x1 -> BN] ---------------
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, use_scse: bool = True, reduction: int = 16, use_sd: bool = True):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        fused_channels = out_channels + skip_channels
        
        # Main convolution branch across fused features:
        self.conv = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Residual shortcut projection across fused channels to out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Attention module (SD-scSE or standard scSE)
        self.scse = SCSEBlock(out_channels, reduction, use_sd=use_sd) if use_scse else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            
        fused = torch.cat([x, skip], dim=1)
        residual = self.shortcut(fused)
        out = self.conv(fused)
        out = self.relu(out + residual)
        return self.scse(out)


# Aliases for 100% backward compatibility
EncoderBlock = ResEncoderBlock
DecoderBlock = ResDecoderBlock