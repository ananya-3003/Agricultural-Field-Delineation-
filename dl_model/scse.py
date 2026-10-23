import torch
import torch.nn as nn


class ChannelSE(nn.Module):
    """
    Standard Squeeze-and-Excitation Channel Attention (Roy et al., 2018).
    Uses Global Average Pooling (GAP) to extract spatial channel means.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SDChannelSE(nn.Module):
    """
    Delta Modification #3: Standard-Deviation Squeeze-and-Excitation Channel Attention.
    
    Instead of relying solely on spatial mean (which is blind to boundaries),
    this block calculates BOTH:
      1. Spatial Mean (μ_c): Captures global channel intensity.
      2. Spatial Standard Deviation (σ_c): Captures high-frequency boundary variance.
      
    Concatenates [μ_c ; σ_c] into a [2C, 1, 1] descriptor vector, then excites channels
    that exhibit high boundary contrast while suppressing flat crop interior noise.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)

        # Input dimension is 2 * in_channels due to concatenation of [mean, std]
        self.fc1 = nn.Conv2d(in_channels * 2, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Compute spatial mean μ_c: [B, C, 1, 1]
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        
        # 2. Compute spatial variance and standard deviation σ_c: [B, C, 1, 1]
        # Using clamp(min=1e-8) for numerical stability during backprop
        var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        
        # 3. Concatenate descriptors along channel dimension: [B, 2C, 1, 1]
        stats = torch.cat([mean, std], dim=1)
        
        # 4. Squeeze-and-Excitation MLP
        y = self.fc1(stats)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # 5. Channel recalibration
        return x * y


class SpatialSE(nn.Module):
    """
    Spatial Squeeze-and-Excitation Attention (Roy et al., 2018).
    Squeezes channels to generate a single-channel spatial attention map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return x * y


class SCSEBlock(nn.Module):
    """
    Dual-branch Concurrent Spatial and Channel Squeeze-and-Excitation (scSE) Block.
    
    Args:
        in_channels (int): Number of feature channels.
        reduction (int): Channel bottleneck reduction ratio (default: 16).
        use_sd (bool): If True, enables Delta #3 Standard-Deviation Attention (SDChannelSE).
                       If False, falls back to standard GAP ChannelSE.
    """
    def __init__(self, in_channels, reduction=16, use_sd=True):
        super().__init__()
        self.use_sd = use_sd
        if use_sd:
            self.cse = SDChannelSE(in_channels, reduction=reduction)
        else:
            self.cse = ChannelSE(in_channels, reduction=reduction)
            
        self.sse = SpatialSE(in_channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)