import torch
import torch.nn as nn
try:
    from .blocks import EncoderBlock, DecoderBlock
    from .scse import SCSEBlock
except (ImportError, ValueError):
    from blocks import EncoderBlock, DecoderBlock
    from scse import SCSEBlock

try:
    from config import Config
except (ImportError, ValueError):
    try:
        from ..config import Config
    except (ImportError, ValueError):
        Config = None

class EarlyLevelSCSEUNet(nn.Module):
    def __init__(self, n_classes = 1, in_channels = None, features = (64, 128, 256, 512), use_scse = True, reduction = 16, use_sd = True):
        super().__init__()
        if in_channels is None:
            raise ValueError("in_channels must be provided.")
        if len(features) != 4:
            raise ValueError("features must contain 4 values.")

        self.initial = EncoderBlock(in_channels=in_channels, out_channels=features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = EncoderBlock(features[0], features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder2 = EncoderBlock(features[0], features[1], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder3 = EncoderBlock(features[1], features[2], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder4 = EncoderBlock(features[2], features[3], use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.center = EncoderBlock(features[3], features[3], use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.decoder4 = DecoderBlock(features[3], features[2], skip_channels=features[3], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder3 = DecoderBlock(features[2], features[1], skip_channels=features[2], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder2 = DecoderBlock(features[1], features[0], skip_channels=features[1], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder1 = DecoderBlock(features[0], features[0], skip_channels=features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.up_final = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, n_classes, kernel_size=1),
        )

    def forward(self, x):
        e0 = self.initial(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        c = self.center(self.pool(e4))

        d4 = self.decoder4(c, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        up = self.up_final(d1)
        return self.final(up)


class FeatureLevelSCSEUNet(nn.Module):
    def __init__(self, n_classes = 1, in_ch_s1 = 3, in_ch_s2 = 4, features = (64, 128, 256, 512), reduction = 16, use_scse = True, use_sd = True):
        super().__init__()
        if len(features) != 4:
            raise ValueError("features must contain 4 values.")

        self.initial_s1 = EncoderBlock(in_ch_s1, features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder1_s1 = EncoderBlock(features[0], features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder2_s1 = EncoderBlock(features[0], features[1], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder3_s1 = EncoderBlock(features[1], features[2], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder4_s1 = EncoderBlock(features[2], features[3], use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.initial_s2 = EncoderBlock(in_ch_s2, features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder1_s2 = EncoderBlock(features[0], features[0], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder2_s2 = EncoderBlock(features[0], features[1], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder3_s2 = EncoderBlock(features[1], features[2], use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.encoder4_s2 = EncoderBlock(features[2], features[3], use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Central Bottleneck (1024 -> 512)
        self.center_conv = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[3], features[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
        )
        self.center_shortcut = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(features[3]),
        )
        self.center_relu = nn.ReLU(inplace=True)
        self.center_scse = SCSEBlock(features[3], reduction, use_sd=use_sd) if use_scse else nn.Identity()

        self.decoder4 = DecoderBlock(features[3], features[2], skip_channels=features[2] * 2, use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder3 = DecoderBlock(features[2], features[1], skip_channels=features[1] * 2, use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder2 = DecoderBlock(features[1], features[0], skip_channels=features[0] * 2, use_scse=use_scse, reduction=reduction, use_sd=use_sd)
        self.decoder1 = DecoderBlock(features[0], features[0], skip_channels=features[0] * 2, use_scse=use_scse, reduction=reduction, use_sd=use_sd)

        self.final = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, n_classes, kernel_size=1),
        )

    def forward(self, x1, x2):
        e0_s1 = self.initial_s1(x1)
        e1_s1 = self.encoder1_s1(self.pool(e0_s1))
        e2_s1 = self.encoder2_s1(self.pool(e1_s1))
        e3_s1 = self.encoder3_s1(self.pool(e2_s1))
        e4_s1 = self.encoder4_s1(self.pool(e3_s1))

        e0_s2 = self.initial_s2(x2)
        e1_s2 = self.encoder1_s2(self.pool(e0_s2))
        e2_s2 = self.encoder2_s2(self.pool(e1_s2))
        e3_s2 = self.encoder3_s2(self.pool(e2_s2))
        e4_s2 = self.encoder4_s2(self.pool(e3_s2))

        e0 = torch.cat([e0_s1, e0_s2], dim=1)
        e1 = torch.cat([e1_s1, e1_s2], dim=1)
        e2 = torch.cat([e2_s1, e2_s2], dim=1)
        e3 = torch.cat([e3_s1, e3_s2], dim=1)
        e4 = torch.cat([e4_s1, e4_s2], dim=1)

        # Residual Center Bottleneck: Main + Shortcut -> ReLU -> scSE
        c = self.center_relu(self.center_conv(e4) + self.center_shortcut(e4))
        c = self.center_scse(c)

        d4 = self.decoder4(c, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        return self.final(d1)


class LateLevelSCSEUNet(nn.Module):
    def __init__(self, n_classes = 1, in_ch_s1 = 3, in_ch_s2 = 4, features = (64, 128, 256, 512), reduction = 16, use_scse = True):
        super().__init__()
        if len(features) != 4:
            raise ValueError("features must contain 4 values.")

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1_0 = EncoderBlock(in_ch_s1, features[0], use_scse=use_scse, reduction=reduction)
        self.enc1_1 = EncoderBlock(features[0], features[1], use_scse=use_scse, reduction=reduction)
        self.enc1_2 = EncoderBlock(features[1], features[2], use_scse=use_scse, reduction=reduction)
        self.enc1_3 = EncoderBlock(features[2], features[3], use_scse=use_scse, reduction=reduction)
        self.center1 = EncoderBlock(features[3], features[3], use_scse=use_scse, reduction=reduction)

        self.dec1_3 = DecoderBlock(features[3], features[2], skip_channels=features[3], use_scse=use_scse, reduction=reduction)
        self.dec1_2 = DecoderBlock(features[2], features[1], skip_channels=features[2], use_scse=use_scse, reduction=reduction)
        self.dec1_1 = DecoderBlock(features[1], features[0], skip_channels=features[1], use_scse=use_scse, reduction=reduction)
        self.dec1_0 = DecoderBlock(features[0], features[0], skip_channels=features[0], use_scse=use_scse, reduction=reduction)

        self.enc2_0 = EncoderBlock(in_ch_s2, features[0], use_scse=use_scse, reduction=reduction)
        self.enc2_1 = EncoderBlock(features[0], features[1], use_scse=use_scse, reduction=reduction)
        self.enc2_2 = EncoderBlock(features[1], features[2], use_scse=use_scse, reduction=reduction)
        self.enc2_3 = EncoderBlock(features[2], features[3], use_scse=use_scse, reduction=reduction)
        self.center2 = EncoderBlock(features[3], features[3], use_scse=use_scse, reduction=reduction)

        self.dec2_3 = DecoderBlock(features[3], features[2], skip_channels=features[3], use_scse=use_scse, reduction=reduction)
        self.dec2_2 = DecoderBlock(features[2], features[1], skip_channels=features[2], use_scse=use_scse, reduction=reduction)
        self.dec2_1 = DecoderBlock(features[1], features[0], skip_channels=features[1], use_scse=use_scse, reduction=reduction)
        self.dec2_0 = DecoderBlock(features[0], features[0], skip_channels=features[0], use_scse=use_scse, reduction=reduction)

        self.final = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], n_classes, kernel_size=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1_0 = self.enc1_0(x1)
        e1_1 = self.enc1_1(self.pool(e1_0))
        e1_2 = self.enc1_2(self.pool(e1_1))
        e1_3 = self.enc1_3(self.pool(e1_2))
        c1 = self.center1(self.pool(e1_3))

        d1_3 = self.dec1_3(c1, e1_3)
        d1_2 = self.dec1_2(d1_3, e1_2)
        d1_1 = self.dec1_1(d1_2, e1_1)
        d1_0 = self.dec1_0(d1_1, e1_0)

        e2_0 = self.enc2_0(x2)
        e2_1 = self.enc2_1(self.pool(e2_0))
        e2_2 = self.enc2_2(self.pool(e2_1))
        e2_3 = self.enc2_3(self.pool(e2_2))
        c2 = self.center2(self.pool(e2_3))

        d2_3 = self.dec2_3(c2, e2_3)
        d2_2 = self.dec2_2(d2_3, e2_2)
        d2_1 = self.dec2_1(d2_2, e2_1)
        d2_0 = self.dec2_0(d2_1, e2_0)

        fused = torch.cat([d1_0, d2_0], dim=1)
        return self.final(fused)