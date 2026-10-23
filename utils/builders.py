from models import EarlyLevelSCSEUNet, FeatureLevelSCSEUNet, LateLevelSCSEUNet

def build_model(cfg):
    if cfg.uNetModel == "Single-MSI":
        return EarlyLevelSCSEUNet(
            n_classes=cfg.num_classes,
            in_channels=cfg.num_channels_S2,
        )

    if cfg.uNetModel == "Single-SAR":
        return EarlyLevelSCSEUNet(
            n_classes=cfg.num_classes,
            in_channels=cfg.num_channels_S1,
        )

    if cfg.uNetModel == "Single-Both":
        return EarlyLevelSCSEUNet(
            n_classes=cfg.num_classes,
            in_channels=cfg.num_channels_S1 + cfg.num_channels_S2,
        )

    if cfg.uNetModel == "Dual":
        if cfg.fusion_name == "Feature-scSE":
            return FeatureLevelSCSEUNet(
                n_classes=cfg.num_classes,
                in_ch_s1=cfg.num_channels_S1,
                in_ch_s2=cfg.num_channels_S2,
            )
        elif cfg.fusion_name == "Late-scSE":
            return LateLevelSCSEUNet(
                n_classes=cfg.num_classes,
                in_ch_s1=cfg.num_channels_S1,
                in_ch_s2=cfg.num_channels_S2,
            )

    raise ValueError(f"Unsupported setup: {cfg.uNetModel}, {cfg.fusion_name}")