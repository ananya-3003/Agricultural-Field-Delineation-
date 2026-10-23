import torch.nn as nn
from losses import TverskyLoss, BCETverskySobelLoss

from losses import BCETverskySobelLoss

def build_loss(cfg):
    weights = [cfg.bce_weight, cfg.tversky_weight, cfg.sobel_weight]

    if all(w == 0 for w in weights):
        raise ValueError("At least one loss weight must be > 0.")

    return BCETverskySobelLoss(
        bce_weight=cfg.bce_weight,
        tversky_weight=cfg.tversky_weight,
        sobel_weight=cfg.sobel_weight,
        tversky_alpha=cfg.tversky_alpha,
        tversky_beta=cfg.tversky_beta,
        reduction=cfg.loss_reduction,
    )