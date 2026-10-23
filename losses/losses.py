import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLoss(nn.Module):
    """
    Sobel Edge Penalty Loss
    Computes spatial gradient magnitude using 2D Sobel filters and measures L1 difference.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        pred_probs = torch.sigmoid(pred_logits).clamp(self.eps, 1.0 - self.eps)

        gx_pred = F.conv2d(pred_probs, self.sobel_x.to(pred_logits.device), padding=1)
        gy_pred = F.conv2d(pred_probs, self.sobel_y.to(pred_logits.device), padding=1)
        mag_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2 + 1e-12)

        gx_tgt = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        gy_tgt = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
        mag_tgt = torch.sqrt(gx_tgt ** 2 + gy_tgt ** 2 + 1e-12)

        return F.l1_loss(mag_pred, mag_tgt)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, eps=1e-7, from_logits=True, log_loss=False, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.clamp(self.eps, 1.0 - self.eps)
        dims = (0, 2, 3)

        tp = torch.sum(pred * target, dim=dims)
        fp = torch.sum(pred * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - pred) * target, dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        ).clamp_min(self.eps)

        if self.log_loss:
            loss = -torch.log(tversky.clamp_min(self.eps))
        else:
            loss = 1.0 - tversky

        return loss.mean() ** self.gamma


class BCETverskySobelLoss(nn.Module):
    def __init__(self, bce_weight=1.0, tversky_weight=1.0, sobel_weight=0.25, tversky_alpha=0.5, tversky_beta=0.5, smooth=1e-6, reduction="mean"):
        super().__init__()

        self.bce_w = float(bce_weight)
        self.tversky_w = float(tversky_weight)
        self.sobel_w = float(sobel_weight)

        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            eps=smooth,
            from_logits=True,
            log_loss=False,
            gamma=1.0,
        )
        self.sobel = SobelLoss(eps=smooth)
        self.last_losses = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        total = torch.zeros((), device=logits.device, dtype=logits.dtype)
        bce_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        tversky_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        sobel_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self.bce_w > 0:
            if weights is not None:
                if weights.dim() == 3:
                    weights = weights.unsqueeze(1)
                bce_loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * weights).mean()
            else:
                bce_loss = self.bce(logits, targets)
            total = total + self.bce_w * bce_loss

        if self.tversky_w > 0:
            tversky_loss = self.tversky(logits, targets)
            total = total + self.tversky_w * tversky_loss

        if self.sobel_w > 0:
            sobel_loss = self.sobel(logits, targets)
            total = total + self.sobel_w * sobel_loss

        self.last_losses = {
            "total": total.detach(),
            "bce": bce_loss.detach(),
            "tversky": tversky_loss.detach(),
            "sobel": sobel_loss.detach(),
        }

        return total