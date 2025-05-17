import torch
import torch.nn as nn

def smooth_targets(targets, smoothing=0.1):
    """Apply label smoothing to target values.

    Label smoothing prevents the model from becoming too confident in its predictions
    by "smoothing" the target labels away from hard 0 and 1 values.

    Args:
        targets: Target tensor with values typically 0 or 1
        smoothing: Smoothing factor (default: 0.1)

    Returns:
        Smoothed target tensor where:
        - 1 becomes (1 - smoothing) + smoothing * 0.5 = 1 - smoothing * 0.5
        - 0 becomes 0 + smoothing * 0.5
    """
    # Apply smoothing: targets * (1 - smoothing) + smoothing * 0.5
    return targets * (1 - smoothing) + smoothing * 0.5

class BCELossWithLabelSmoothing(nn.Module):
    """BCE Loss with built-in label smoothing.

    This is a wrapper around nn.BCELoss that applies label smoothing
    to the targets before computing the loss.
    """
    def __init__(self, smoothing=0.1):
        super(BCELossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # Apply label smoothing
        target_smooth = smooth_targets(target, self.smoothing)
        return self.bce(pred, target_smooth)

class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing more on hard examples.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y = 1, and p_t = 1 - p if y = 0

    Args:
        alpha: Weighting factor for the rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # Small constant to prevent log(0)

    def forward(self, pred, target):
        # Ensure pred is between [0, 1]
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # Calculate binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Apply focal weighting
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * bce

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLossWithLabelSmoothing(nn.Module):
    """Focal Loss with built-in label smoothing.

    This combines Focal Loss with label smoothing to get the benefits of both:
    - Focal Loss addresses class imbalance
    - Label smoothing prevents overconfidence

    Args:
        alpha: Weighting factor for the rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        smoothing: Label smoothing factor (default: 0.1)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.eps = 1e-6  # Small constant to prevent log(0)

    def forward(self, pred, target):
        # Apply label smoothing
        target_smooth = smooth_targets(target, self.smoothing)

        # Ensure pred is between [0, 1]
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # Calculate binary cross entropy with smoothed targets
        bce = -target_smooth * torch.log(pred) - (1 - target_smooth) * torch.log(1 - pred)

        # Apply focal weighting (using original targets for determining easy/hard examples)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * bce

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def get_loss_function(loss_type='bce', label_smoothing=0.1, focal_alpha=0.25, focal_gamma=2.0):
    """Create a loss function based on the specified type.

    Args:
        loss_type: Type of loss function ('bce', 'bce_smoothing', 'focal', 'focal_smoothing')
        label_smoothing: Label smoothing factor (default: 0.1)
        focal_alpha: Alpha parameter for Focal Loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0)

    Returns:
        A loss function (nn.Module)
    """
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'bce_smoothing':
        return BCELossWithLabelSmoothing(smoothing=label_smoothing)
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'focal_smoothing':
        return FocalLossWithLabelSmoothing(alpha=focal_alpha, gamma=focal_gamma, smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Expected one of: 'bce', 'bce_smoothing', 'focal', 'focal_smoothing'")