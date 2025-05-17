# Focal Loss Implementation Summary

## Overview

This document summarizes the implementation of Focal Loss as an alternative to BCE Loss in the @SelfMAD-siam/ codebase. Focal Loss is designed to address class imbalance by down-weighting easy examples and focusing more on hard examples during training.

## Implementation Details

### 1. Added Focal Loss Implementation in utils/loss.py

Two new loss functions were added to `utils/loss.py`:

1. **FocalLoss**: A standard implementation of Focal Loss for binary classification
2. **FocalLossWithLabelSmoothing**: A version of Focal Loss that also applies label smoothing

```python
class FocalLoss(nn.Module):
    """Focal Loss for binary classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing more on hard examples.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y = 1, and p_t = 1 - p if y = 0
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
```

### 2. Added a Factory Function for Loss Selection

A factory function `get_loss_function` was added to create the appropriate loss function based on configuration:

```python
def get_loss_function(loss_type='bce', label_smoothing=0.1, focal_alpha=0.25, focal_gamma=2.0):
    """Create a loss function based on the specified type."""
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
```

### 3. Added Configuration Parameters in automation_config.py

New parameters were added to `automation_config.py` to configure the loss function:

```python
# Loss function parameters
SIAM_LOSS_TYPE = 'bce_smoothing'       # Loss function type: 'bce', 'bce_smoothing', 'focal', 'focal_smoothing'
SIAM_FOCAL_ALPHA = 0.25                # Alpha parameter for Focal Loss (weighting factor for rare class)
SIAM_FOCAL_GAMMA = 2.0                 # Gamma parameter for Focal Loss (focusing parameter)
```

These parameters are also added to the configuration dictionary in the `get_config()` function.

### 4. Updated train__.py to Use the New Loss Function

The training script was updated to use the `get_loss_function` factory function:

```python
# Get loss function configuration
try:
    from automation_config import get_config
    cfg = get_config()
    loss_type = cfg.get("siam_loss_type", "bce_smoothing")
    label_smoothing = cfg.get("siam_vit_label_smoothing", 0.1)
    focal_alpha = cfg.get("siam_focal_alpha", 0.25)
    focal_gamma = cfg.get("siam_focal_gamma", 2.0)
except (ImportError, AttributeError):
    # Default values if config is not available
    loss_type = "bce_smoothing"
    label_smoothing = 0.1
    focal_alpha = 0.25
    focal_gamma = 2.0
    
# Create loss function based on configuration
criterion = get_loss_function(
    loss_type=loss_type,
    label_smoothing=label_smoothing,
    focal_alpha=focal_alpha,
    focal_gamma=focal_gamma
)
```

### 5. Updated utils/model.py to Use the New Loss Function

The model initialization was updated to use the `get_loss_function` factory function:

```python
# Get loss function configuration
try:
    import sys
    sys.path.append('..')
    from automation_config import get_config
    cfg = get_config()
    loss_type = cfg.get("siam_loss_type", "bce_smoothing")
    label_smoothing = cfg.get("siam_vit_label_smoothing", 0.1)
    focal_alpha = cfg.get("siam_focal_alpha", 0.25)
    focal_gamma = cfg.get("siam_focal_gamma", 2.0)
except (ImportError, AttributeError):
    # Default values if config is not available
    loss_type = "bce_smoothing"
    label_smoothing = 0.1
    focal_alpha = 0.25
    focal_gamma = 2.0
    
# Create loss function based on configuration
self.cel = get_loss_function(
    loss_type=loss_type,
    label_smoothing=label_smoothing,
    focal_alpha=focal_alpha,
    focal_gamma=focal_gamma
)
```

## Usage

To switch between different loss functions, modify the `SIAM_LOSS_TYPE` parameter in `automation_config.py`:

- `'bce'`: Standard Binary Cross Entropy Loss
- `'bce_smoothing'`: BCE Loss with Label Smoothing
- `'focal'`: Focal Loss
- `'focal_smoothing'`: Focal Loss with Label Smoothing

You can also adjust the parameters for Focal Loss:

- `SIAM_FOCAL_ALPHA`: Controls the weight given to the rare class (default: 0.25)
- `SIAM_FOCAL_GAMMA`: Controls the down-weighting of easy examples (default: 2.0)

## Benefits of Focal Loss

1. **Addresses Class Imbalance**: Focal Loss is particularly useful for datasets with class imbalance, as it gives more weight to the minority class.
2. **Focuses on Hard Examples**: By down-weighting easy examples, Focal Loss helps the model focus on hard examples that contribute more to the loss.
3. **Improves Convergence**: Focal Loss can help the model converge faster and achieve better performance on imbalanced datasets.

## Conclusion

The implementation of Focal Loss provides an alternative to BCE Loss that can be particularly useful for datasets with class imbalance. The ability to switch between different loss functions allows for experimentation to find the best approach for a given dataset.
