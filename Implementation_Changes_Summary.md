# Implementation Changes to Fix Underfitting

## Overview of Changes

We've made several key changes to address the underfitting issue in the ViT-MAE model:

1. **Configuration Changes** (in `automation_config.py`):
   - Reduced learning rate from 0.1 to 5e-4
   - Unfrozen 6 encoder layers (from 0)
   - Increased gradient accumulation steps to 2
   - Extended training duration to 100 epochs
   - Increased early stopping patience to 12
   - Increased warmup percentage to 0.2

2. **Model Architecture Changes** (in `SelfMAD-siam/utils/model.py`):
   - Added sigmoid activation to the model output to make it compatible with BCE loss
   - Modified the forward method to return probabilities instead of raw logits

3. **Training Process Changes** (in `SelfMAD-siam/train__.py`):
   - Converted target labels to float for BCE loss compatibility
   - Updated output processing to handle sigmoid outputs correctly
   - Fixed path issues for model loading and saving

## Detailed Code Changes

### 1. Added Sigmoid Activation in Model Forward Method

```python
# In SelfMAD-siam/utils/model.py
# Process with ViT-MAE model
outputs = self.net(x, output_hidden_states=True)
# Get the [CLS] token from the last hidden state
cls_token = outputs.last_hidden_state[:, 0]
# Pass through classifier and apply sigmoid for binary classification
logits = self.classifier(cls_token)
x = torch.sigmoid(logits)  # Convert logits to probabilities
```

This change ensures the model outputs probabilities in the range [0,1] instead of raw logits, making it compatible with BCE loss.

### 2. Target Conversion for BCE Loss

```python
# In SelfMAD-siam/train__.py (training loop)
# Convert target to float for BCE loss (expects float targets)
if args.get("model") == "vit_mae_large":
    # For binary classification with BCE loss, convert to float and reshape
    target_float = target.float()
    
    # For ViT MAE, use gradient accumulation
    output = model(img)
    loss = criterion(output, target_float) / accumulation_steps  # Scale loss
```

```python
# In SelfMAD-siam/train__.py (validation loop)
# Convert target to float for BCE loss (expects float targets)
target_float = target.float()

with torch.no_grad():
    output = model(img)
    loss = criterion(output, target_float)
    val_loss += loss.item()
```

These changes ensure that the target labels are in the correct format for BCE loss, which expects float targets.

### 3. Output Processing for Metrics Calculation

```python
# In SelfMAD-siam/train__.py (training metrics)
# Collect outputs and targets for metrics
if args.get("model") == "vit_mae_large":
    # For ViT-MAE with sigmoid activation, use the output directly
    train_outputs.extend(output.detach().cpu().numpy().tolist())
else:
    # For other models using softmax
    train_outputs.extend(output.softmax(1)[:, 1].detach().cpu().numpy().tolist())
```

```python
# In SelfMAD-siam/train__.py (validation metrics)
# Process outputs based on model type
if args.get("model") == "vit_mae_large":
    # For ViT-MAE with sigmoid activation, use the output directly
    val_outputs.extend(output.cpu().data.numpy().tolist())
else:
    # For other models using softmax
    val_outputs.extend(output.softmax(1)[:,1].cpu().data.numpy().tolist())
```

These changes ensure that the model outputs are processed correctly for metrics calculation, taking into account the sigmoid activation.

## Expected Impact

These changes should address the underfitting issues by:

1. **Enabling Feature Learning**: Unfreezing encoder layers allows the model to adapt pre-trained features to the morph detection task.

2. **Optimizing Learning Rate**: The reduced learning rate should provide more stable optimization.

3. **Improving Training Stability**: Gradient accumulation and increased warmup percentage should lead to more stable training.

4. **Allowing Sufficient Training Time**: Extended training duration and increased early stopping patience give the model more time to learn.

5. **Ensuring Loss Compatibility**: The sigmoid activation and target conversion ensure that the BCE loss works correctly.

## Next Steps

1. Run a training cycle with these changes and monitor the loss and metrics.
2. If underfitting persists, consider:
   - Further increasing the number of unfrozen layers
   - Implementing class balancing if the dataset is imbalanced
   - Adding more domain-specific data augmentations
   - Exploring alternative model architectures or pre-trained models

3. Monitor the training process closely, especially in the early epochs, to ensure the loss is decreasing properly.
