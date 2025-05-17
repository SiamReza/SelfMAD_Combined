# Analysis of Underfitting in ViT-MAE Large Model

## Problem Statement
The ViT-MAE Large model is experiencing significant underfitting during training for morph detection, resulting in poor performance metrics that are close to random guessing.

## Data Analysis

### Training Metrics
- **Training Loss**: Stabilizes around 0.69-0.70 (close to random prediction loss of log(2) ≈ 0.693)
- **Validation Loss**: Final value of 0.6943 after 14 epochs
- **AUC**: Consistently around 0.5 (random chance) in both training and validation
- **EER**: Approximately 0.5 throughout training
- **Accuracy**: Consistently 0.5 on validation set

### Batch Metrics
- High variability in loss during early batches (e.g., batch 2 with loss 5.58)
- Loss quickly stabilizes around 0.69-0.70 with minimal improvement
- Learning rate increases according to schedule without corresponding performance improvement

### Evaluation Metrics
- **AUC**: Decreases from 0.112 (epoch 1) to 0.033 (epoch 11) on LMA dataset
- **EER**: Increases from 0.803 (epoch 1) to 0.908 (epoch 11)
- **Prediction Behavior**: Model consistently predicts all samples as one class (APCER = 1.0, BPCER = 0.0)
- **Prediction Values**: bonafide_mean and morph_mean values are nearly identical (around 0.475)

## Root Causes of Underfitting

### 1. Frozen Encoder Layers
- Configuration shows `SIAM_VIT_UNFREEZE_LAYERS = 0`, meaning all encoder layers are frozen
- Pre-trained ViT-MAE features cannot adapt to the specific task of morph detection
- Morph detection requires identifying subtle differences not captured by generic pre-trained features

### 2. Excessive Learning Rate
- Base learning rate in automation_config.py is set to 0.1, which is extremely high for transformer models
- Typical fine-tuning learning rates for transformers with AdamW are 1e-5 to 5e-4
- Even with the 0.1 factor for encoder layers, the resulting rate of 0.01 is still too high
- High learning rate prevents convergence and stable optimization

### 3. Insufficient Gradient Accumulation
- Gradient accumulation steps are set to 1 (effectively no accumulation)
- Large models like ViT-MAE benefit from gradient accumulation for training stability
- Without accumulation, gradient updates may be too noisy

### 4. Premature Training Termination
- Training stopped after only 14 epochs due to early stopping with patience of 5
- Insufficient time for the model to learn complex patterns, especially with frozen encoder

### 5. Potential Dataset Issues
- Possible class imbalance or domain-specific challenges
- Varying performance across different test datasets suggests domain shift problems

## Recommended Solutions

### High Priority Changes

#### 1. Adjust Learning Rate
```python
# Change in automation_config.py
SIAM_LEARNING_RATE = 5e-4  # From 0.1 to 5e-4
```
- Reduces learning rate to an appropriate range for transformer fine-tuning
- Maintains differential learning rates for encoder and classifier
- Expected impact: Immediate improvement in training stability

#### 2. Unfreeze Encoder Layers
```python
# Change in automation_config.py
SIAM_VIT_UNFREEZE_LAYERS = 6  # From 0 to 6
```
- Unfreezes the last 6 layers of the encoder
- Allows the model to adapt pre-trained features to morph detection
- Expected impact: Significant improvement in feature learning

#### 3. Increase Gradient Accumulation
```python
# Change in automation_config.py
SIAM_VIT_GRADIENT_ACCUMULATION_STEPS = 4  # From 1 to 4
```
- Accumulates gradients over 4 batches before updating weights
- Improves training stability for the large model
- Expected impact: More consistent training progress

#### 4. Extend Training Duration
```python
# Change in automation_config.py
SIAM_EPOCHS = 100  # From 70 to 100
SIAM_EARLY_STOPPING_PATIENCE = 10  # From 5 to 10
```
- Allows more time for the model to learn with the new configuration
- Prevents premature stopping
- Expected impact: More complete learning process

### Medium Priority Changes

#### 5. Enhance Data Augmentation
```python
# Ensure this is set in automation_config.py
SIAM_VIT_ADVANCED_AUGMENTATIONS = True
```
- Verify advanced augmentations are enabled
- Consider adding domain-specific augmentations for morph detection
- Expected impact: Better generalization

#### 6. Implement Class Balancing
- Analyze dataset for class imbalance
- If imbalanced, implement one of:
  ```python
  # Option 1: Class weights in loss function
  class_weights = torch.tensor([1.0, class_weight_ratio]).to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
  
  # Option 2: Weighted sampling in DataLoader
  sampler = WeightedRandomSampler(weights, len(weights))
  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
  ```
- Expected impact: More balanced learning across classes

### Additional Recommendations

#### 7. Progressive Unfreezing Strategy
- Consider implementing a progressive unfreezing strategy:
  ```python
  # Pseudocode for progressive unfreezing
  for epoch in range(n_epochs):
      if epoch == 10:  # After 10 epochs
          # Unfreeze 2 more layers
          unfreeze_layers(model, 8)
      if epoch == 20:  # After 20 epochs
          # Unfreeze 2 more layers
          unfreeze_layers(model, 10)
  ```
- Gradually unfreezes more layers as training progresses
- Expected impact: More controlled adaptation of pre-trained features

#### 8. Feature Visualization
- Implement tools to visualize what features the model is learning
- Use techniques like Grad-CAM to understand attention patterns
- Expected impact: Better understanding of model behavior

## Implementation Plan

1. Make the high-priority changes first (learning rate, unfreezing, gradient accumulation, training duration)
2. Run a training cycle and monitor performance
3. If underfitting persists, implement medium-priority changes
4. Continue iterating and monitoring until satisfactory performance is achieved

By addressing these issues, the ViT-MAE model should be able to learn meaningful patterns for morph detection and overcome the current underfitting problem.
