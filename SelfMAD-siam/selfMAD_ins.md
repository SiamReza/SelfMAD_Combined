# Adapting SelfMAD for ViT-MAE Morph Detection

This document outlines the necessary modifications to adapt the SelfMAD codebase for fine-tuning a ViT-MAE large model on custom morph datasets. The focus is on preserving the original training and testing logic while adapting the data loading strategy to work with our specific dataset structure with minimal necessary changes.

## Project Overview

The goal is to fine-tune the ViT-MAE large model on 6 different morph datasets (LMA, LMA-UBO, MIPGAN-I, MIPGAN-II, MorDiff, StyleGAN) to evaluate cross-dataset generalization. We will train on one dataset at a time and test on all datasets to assess how well the model generalizes across different morphing techniques.

## Dataset Structure

Our dataset structure is organized as follows:
```
datasets/
├─ bonafide/
│   ├─ LMA/
│   │   ├─ train/  # Used for training/validation
│   │   └─ test/   # Used for testing
│   ├─ LMA_UBO/
│   │   ├─ train/
│   │   └─ test/
│   ├─ MIPGAN_I/
│   │   ├─ train/
│   │   └─ test/
│   ├─ MIPGAN_II/
│   │   ├─ train/
│   │   └─ test/
│   ├─ MorDiff/
│   │   ├─ train/
│   │   └─ test/
│   └─ StyleGAN/
│       ├─ train/
│       └─ test/
└─ morph/
    ├─ LMA/
    │   ├─ train/
    │   └─ test/
    ├─ LMA_UBO/
    │   ├─ train/
    │   └─ test/
    ├─ MIPGAN_I/
    │   ├─ train/
    │   └─ test/
    ├─ MIPGAN_II/
    │   ├─ train/
    │   └─ test/
    ├─ MorDiff/
    │   ├─ train/
    │   └─ test/
    └─ StyleGAN/
        ├─ train/
        └─ test/
```

## Training and Testing Flow

### Training Phase
```
┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Select training dataset (e.g., LMA)                                  │
│ 2. Load bonafide images from 'train' folder (label 0)                   │
│ 3. Load morph images from 'train' folder (label 1)                      │
│ 4. Combine and shuffle the dataset                                      │
│ 5. Split into training (80%) and validation (20%) sets                  │
│ 6. Generate CSV file with image paths, labels, and splits               │
│ 7. Apply SelfMAD data augmentation techniques                           │
│ 8. Train ViT-MAE model for binary classification                        │
│ 9. Save best model based on validation performance                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Testing Phase
```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TESTING PHASE                               │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Load trained model                                                    │
│ 2. For each dataset (LMA, LMA-UBO, MIPGAN-I, MIPGAN-II, MorDiff, StyleGAN):│
│    a. Load bonafide images from 'test' folder (label 0)                  │
│    b. Load morph images from 'test' folder (label 1)                     │
│    c. Run inference on all test images                                   │
│    d. Calculate metrics (EER, AUC) for bonafide vs. morph classification │
│ 3. Generate performance comparison across all datasets                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Required Modifications

After analyzing the SelfMAD codebase, we need to make the following modifications to adapt it to our custom dataset structure and requirements:

### 1. Configuration Files Adaptation

Modify the existing configuration files to work with our dataset structure:

1. **Update `configs/data_config.json`**:
   - Replace the existing dataset paths with our custom dataset paths
   - Add entries for all 6 morph datasets (LMA, LMA-UBO, MIPGAN-I, MIPGAN-II, MorDiff, StyleGAN)
   - Add configuration for train/validation split percentage

2. **Update `configs/train_config.json`**:
   - Add ViT-MAE model option
   - Configure training parameters for the ViT-MAE model
   - Add CSV generation configuration

### 2. Custom Dataset Adapter

Create a custom dataset adapter that extends the existing `selfMAD_Dataset` class:

1. **Create `utils/custom_dataset.py`**:
   - Implement a `MorphDataset` class that loads images from our custom dataset structure
   - Support train/test folder structure
   - Generate CSV files with image paths, labels, and splits
   - Implement train/validation split logic controlled by configuration parameter
   - Resize images to 224×224 using Lanczos resampling for optimal quality

2. **CSV Generation Logic**:
   - Generate a CSV file for each training run with columns: image_path, label, split
   - Example format:
     ```
     image_path,label,split
     datasets/bonafide/LMA/train/img_001.jpg,0,train
     datasets/morph/LMA/train/img_002.jpg,1,val
     ```

### 3. Model Integration

Modify the model implementation to support the ViT-MAE large model:

1. **Update `utils/model.py` to support ViT-MAE large**:
   - Add a new model option "vit_mae_large"
   - Load the pre-trained ViT-MAE large model weights
   - Configure the model for binary classification (bonafide vs. morph)

2. **Implement ViT-MAE Initialization**:
   - Load the pre-trained ViT-MAE model from the model/vit_mae directory
   - Extract the encoder part of the model
   - Add a classification head for binary classification

### 4. Training Script Adaptation

Modify the training script to work with our custom dataset adapter and ViT-MAE model:

1. **Update `train__.py`**:
   - Add support for our custom dataset adapter
   - Add "vit_mae_large" to the list of valid models
   - Implement CSV generation logic
   - Update the training loop to work with our dataset structure

2. **Train/Validation Split Logic**:
   - Implement logic to split the combined dataset according to the configured percentage
   - Ensure reproducibility by using a fixed random seed
   - Save the split information in the CSV file

### 5. Evaluation Script Adaptation

Modify the evaluation script to work with our test datasets:

1. **Update `eval__.py`**:
   - Add support for our custom dataset structure
   - Update the `default_datasets` function to include our 6 morph datasets
   - Implement logic to load bonafide and morph images from the 'test' folders
   - Calculate metrics (EER, AUC) for bonafide vs. morph classification

2. **Cross-Dataset Evaluation**:
   - Implement logic to evaluate a model trained on one dataset against all test datasets
   - Generate a performance comparison table across all datasets

## Implementation Details

### Bonafide vs. Morph Classification Testing

For testing, we need to load bonafide and morph images separately to evaluate classification performance:

1. For each dataset in the test list:
   a. Load all bonafide images from the 'test' folder (label 0)
   b. Load all morph images from the 'test' folder (label 1)
   c. Create separate data loaders for bonafide and morph images

This approach allows us to calculate metrics specifically for bonafide vs. morph classification and assess the model's ability to generalize across different morphing techniques.

### Leveraging Existing Components

While adapting the codebase, we will leverage the following existing components:

1. **Self-supervised morphing techniques** from `utils/selfMAD.py`:
   - Self-blending
   - Self-morphing
   - Image augmentation pipeline

2. **Training infrastructure** from `train__.py`:
   - Model initialization
   - Training loop
   - Validation logic
   - Model saving

3. **Evaluation metrics** from `utils/metrics.py`:
   - EER (Equal Error Rate)
   - AUC (Area Under Curve)

4. **Preprocessing tools** from the original codebase:
   - Landmark extraction
   - Face parsing

## Implementation Steps

1. **Create Configuration Files**:
   - Create `config.py` with dataset configuration
   - Define which dataset to use for training and which for testing

2. **Create Custom Dataset Adapter**:
   - Create `utils/custom_dataset.py` extending `selfMAD_Dataset`
   - Implement train/validation split logic
   - Implement CSV generation

3. **Update Model Support**:
   - Modify `utils/model.py` to support ViT-MAE large

4. **Update Training Script**:
   - Modify `train__.py` to use our custom dataset adapter
   - Implement initialization for ViT-MAE large model

5. **Update Evaluation Script**:
   - Modify `eval__.py` to work with our test datasets
   - Ensure proper evaluation of bonafide vs. morph classification

6. **Test the Implementation**:
   - Train on one dataset (e.g., LMA)
   - Test on all datasets
   - Verify metrics calculation

By making these targeted modifications, we can adapt the SelfMAD codebase to work with our custom dataset structure and ViT-MAE large model without changing the core functionality of the original implementation.
