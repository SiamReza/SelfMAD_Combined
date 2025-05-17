# Combined Dataset Feature Documentation

## Overview

This document describes the new combined dataset loading feature implemented in both SelfMAD-siam and SelfMAD-main codebases. This feature allows training on multiple datasets simultaneously, which can improve model generalization and performance.

## Configuration

The combined dataset feature is controlled by a new parameter in the `automation_config.py` file:

```python
# Enable combined dataset loading for training
# When True, all datasets in DATASETS will be combined for training
# When False, each dataset will be processed separately
ENABLE_COMBINED_DATASET = False
```

When `ENABLE_COMBINED_DATASET` is set to `True`, all datasets specified in the `DATASETS` list will be combined for training. For example:

```python
# List of datasets to process
DATASETS = ["LMA", "MIPGAN_I"]  # These datasets will be combined when ENABLE_COMBINED_DATASET is True
```

## Implementation Details

### New Classes

Two new dataset classes have been implemented to support combined dataset loading:

1. `CombinedMorphDataset` in `SelfMAD-siam/utils/custom_dataset.py`
2. `CombinedMorphDataset` in `SelfMAD-main/utils/dataset.py`

These classes inherit from `selfMAD_Dataset` and `Dataset` respectively, and override the `create_dataset` method to load images from multiple datasets.

### How It Works

When `ENABLE_COMBINED_DATASET` is set to `True`:

1. The training script reads the `DATASETS` list from `automation_config.py`
2. Instead of creating a `MorphDataset` instance for a single dataset, it creates a `CombinedMorphDataset` instance with all datasets in the list
3. The `CombinedMorphDataset` class loads images from all specified datasets and combines them into a single dataset
4. The combined dataset is then split into training and validation sets according to the `train_val_split` parameter

### Benefits

Training on multiple datasets simultaneously can provide several benefits:

1. **Improved Generalization**: Models trained on diverse data tend to generalize better to unseen data
2. **Increased Training Data**: Combining datasets increases the total amount of training data, which can improve model performance
3. **Balanced Learning**: The model learns from multiple data distributions simultaneously, which can help prevent overfitting to a single dataset

## Usage

To use the combined dataset feature:

1. Edit `automation_config.py` to set `ENABLE_COMBINED_DATASET = True`
2. Add the datasets you want to combine to the `DATASETS` list
3. Run the training script as usual

Example configuration:

```python
# Enable combined dataset loading
ENABLE_COMBINED_DATASET = True

# List of datasets to combine
DATASETS = ["LMA", "MIPGAN_I", "LMA_UBO"]
```

## Limitations

- All datasets must have the same structure (bonafide/morph folders with train/test subfolders)
- The combined dataset feature only works with custom morph datasets, not with the original SelfMAD datasets (FF++ or SMDD)
- The combined dataset feature is only available for training, not for testing

## Example

Here's an example of how to train a model on a combined dataset:

```bash
# Edit automation_config.py to enable combined dataset loading
# ENABLE_COMBINED_DATASET = True
# DATASETS = ["LMA", "MIPGAN_I"]

# Run the training script
python SelfMAD-siam/train__.py --model vit_mae_large --train_dataset LMA_MIPGAN_I
```

Note that the `--train_dataset` parameter is still required, but it's only used for naming the output directory. The actual datasets used for training are determined by the `DATASETS` list in `automation_config.py`.

## Troubleshooting

If you encounter issues with the combined dataset feature:

1. Check that all datasets in the `DATASETS` list exist and have the correct structure
2. Verify that the `ENABLE_COMBINED_DATASET` parameter is set to `True` in `automation_config.py`
3. Check the console output for any error messages related to dataset loading
4. Ensure that the datasets are properly located in the `datasets` directory

## Implementation Changes

The following files were modified to implement the combined dataset feature:

1. `automation_config.py`: Added `ENABLE_COMBINED_DATASET` parameter
2. `SelfMAD-siam/utils/custom_dataset.py`: Added `CombinedMorphDataset` class
3. `SelfMAD-siam/train__.py`: Modified to use `CombinedMorphDataset` when `ENABLE_COMBINED_DATASET` is `True`
4. `SelfMAD-main/utils/dataset.py`: Added `CombinedMorphDataset` class
5. `SelfMAD-main/train__.py`: Modified to use `CombinedMorphDataset` when `ENABLE_COMBINED_DATASET` is `True`
