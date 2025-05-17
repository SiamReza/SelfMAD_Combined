# LMA_MIPGAN_I Dataset Implementation

This document summarizes the changes made to integrate the new LMA_MIPGAN_I dataset into both the SelfMAD-siam and SelfMAD-main codebases.

## 1. Overview

The LMA_MIPGAN_I dataset has been fully integrated into the codebase, allowing it to be used for training and evaluation just like the existing datasets (LMA, LMA_UBO, MIPGAN_I, MIPGAN_II, MorDiff, StyleGAN).

## 2. Dataset Structure

The LMA_MIPGAN_I dataset follows the same structure as the existing datasets:

```
datasets/
├── bonafide/
│   └── LMA_MIPGAN_I/
│       ├── train/
│       │   └── [genuine images for training]
│       └── test/
│           └── [genuine images for testing]
└── morph/
    └── LMA_MIPGAN_I/
        ├── train/
        │   └── [morphed images for training]
        └── test/
            └── [morphed images for testing]
```

## 3. Changes Made

The following files were updated to include the LMA_MIPGAN_I dataset:

### 3.1. Root Directory Files

1. **automation_config.py**
   - Added "LMA_MIPGAN_I" to the DATASETS list
   - Added "LMA_MIPGAN_I_path" to the DATASET_PATHS dictionary

2. **test_dataset_loading.py**
   - Added "LMA_MIPGAN_I_path" to both morph_config dictionaries

3. **run_experiments_direct.py**
   - Added "LMA_MIPGAN_I_path" to the dataset_path_args construction

### 3.2. SelfMAD-siam Files

1. **SelfMAD-siam/train__.py**
   - Added "LMA_MIPGAN_I" to the list of valid train_dataset values
   - Added "LMA_MIPGAN_I_path" to the morph_config dictionary

2. **SelfMAD-siam/eval__.py**
   - Added "LMA_MIPGAN_I" to the morph_datasets list
   - Added a new argument for LMA_MIPGAN_I_path

3. **SelfMAD-siam/configs/morph_config.json**
   - Added "LMA_MIPGAN_I_path": "../datasets" to the JSON

4. **SelfMAD-siam/config.py**
   - Added "LMA_MIPGAN_I" to the datasets dictionary
   - Added "LMA_MIPGAN_I" to the test_datasets list

5. **SelfMAD-siam/run_train.sh**
   - Added "LMA_MIPGAN_I" to the list of available datasets in the error message

6. **SelfMAD-siam/run_test.sh**
   - Added "LMA_MIPGAN_I_path ../datasets" to the DATASET_PARAMS construction

### 3.3. SelfMAD-main Files

1. **SelfMAD-main/train__.py**
   - Added "LMA_MIPGAN_I" to the list of valid train_dataset values
   - Added "LMA_MIPGAN_I" to the custom_morph_datasets list

2. **SelfMAD-main/eval__.py**
   - Added "LMA_MIPGAN_I" to the custom_morph_datasets list
   - Added a new argument for LMA_MIPGAN_I_path

3. **SelfMAD-main/configs/data_config.json**
   - Added "LMA_MIPGAN_I_path": "../datasets" to the JSON

## 4. How to Use the New Dataset

### 4.1. Training

To train a model on the LMA_MIPGAN_I dataset, you can use the following commands:

1. Using run_experiments_direct.py:
   ```bash
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models siam
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models main
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models both
   ```

2. Using SelfMAD-siam directly:
   ```bash
   cd SelfMAD-siam
   ./run_train.sh --dataset LMA_MIPGAN_I
   ```

### 4.2. Evaluation

To evaluate a trained model on the LMA_MIPGAN_I dataset:

1. Using SelfMAD-siam directly:
   ```bash
   cd SelfMAD-siam
   ./run_test.sh --model_path <path_to_model> --datasets LMA_MIPGAN_I
   ```

## 5. Verification

The implementation has been verified to ensure that:
- The dataset can be loaded correctly
- Training can be performed on the dataset
- Evaluation can be performed on the dataset

## 6. Conclusion

The LMA_MIPGAN_I dataset has been fully integrated into both the SelfMAD-siam and SelfMAD-main codebases. It can now be used for training and evaluation just like the existing datasets.
