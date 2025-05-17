# Plan for Introducing LMA_MIPGAN_I Dataset to the Codebase

This document outlines the comprehensive plan for introducing the new LMA_MIPGAN_I dataset to both @SelfMAD-siam/ and @SelfMAD-main/ codebases. The plan includes all necessary code changes to ensure the dataset can be used for training and evaluation.

## 1. Dataset Structure Requirements

The new LMA_MIPGAN_I dataset must follow the same structure as the existing datasets:

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

## 2. Code Changes Required

### 2.1. Update `automation_config.py`

```python
# List of datasets to process
DATASETS = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]  # Add "LMA_MIPGAN_I"

# Dataset paths
DATASET_PATHS = {
    "LMA_path": datasets_dir,
    "LMA_UBO_path": datasets_dir,
    "MIPGAN_I_path": datasets_dir,
    "MIPGAN_II_path": datasets_dir,
    "MorDiff_path": datasets_dir,
    "StyleGAN_path": datasets_dir,
    "LMA_MIPGAN_I_path": datasets_dir  # Add this line
}
```

### 2.2. Update `SelfMAD-siam/train__.py`

```python
# Line 85: Add "LMA_MIPGAN_I" to the list of valid train_dataset values
assert args["train_dataset"] in ["FF++", "SMDD", "LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]

# Line 350-356: Add "LMA_MIPGAN_I_path" to the morph_config dictionary
morph_config = {
    "LMA_path": args.get("LMA_path", datasets_dir),
    "LMA_UBO_path": args.get("LMA_UBO_path", datasets_dir),
    "MIPGAN_I_path": args.get("MIPGAN_I_path", datasets_dir),
    "MIPGAN_II_path": args.get("MIPGAN_II_path", datasets_dir),
    "MorDiff_path": args.get("MorDiff_path", datasets_dir),
    "StyleGAN_path": args.get("StyleGAN_path", datasets_dir),
    "LMA_MIPGAN_I_path": args.get("LMA_MIPGAN_I_path", datasets_dir)  # Add this line
}
```

### 2.3. Update `SelfMAD-main/train__.py`

```python
# Line 84: Add "LMA_MIPGAN_I" to the list of valid train_dataset values
assert args["train_dataset"] in ["FF++", "SMDD", "LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]

# Line 187: Add "LMA_MIPGAN_I" to the custom_morph_datasets list
custom_morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]
```

### 2.4. Update `SelfMAD-siam/eval__.py`

```python
# Line 328: Add "LMA_MIPGAN_I" to the morph_datasets list
morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]

# Line 427-430: Add a new argument for LMA_MIPGAN_I_path
parser.add_argument('-LMA_MIPGAN_I_path', type=str, required=False)
```

### 2.5. Update `SelfMAD-main/eval__.py`

```python
# Line 310: Add "LMA_MIPGAN_I" to the custom_morph_datasets list
custom_morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]

# Line 419-424: Add a new argument for LMA_MIPGAN_I_path
parser.add_argument('-LMA_MIPGAN_I_path', type=str, required=False)
```

### 2.6. Update `SelfMAD-siam/configs/morph_config.json`

```json
{
    "LMA_path": "../datasets",
    "LMA_UBO_path": "../datasets",
    "MIPGAN_I_path": "../datasets",
    "MIPGAN_II_path": "../datasets",
    "MorDiff_path": "../datasets",
    "StyleGAN_path": "../datasets",
    "LMA_MIPGAN_I_path": "../datasets",  // Add this line
    "train_val_split": 0.8,
    "csv_output_path": "./output/train"
}
```

### 2.7. Update `SelfMAD-main/configs/data_config.json`

```json
{
    "LMA_path": "../datasets",
    "LMA_UBO_path": "../datasets",
    "MIPGAN_I_path": "../datasets",
    "MIPGAN_II_path": "../datasets",
    "MorDiff_path": "../datasets",
    "StyleGAN_path": "../datasets",
    "LMA_MIPGAN_I_path": "../datasets"  // Add this line
}
```

### 2.8. Update `SelfMAD-siam/config.py`

```python
# Line 8-14: Add "LMA_MIPGAN_I" to the datasets dictionary
self.datasets = {
    "LMA": os.path.join("..", "datasets"),
    "LMA_UBO": os.path.join("..", "datasets"),
    "MIPGAN_I": os.path.join("..", "datasets"),
    "MIPGAN_II": os.path.join("..", "datasets"),
    "MorDiff": os.path.join("..", "datasets"),
    "StyleGAN": os.path.join("..", "datasets"),
    "LMA_MIPGAN_I": os.path.join("..", "datasets")  # Add this line
}

# Line 19: Add "LMA_MIPGAN_I" to the test_datasets list
self.test_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]
```

### 2.9. Update `SelfMAD-siam/run_train.sh`

```bash
# Line 96: Add "LMA_MIPGAN_I" to the list of available datasets in the error message
echo "Available datasets: LMA, LMA_UBO, MIPGAN_I, MIPGAN_II, MorDiff, StyleGAN, LMA_MIPGAN_I"
```

### 2.10. Update `SelfMAD-siam/run_test.sh`

```bash
# Line 63: Add "LMA_MIPGAN_I_path ../datasets" to the DATASET_PARAMS construction
DATASET_PARAMS="-LMA_path ../datasets -LMA_UBO_path ../datasets -MIPGAN_I_path ../datasets -MIPGAN_II_path ../datasets -MorDiff_path ../datasets -StyleGAN_path ../datasets -LMA_MIPGAN_I_path ../datasets"
```

### 2.11. Update `test_dataset_loading.py`

```python
# Line 47-54: Add "LMA_MIPGAN_I_path" to the morph_config dictionary
morph_config = {
    "LMA_path": config["dataset_paths"]["LMA_path"],
    "LMA_UBO_path": config["dataset_paths"]["LMA_UBO_path"],
    "MIPGAN_I_path": config["dataset_paths"]["MIPGAN_I_path"],
    "MIPGAN_II_path": config["dataset_paths"]["MIPGAN_II_path"],
    "MorDiff_path": config["dataset_paths"]["MorDiff_path"],
    "StyleGAN_path": config["dataset_paths"]["StyleGAN_path"],
    "LMA_MIPGAN_I_path": config["dataset_paths"]["LMA_MIPGAN_I_path"]  # Add this line
}

# Line 95-103: Add "LMA_MIPGAN_I_path" to the morph_config dictionary
morph_config = {
    "LMA_path": datasets_dir,
    "LMA_UBO_path": datasets_dir,
    "MIPGAN_I_path": datasets_dir,
    "MIPGAN_II_path": datasets_dir,
    "MorDiff_path": datasets_dir,
    "StyleGAN_path": datasets_dir,
    "LMA_MIPGAN_I_path": datasets_dir  # Add this line
}
```

### 2.12. Update `run_experiments_direct.py`

```python
# Line 550-560: Add "LMA_MIPGAN_I_path" to the dataset_path_args construction
dataset_path_args = f" -LMA_path {datasets_dir} -LMA_UBO_path {datasets_dir} -MIPGAN_I_path {datasets_dir} -MIPGAN_II_path {datasets_dir} -MorDiff_path {datasets_dir} -StyleGAN_path {datasets_dir} -LMA_MIPGAN_I_path {datasets_dir}"
```

## 3. Testing the Changes

After making all the changes, you should test the integration by:

1. Training a model on the new dataset using SelfMAD-siam:
   ```bash
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models siam
   ```

2. Training a model on the new dataset using SelfMAD-main:
   ```bash
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models main
   ```

3. Training models on both codebases:
   ```bash
   python run_experiments_direct.py --datasets LMA_MIPGAN_I --run-models both
   ```

## 4. Conclusion

By making these changes, you will fully integrate the LMA_MIPGAN_I dataset into both the SelfMAD-siam and SelfMAD-main codebases. The dataset will be available for training and evaluation just like the existing datasets.
