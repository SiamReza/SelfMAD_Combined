# Running the Morph Detection Code

This document explains how to run the code for training and testing the ViT-MAE model on morph datasets.

## Prerequisites

- Python 3.6+
- PyTorch 1.7+
- CUDA-enabled GPU (recommended)
- Required Python packages (install via pip):
  - torch
  - torchvision
  - numpy
  - pandas
  - opencv-python
  - scikit-learn
  - tqdm
  - efficientnet_pytorch
  - timm
  - transformers

## Dataset Structure

The code expects the following dataset structure:

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

## Configuration Files

The code uses several configuration files:

1. `configs/train_config.json`: Contains training parameters like model type, batch size, epochs, etc.
2. `configs/morph_config.json`: Contains dataset paths and train/val split ratio.
3. `configs/data_config.json`: Contains paths for the original SelfMAD datasets (not required for our custom datasets).

## Training

### Using the Shell Script

The easiest way to train the model is to use the provided shell script:

```bash
# Make the script executable
chmod +x run_train.sh

# Run training on LMA dataset with default parameters
./run_train.sh --dataset LMA

# Run training with custom parameters
./run_train.sh \
  --dataset LMA \
  --model vit_mae_large \
  --batch_size 32 \
  --epochs 100 \
  --lr 5e-4 \
  --saving_strategy testset_best \
  --train_val_split 0.8 \
  --early_stopping_patience 5 \
  --early_stopping_monitor val_loss \
  --save_path ./output/model \
  --csv_output_path ./output/train
```

### Available Parameters

- `--dataset`: Dataset to train on (LMA, LMA_UBO, MIPGAN_I, MIPGAN_II, MorDiff, StyleGAN)
- `--model`: Model type (vit_mae_large)
- `--batch_size`: Batch size for training
- `--epochs`: Maximum number of epochs
- `--lr`: Learning rate
- `--saving_strategy`: Strategy for saving models (testset_best or original)
- `--train_val_split`: Train/validation split ratio (0.8 = 80% train, 20% validation)
- `--early_stopping_patience`: Number of epochs to wait before early stopping
- `--early_stopping_monitor`: Metric to monitor for early stopping (val_loss, train_loss, val_acc)
- `--save_path`: Path to save trained models
- `--csv_output_path`: Path to save CSV files with image paths and labels

### Using Python Directly

You can also run the training script directly:

```bash
python train__.py \
  -n "LMA_vit_mae_large" \
  -m "vit_mae_large" \
  -b 32 \
  -e 100 \
  -lr 5e-4 \
  -v "testset_best" \
  -t "LMA" \
  -s "./output/model" \
  -train_val_split 0.8 \
  -csv_output_path "./output/train" \
  -early_stopping_patience 5 \
  -early_stopping_monitor "val_loss"
```

## Testing

### Using the Shell Script

The easiest way to test the model is to use the provided shell script:

```bash
# Make the script executable
chmod +x run_test.sh

# Test on all available datasets
./run_test.sh --model_path ./output/model/LMA_vit_mae_large_20230501_120000/weights/early_stopped_best.tar

# Test on specific datasets
./run_test.sh \
  --model_path ./output/model/LMA_vit_mae_large_20230501_120000/weights/early_stopped_best.tar \
  --model_type vit_mae_large \
  --verbose true \
  --datasets LMA,LMA_UBO,MIPGAN_I
```

### Available Parameters

- `--model_path`: Path to the trained model file
- `--model_type`: Type of the model (vit_mae_large)
- `--verbose`: Whether to print detailed information (true or false)
- `--datasets`: Comma-separated list of datasets to test on (if not specified, tests on all available datasets)

### Using Python Directly

You can also run the testing script directly:

```bash
python eval__.py \
  -m "vit_mae_large" \
  -p "./output/model/LMA_vit_mae_large_20230501_120000/weights/early_stopped_best.tar" \
  -v \
  -LMA_path ../datasets \
  -LMA_UBO_path ../datasets \
  -MIPGAN_I_path ../datasets \
  -MIPGAN_II_path ../datasets \
  -MorDiff_path ../datasets \
  -StyleGAN_path ../datasets
```

## Inference on a Single Image

To run inference on a single image:

```bash
python infer__.py \
  -m "vit_mae_large" \
  -p "./output/model/LMA_vit_mae_large_20230501_120000/weights/early_stopped_best.tar" \
  -in "path/to/image.jpg"
```

## Early Stopping

The training script includes an early stopping mechanism with the following parameters:

- `early_stopping_patience`: Number of epochs to wait before early stopping (default: 5)
- `early_stopping_monitor`: Metric to monitor for early stopping (default: val_loss)
  - Options: val_loss, train_loss, val_acc

When early stopping is triggered, the best model (based on the monitored metric) is saved to `early_stopped_best.tar` in the weights directory.

## Output Directory Structure

The code organizes all outputs in the following directory structure:

```
output/
├─ model/  # Contains trained models
├─ train/  # Contains CSV files with image paths and labels
└─ eval/   # Contains evaluation results in JSON format
```

The training script creates the following output files:

- Trained models in `output/model/[session_name]/weights/`
- CSV files with image paths and labels in `output/train/`
- Log files with training and validation metrics in `output/model/[session_name]/logs/`

The evaluation script creates the following output files:

- Evaluation results in JSON format in `output/eval/`

## Cross-Dataset Evaluation

The testing script evaluates the trained model on all specified datasets, allowing for cross-dataset generalization assessment. It calculates metrics such as EER (Equal Error Rate) and AUC (Area Under Curve) for each dataset.
