#!/bin/bash

# Script to train the ViT-MAE model on morph datasets

# Default values
MODEL="vit_mae_large"
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=5e-4
SAVING_STRATEGY="testset_best"
TRAIN_VAL_SPLIT=0.8
EARLY_STOPPING_PATIENCE=5
EARLY_STOPPING_MONITOR="val_loss"
# Use absolute paths for output directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SAVE_PATH="$ROOT_DIR/output/siam/vit_mae_large/model"
CSV_OUTPUT_PATH="$ROOT_DIR/output/train"
RESUME_TRAINING=false
RESUME_CHECKPOINT=""

# Create output directory structure if it doesn't exist
mkdir -p "$ROOT_DIR/output/siam/vit_mae_large/model"
mkdir -p "$ROOT_DIR/output/train"
mkdir -p "$ROOT_DIR/output/siam/vit_mae_large/eval"

echo "Using absolute paths:"
echo "Save path: $SAVE_PATH"
echo "CSV output path: $CSV_OUTPUT_PATH"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --saving_strategy)
      SAVING_STRATEGY="$2"
      shift 2
      ;;
    --train_val_split)
      TRAIN_VAL_SPLIT="$2"
      shift 2
      ;;
    --early_stopping_patience)
      EARLY_STOPPING_PATIENCE="$2"
      shift 2
      ;;
    --early_stopping_monitor)
      EARLY_STOPPING_MONITOR="$2"
      shift 2
      ;;
    --save_path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --csv_output_path)
      CSV_OUTPUT_PATH="$2"
      shift 2
      ;;
    --resume)
      RESUME_TRAINING=true
      shift
      ;;
    --resume_checkpoint)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if dataset is provided
if [ -z "$DATASET" ]; then
  echo "Error: Dataset not specified. Use --dataset option."
  echo "Available datasets: LMA, LMA_UBO, MIPGAN_I, MIPGAN_II, MorDiff, StyleGAN, LMA_MIPGAN_I"
  exit 1
fi

# Create session name with timestamp
SESSION_NAME="${DATASET}_${MODEL}_$(date +%Y%m%d_%H%M%S)"

# Print training configuration
echo "Training configuration:"
echo "----------------------"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Saving strategy: $SAVING_STRATEGY"
echo "Train/val split: $TRAIN_VAL_SPLIT"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "Early stopping monitor: $EARLY_STOPPING_MONITOR"
echo "Save path: $SAVE_PATH"
echo "CSV output path: $CSV_OUTPUT_PATH"
echo "Session name: $SESSION_NAME"
echo "Resume training: $RESUME_TRAINING"
if [ "$RESUME_TRAINING" = true ] && [ -n "$RESUME_CHECKPOINT" ]; then
  echo "Resume checkpoint: $RESUME_CHECKPOINT"
fi
echo "----------------------"

# Run the training script
# Prepare resume arguments
if [ "$RESUME_TRAINING" = true ]; then
  RESUME_ARG="-resume"
  if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_CHECKPOINT_ARG="-resume_checkpoint $RESUME_CHECKPOINT"
  else
    RESUME_CHECKPOINT_ARG=""
  fi
else
  RESUME_ARG=""
  RESUME_CHECKPOINT_ARG=""
fi

python train__.py \
  -n "$SESSION_NAME" \
  -m "$MODEL" \
  -b "$BATCH_SIZE" \
  -e "$EPOCHS" \
  -lr "$LEARNING_RATE" \
  -v "$SAVING_STRATEGY" \
  -t "$DATASET" \
  -s "$SAVE_PATH" \
  -train_val_split "$TRAIN_VAL_SPLIT" \
  -csv_output_path "$CSV_OUTPUT_PATH" \
  -early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  -early_stopping_monitor "$EARLY_STOPPING_MONITOR" \
  $RESUME_ARG $RESUME_CHECKPOINT_ARG

echo "Training completed!"
