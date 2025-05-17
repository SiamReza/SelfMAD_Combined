#!/bin/bash

# Script to test the trained ViT-MAE model on morph datasets

# Default values
MODEL_TYPE="vit_mae_large"
VERBOSE=true

# Create output directory structure if it doesn't exist
mkdir -p ./output/model
mkdir -p ./output/train
mkdir -p ./output/eval

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
  echo "Error: Model path not specified. Use --model_path option."
  exit 1
fi

# Set verbose flag
if [ "$VERBOSE" = true ]; then
  VERBOSE_FLAG="-v"
else
  VERBOSE_FLAG=""
fi

# Set dataset parameters
DATASET_PARAMS=""
if [ ! -z "$DATASETS" ]; then
  IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
  for dataset in "${DATASET_ARRAY[@]}"; do
    DATASET_PARAMS="$DATASET_PARAMS -${dataset}_path ../datasets"
  done
else
  # If no specific datasets are provided, test on all available datasets
  DATASET_PARAMS="-LMA_path ../datasets -LMA_UBO_path ../datasets -MIPGAN_I_path ../datasets -MIPGAN_II_path ../datasets -MorDiff_path ../datasets -StyleGAN_path ../datasets"
fi

# Print testing configuration
echo "Testing configuration:"
echo "----------------------"
echo "Model path: $MODEL_PATH"
echo "Model type: $MODEL_TYPE"
echo "Verbose: $VERBOSE"
echo "Datasets: ${DATASETS:-All available datasets}"
echo "----------------------"

# Run the evaluation script
python eval__.py \
  -m "$MODEL_TYPE" \
  -p "$MODEL_PATH" \
  $VERBOSE_FLAG \
  $DATASET_PARAMS

echo "Testing completed!"
