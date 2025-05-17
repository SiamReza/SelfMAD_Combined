#!/usr/bin/env python3
"""
Evaluation script for SelfMAD-main models.

This script evaluates a trained model on various datasets and generates metrics and visualizations.
It uses the robust_model_handler module for improved model file handling.
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
from utils.model import Detector
from utils.dataset import default_datasets
from evaluation_1 import calculate_precision, calculate_recall, calculate_f1, calculate_auc, calculate_eer, calculate_apcer_bpcer_acer
from evaluation_2 import evaluate
# Import the visualization modules
from visualization_1 import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from visualization_2 import plot_det_curve, plot_score_distributions, plot_threshold_analysis
from visualization_3 import visualize

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add the parent directory to the path to import the robust_model_handler module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robust_model_handler import normalize_path, find_model_file, copy_model_file, load_model_with_retry

def prep_dataloaders(test_datasets, batch_size):
    """Prepare data loaders for evaluation."""
    test_loaders = {
        dataset: {
            method: torch.utils.data.DataLoader(test_datasets[dataset][method],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
            for method in test_datasets[dataset]
        }
        for dataset in test_datasets
    }
    return test_loaders

def main(eval_config):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model path and normalize it
    model_state_path = normalize_path(eval_config["model_path"])

    # Extract dataset and model type information for better model handling
    model_type = eval_config["model_type"]
    model_dataset = eval_config.get("train_dataset", "unknown")

    # Try to extract dataset from model path if not provided
    if model_dataset == "unknown" and "_main_" in model_state_path:
        try:
            model_dataset = os.path.basename(model_state_path).split("_main_")[0]
        except:
            pass

    # Load the model state with retry mechanism
    model_state = load_model_with_retry(model_state_path, "main", model_dataset)

    # If model loading failed, try to find an alternative model file
    if model_state is None:
        print(f"Failed to load model from {model_state_path}. Trying to find an alternative model file...")
        alt_model_path = find_model_file(os.path.dirname(os.path.dirname(model_state_path)), "main", model_dataset)
        if alt_model_path:
            print(f"Found alternative model file: {alt_model_path}")
            model_state = load_model_with_retry(alt_model_path, "main", model_dataset)
            if model_state:
                model_state_path = alt_model_path

    # If still failed, raise an error
    if model_state is None:
        raise FileNotFoundError(f"Could not load model from {model_state_path} or any alternative path.")

    # Detect model architecture from state dictionary
    if 'model' in model_state:
        model_state_dict = model_state['model']
    else:
        model_state_dict = model_state

    # Check for EfficientNet architecture
    if any(key.startswith('net._conv_stem') for key in model_state_dict):
        print(f"Detected EfficientNet architecture in model state dictionary")
        model_type = "efficientnet-b4"
    # Check for EfficientNet-B7 (has more blocks than B4)
    elif any(key.startswith('net._blocks.30') for key in model_state_dict):
        print(f"Detected EfficientNet-B7 architecture in model state dictionary")
        model_type = "efficientnet-b7"
    # Check for HRNet architecture
    elif any(key.startswith('net.conv1') for key in model_state_dict):
        print(f"Detected HRNet architecture in model state dictionary")
        model_type = "hrnet_w18"
    # Check for Swin architecture
    elif any(key.startswith('net.features') for key in model_state_dict):
        print(f"Detected Swin architecture in model state dictionary")
        model_type = "swin"
    # Check for ResNet architecture
    elif any(key.startswith('net.layer1') for key in model_state_dict):
        print(f"Detected ResNet architecture in model state dictionary")
        model_type = "resnet"

    print(f"Using model_type: {model_type}")

    # Initialize the model with detected architecture
    model = Detector(model=model_type)

    # Load the model state
    if 'model' in model_state:
        model.load_state_dict(model_state['model'])
    else:
        model.load_state_dict(model_state)
    model.train(mode=False)
    model.to(device)

    # Set image size based on model type
    image_size = 384 if "hrnet" in model_type else 380
    batch_size = 32

    # ORIGINAL DATASETS (FRGC, FERET, FRLL)
    # Skipping evaluation on original datasets as they are not available
    print("Skipping evaluation on original datasets (FRGC, FERET, FRLL)...")

    # Extract model name and create output directory structure
    model_path_parts = os.path.basename(model_state_path).split('.')
    model_name = model_path_parts[0]

    # Remove epoch number if present to avoid epoch-specific directories
    if model_name.startswith("epoch_"):
        # Extract the base model name from the directory structure
        model_dir = os.path.dirname(model_state_path)
        base_dir = os.path.dirname(model_dir)
        model_name = os.path.basename(base_dir)

    repo_name = "main"  # This is the SelfMAD-main repository

    # Create centralized output directory structure
    # If output directory is provided via command line, use that instead
    if args.output_dir:
        eval_dir = normalize_path(args.output_dir)
    else:
        output_dir = normalize_path(os.path.join("..", "output", repo_name, model_name))
        eval_dir = os.path.join(output_dir, "eval")

    os.makedirs(eval_dir, exist_ok=True)

    # CUSTOM MORPH DATASETS
    custom_test_datasets = default_datasets(image_size, datasets="custom_morph", config=eval_config)

    if custom_test_datasets:
        custom_test_loaders = prep_dataloaders(custom_test_datasets, batch_size)

        # Try to extract epoch number from model path
        epoch_number = None
        if "epoch_" in model_state_path:
            try:
                epoch_number = int(os.path.basename(model_state_path).split("epoch_")[1].split(".")[0])
            except (IndexError, ValueError):
                epoch_number = None

        # If epoch number not in path, try to get it from model state
        if epoch_number is None and 'epoch' in model_state:
            epoch_number = model_state['epoch']

        # Get model parameters
        model_params = {
            "num_parameters": sum(p.numel() for p in model.parameters())
        }

        # Use the evaluation module
        results, targets_outputs_dict, class_separated_dict = evaluate(
            model=model,
            test_loaders=custom_test_loaders,
            device=device,
            output_dir=eval_dir,
            model_name=model_name,
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            verbose=eval_config["verbose"],
            classification_threshold=eval_config.get("classification_threshold", 0.5),
            apcer_thresholds=eval_config.get("apcer_thresholds", [0.05, 0.10, 0.20]),
            model_params=model_params
        )

        # Use the visualization module to generate plots
        visualize(
            results=results,
            targets_outputs_dict=targets_outputs_dict,
            class_separated_dict=class_separated_dict,
            output_dir=eval_dir,
            model_name=model_name,
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number
        )

        print(f"Evaluation results saved to {eval_dir}")

        # Create marker file for evaluation completion
        marker_info = {
            "model_path": model_state_path,
            "model_type": model_type,
            "model_dataset": model_dataset,
            "epoch_number": epoch_number,
            "datasets_evaluated": list(custom_test_datasets.keys()),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        with open(os.path.join(eval_dir, "evaluation_complete.marker"), "w") as f:
            json.dump(marker_info, f, cls=NumpyEncoder)
    else:
        print("No custom datasets found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on various datasets.")
    parser.add_argument('-m', '--model_type', type=str, required=True, help="Model type (e.g., hrnet_w18)")
    parser.add_argument('-p', '--model_path', type=str, required=True, help="Path to the model file")
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory for evaluation results")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose output")
    parser.add_argument('-t', '--classification_threshold', type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument('-apcer_thresholds', nargs='+', type=float, default=[0.05, 0.10, 0.20], help="APCER thresholds for BPCER calculation")

    # Custom morph dataset parameters
    parser.add_argument('-LMA_path', type=str, required=False)
    parser.add_argument('-LMA_UBO_path', type=str, required=False)
    parser.add_argument('-MIPGAN_I_path', type=str, required=False)
    parser.add_argument('-MIPGAN_II_path', type=str, required=False)
    parser.add_argument('-MorDiff_path', type=str, required=False)
    parser.add_argument('-StyleGAN_path', type=str, required=False)
    args = parser.parse_args()

    # Load base data configuration
    eval_config = json.load(open("./configs/data_config.json"))
    for key in vars(args):
        if vars(args)[key] is not None:
            eval_config[key] = vars(args)[key]

    # Load morph dataset configuration if needed
    if any(key.startswith(prefix) for prefix in ["LMA", "MIPGAN", "MorDiff", "StyleGAN"] for key in eval_config):
        try:
            morph_config = json.load(open("./configs/morph_config.json"))
            for key in morph_config:
                if key not in eval_config:
                    eval_config[key] = morph_config[key]
        except FileNotFoundError:
            print("Warning: morph_config.json not found. Using default values.")

    main(eval_config)
