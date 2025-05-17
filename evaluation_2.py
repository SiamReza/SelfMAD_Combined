"""
Main evaluation functions for SelfMAD project.

This module provides the main functions for evaluating models on various datasets.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
from datetime import datetime

# Import functions from evaluation_1
from evaluation_1 import (
    calculate_metrics,
    separate_by_class
)

def evaluate_model(model, test_loaders, device, output_dir=None, model_name=None,
                model_dataset=None, model_type=None, epoch_number=None,
                verbose=False, classification_threshold=0.5, apcer_thresholds=None,
                model_params=None, training_params=None):
    """
    Evaluate a model on multiple test datasets and calculate various metrics.

    Args:
        model: The PyTorch model to evaluate
        test_loaders: Dictionary of test data loaders
        device: Device to run evaluation on (cuda or cpu)
        output_dir: Directory to save evaluation results
        model_name: Name of the model (used for saving results)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        verbose: Whether to print verbose output
        classification_threshold: Threshold for binary classification (default: 0.5)
        apcer_thresholds: List of APCER thresholds for BPCER calculation (default: [0.05, 0.10, 0.20])
        model_params: Dictionary of model parameters (e.g., num_parameters)
        training_params: Dictionary of training parameters (e.g., learning_rate, batch_size)

    Returns:
        Dictionary of evaluation results, dictionary of targets/outputs for visualization,
        and dictionary of class-separated outputs for bonafide vs morph analysis
    """
    # Start timing for the entire evaluation
    eval_start_time = time.time()

    # Set default APCER thresholds if not provided
    if apcer_thresholds is None:
        apcer_thresholds = [0.05, 0.10, 0.20]

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results = {}

    # Initialize dictionaries for visualization
    targets_outputs_dict = {}
    class_separated_dict = {}

    # Evaluate on each dataset
    all_metrics = []
    for dataset_name, method_loaders in test_loaders.items():
        if verbose:
            print(f"Evaluating on {dataset_name}...")

        # Initialize dataset results
        dataset_results = {}

        # Initialize lists for targets and outputs
        all_targets = []
        all_outputs = []

        # Evaluate on each method
        for method_name, loader in method_loaders.items():
            if verbose:
                print(f"  Method: {method_name}")

            # Initialize lists for targets and outputs
            method_targets = []
            method_outputs = []

            # Evaluate on each batch
            for data in tqdm(loader, desc=f"Evaluating {dataset_name}_{method_name}", disable=not verbose):
                # Handle different data formats: tuple, dictionary, or list
                try:
                    if isinstance(data, tuple):
                        img = data[0].to(device, non_blocking=True).float()
                        target = data[1].to(device, non_blocking=True).long()
                    elif isinstance(data, dict):
                        img = data['img'].to(device, non_blocking=True).float()
                        target = data['label'].to(device, non_blocking=True).long()
                    elif isinstance(data, list):
                        img = data[0].to(device, non_blocking=True).float()
                        target = data[1].to(device, non_blocking=True).long()
                    else:
                        print(f"Warning: Unexpected data format: {type(data)}. Skipping batch.")
                        continue

                    # Check for NaN or Inf values in the image
                    if torch.isnan(img).any() or torch.isinf(img).any():
                        print(f"Warning: Image contains NaN or Inf values. Skipping batch.")
                        continue

                except Exception as e:
                    print(f"Error processing data batch: {str(e)}. Data type: {type(data)}. Skipping batch.")
                    continue

                # Forward pass
                try:
                    with torch.no_grad():
                        raw_output = model(img)

                        # Process output based on model type
                        if model_type == "vit_mae_large":
                            # For ViT-MAE with sigmoid activation (already applied in model's forward method)
                            # The output should have shape [batch_size, 1]
                            output = raw_output.squeeze(1).detach().cpu().numpy()
                        else:
                            # For other models using softmax
                            # The output should have shape [batch_size, num_classes]
                            output = raw_output.softmax(dim=1)[:, 1].detach().cpu().numpy()
                except Exception as e:
                    print(f"Error during model forward pass: {str(e)}. Skipping batch.")
                    continue

                # Collect targets and outputs
                method_targets.extend(target.cpu().data.numpy())
                method_outputs.extend(output)

            # Calculate metrics for this method
            method_metrics = calculate_metrics(
                method_targets,
                method_outputs,
                threshold=classification_threshold,
                apcer_thresholds=apcer_thresholds
            )

            # Add method name to metrics
            method_metrics["method"] = method_name

            # Add to dataset results
            dataset_results[method_name] = method_metrics

            # Add to all targets and outputs
            all_targets.extend(method_targets)
            all_outputs.extend(method_outputs)

        # Calculate metrics for the entire dataset
        dataset_metrics = calculate_metrics(
            all_targets,
            all_outputs,
            threshold=classification_threshold,
            apcer_thresholds=apcer_thresholds
        )

        # Add dataset name to metrics
        dataset_metrics["dataset"] = dataset_name

        # Add to results
        results[dataset_name] = dataset_metrics

        # Add to all metrics for mean calculation
        all_metrics.append(dataset_metrics)

        # Add to targets_outputs_dict for visualization
        targets_outputs_dict[dataset_name] = (all_targets, all_outputs)

        # Add to class_separated_dict for visualization
        class_separated_dict[dataset_name] = separate_by_class(all_targets, all_outputs)

        # Print dataset results if verbose
        if verbose:
            print(f"Results for {dataset_name}:")
            print(f"  Accuracy: {dataset_metrics['accuracy']:.4f}")
            print(f"  AUC: {dataset_metrics['auc']:.4f}")
            print(f"  EER: {dataset_metrics['eer']:.4f}")
            print(f"  APCER: {dataset_metrics['apcer']:.4f}")
            print(f"  BPCER: {dataset_metrics['bpcer']:.4f}")
            print(f"  ACER: {dataset_metrics['acer']:.4f}")

    # Calculate mean metrics across all datasets
    if all_metrics:
        mean_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "auc": np.mean([m["auc"] for m in all_metrics]),
            "eer": np.mean([m["eer"] for m in all_metrics]),
            "apcer": np.mean([m["apcer"] for m in all_metrics]),
            "bpcer": np.mean([m["bpcer"] for m in all_metrics]),
            "acer": np.mean([m["acer"] for m in all_metrics]),
            "dataset": "mean"
        }

        # Add APCER metrics to mean_metrics
        if all_metrics:
            for key in all_metrics[0].keys():
                if key not in mean_metrics and key != "dataset" and key != "bonafide_count" and key != "morph_count":
                    try:
                        # Use a list comprehension with error handling to get values
                        values = [m.get(key, 0) for m in all_metrics if key in m]
                        if values:
                            mean_metrics[key] = np.mean(values)
                        else:
                            mean_metrics[key] = 0
                    except Exception as e:
                        print(f"Error calculating mean for {key}: {str(e)}. Using default value 0.")
                        mean_metrics[key] = 0

        # Ensure mean_metrics has all required keys
        required_keys = ["accuracy", "auc", "eer", "apcer", "bpcer", "acer",
                         "bonafide_mean", "morph_mean", "bonafide_std", "morph_std"]
        for key in required_keys:
            if key not in mean_metrics:
                mean_metrics[key] = 0.5 if key in ["auc", "eer"] else 0

        results["mean"] = mean_metrics

        # Print mean results if verbose
        if verbose:
            print("Mean metrics:")
            print(f"  Accuracy: {mean_metrics['accuracy']:.4f}")
            print(f"  Mean bonafide score: {mean_metrics['bonafide_mean']:.4f}")
            print(f"  Mean morph score: {mean_metrics['morph_mean']:.4f}")

    # End timing for the entire evaluation
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time

    # Add total evaluation time to mean metrics
    if "mean" in results:
        results["mean"]["total_time"] = eval_time

    # Save results to CSV
    if output_dir:
        # Create detailed CSV with all metrics
        detailed_df = pd.DataFrame()
        for dataset_name, metrics in results.items():
            row = pd.DataFrame([metrics])
            detailed_df = pd.concat([detailed_df, row], ignore_index=True)

        # Add model information to detailed CSV
        detailed_df["model_name"] = model_name
        detailed_df["model_dataset"] = model_dataset
        detailed_df["model_type"] = model_type
        detailed_df["epoch_number"] = epoch_number
        detailed_df["classification_threshold"] = classification_threshold
        detailed_df["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add model parameters if provided
        if model_params:
            for key, value in model_params.items():
                detailed_df[f"model_{key}"] = value

        # Add training parameters if provided
        if training_params:
            for key, value in training_params.items():
                detailed_df[f"training_{key}"] = value

        # Save detailed CSV
        detailed_csv_path = os.path.join(output_dir, f"{model_name}_detailed.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)

        # Create metrics CSV with only the most important metrics
        metrics_df = pd.DataFrame()
        for dataset_name, metrics in results.items():
            row = {
                "dataset": dataset_name,
                "accuracy": metrics["accuracy"],
                "auc": metrics.get("auc", 0.5),
                "eer": metrics.get("eer", 0.5),
                "apcer": metrics.get("apcer", 0),
                "bpcer": metrics.get("bpcer", 0),
                "acer": metrics.get("acer", 0),
                "bonafide_mean": metrics.get("bonafide_mean", 0),
                "morph_mean": metrics.get("morph_mean", 0),
                "bonafide_count": metrics.get("bonafide_count", 0),
                "morph_count": metrics.get("morph_count", 0)
            }

            # Add BPCER at different APCER thresholds
            for key in metrics.keys():
                if key.startswith("bpcer_at_apcer_"):
                    row[key] = metrics[key]

            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

        # Add model information to metrics CSV
        metrics_df["model_name"] = model_name
        metrics_df["model_dataset"] = model_dataset
        metrics_df["model_type"] = model_type
        metrics_df["epoch_number"] = epoch_number
        metrics_df["classification_threshold"] = classification_threshold
        metrics_df["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics CSV
        metrics_csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Save combined CSV if this is not the first evaluation
        combined_detailed_csv_path = os.path.join(output_dir, "combined_detailed.csv")
        combined_metrics_csv_path = os.path.join(output_dir, "combined_metrics.csv")

        if os.path.exists(combined_detailed_csv_path):
            # Load existing combined CSV
            combined_detailed_df = pd.read_csv(combined_detailed_csv_path)
            # Append new results
            combined_detailed_df = pd.concat([combined_detailed_df, detailed_df], ignore_index=True)
            # Save combined CSV
            combined_detailed_df.to_csv(combined_detailed_csv_path, index=False)
        else:
            # Save as combined CSV
            detailed_df.to_csv(combined_detailed_csv_path, index=False)

        if os.path.exists(combined_metrics_csv_path):
            # Load existing combined CSV
            combined_metrics_df = pd.read_csv(combined_metrics_csv_path)
            # Append new results
            combined_metrics_df = pd.concat([combined_metrics_df, metrics_df], ignore_index=True)
            # Save combined CSV
            combined_metrics_df.to_csv(combined_metrics_csv_path, index=False)
        else:
            # Save as combined CSV
            metrics_df.to_csv(combined_metrics_csv_path, index=False)

    return results, targets_outputs_dict, class_separated_dict

# Function to be called from train__.py and eval__.py
def evaluate(model, test_loaders, device, output_dir=None, model_name=None,
           model_dataset=None, model_type=None, epoch_number=None,
           verbose=False, classification_threshold=0.5, apcer_thresholds=None,
           model_params=None, training_params=None):
    """
    Main evaluation function to be called from train__.py and eval__.py.

    Args:
        model: The PyTorch model to evaluate
        test_loaders: Dictionary of test data loaders
        device: Device to run evaluation on (cuda or cpu)
        output_dir: Directory to save evaluation results
        model_name: Name of the model (used for saving results)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        verbose: Whether to print verbose output
        classification_threshold: Threshold for binary classification (default: 0.5)
        apcer_thresholds: List of APCER thresholds for BPCER calculation (default: [0.05, 0.10, 0.20])
        model_params: Dictionary of model parameters (e.g., num_parameters)
        training_params: Dictionary of training parameters (e.g., learning_rate, batch_size)

    Returns:
        Dictionary of evaluation results, dictionary of targets/outputs for visualization,
        and dictionary of class-separated outputs for bonafide vs morph analysis
    """
    # Set default APCER thresholds if not provided
    if apcer_thresholds is None:
        apcer_thresholds = [0.05, 0.10, 0.20]

    return evaluate_model(
        model, test_loaders, device, output_dir, model_name,
        model_dataset, model_type, epoch_number,
        verbose, classification_threshold, apcer_thresholds,
        model_params, training_params
    )
