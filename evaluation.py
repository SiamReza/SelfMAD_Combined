"""
Evaluation module for SelfMAD project.

This module provides functions for evaluating models on various datasets and calculating metrics.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import torch
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from datetime import datetime

def calculate_accuracy(targets, outputs, threshold=0.5):
    """Calculate accuracy from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_accuracy. Returning default value 0.")
        return 0

    try:
        preds = (np.array(outputs) > threshold).astype(int)
        return accuracy_score(targets, preds)
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}. Returning default value 0.")
        return 0

def calculate_precision_recall_f1(targets, outputs, threshold=0.5):
    """Calculate precision, recall, and F1 score from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_precision_recall_f1. Returning default values (0, 0, 0).")
        return 0, 0, 0

    try:
        preds = (np.array(outputs) > threshold).astype(int)

        # Handle case where all predictions are one class
        if len(np.unique(preds)) < 2 or len(np.unique(targets)) < 2:
            print("Warning: Only one class present in predictions or targets. Metrics may be undefined.")
            # If all predictions are negative, precision is undefined (set to 0)
            if np.sum(preds) == 0:
                return 0, 0, 0

        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        return precision, recall, f1
    except Exception as e:
        print(f"Error calculating precision/recall/f1: {str(e)}. Returning default values (0, 0, 0).")
        return 0, 0, 0

def calculate_auc_roc(targets, outputs):
    """Calculate AUC-ROC from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_auc_roc. Returning default value 0.5.")
        return 0.5

    # Check if all targets are the same class (only one class present)
    if len(np.unique(targets)) < 2:
        print(f"Warning: Only one class present in targets for AUC calculation. Returning default value 0.5.")
        return 0.5

    try:
        return roc_auc_score(targets, outputs)
    except Exception as e:
        print(f"Error calculating AUC-ROC: {str(e)}. Returning default value 0.5.")
        return 0.5

def calculate_eer(targets, outputs):
    """Calculate Equal Error Rate (EER) from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_eer. Returning default value 0.5.")
        return 0.5

    # Check if all targets are the same class (only one class present)
    if len(np.unique(targets)) < 2:
        print(f"Warning: Only one class present in targets for EER calculation. Returning default value 0.5.")
        return 0.5

    try:
        fpr, tpr, _ = roc_curve(targets, outputs)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        return eer
    except Exception as e:
        print(f"Error calculating EER: {str(e)}. Returning default value 0.5.")
        return 0.5

def calculate_specificity(targets, outputs, threshold=0.5):
    """Calculate specificity (true negative rate) from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_specificity. Returning default value 0.")
        return 0

    try:
        preds = (np.array(outputs) > threshold).astype(int)
        tn = np.sum((preds == 0) & (np.array(targets) == 0))
        fp = np.sum((preds == 1) & (np.array(targets) == 0))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        print(f"Error calculating specificity: {str(e)}. Returning default value 0.")
        return 0

def calculate_mcc(targets, outputs, threshold=0.5):
    """Calculate Matthews Correlation Coefficient from model outputs and targets."""
    preds = (np.array(outputs) > threshold).astype(int)
    tp = np.sum((preds == 1) & (np.array(targets) == 1))
    tn = np.sum((preds == 0) & (np.array(targets) == 0))
    fp = np.sum((preds == 1) & (np.array(targets) == 0))
    fn = np.sum((preds == 0) & (np.array(targets) == 1))

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0

def calculate_confidence_interval(metric_value, n_samples, confidence=0.95):
    """Calculate confidence interval for a metric using the normal approximation."""
    z = 1.96  # z-score for 95% confidence
    std_err = np.sqrt((metric_value * (1 - metric_value)) / n_samples)
    margin = z * std_err
    return (metric_value - margin, metric_value + margin)

def calculate_apcer_bpcer_acer(targets, outputs, threshold=0.5, apcer_thresholds=None):
    """
    Calculate APCER, BPCER, and ACER metrics, plus BPCER at different APCER thresholds.

    Args:
        targets: Ground truth labels (0 for bonafide, 1 for morph)
        outputs: Model prediction scores
        threshold: Threshold for binary classification
        apcer_thresholds: List of APCER thresholds for BPCER calculation (e.g., [0.05, 0.10, 0.20])

    Returns:
        Dictionary with APCER, BPCER, ACER values and BPCER at different APCER thresholds
    """
    # Convert to numpy arrays
    targets = np.array(targets)
    outputs = np.array(outputs)
    preds = (outputs > threshold).astype(int)

    # APCER: Attack Presentation Classification Error Rate (proportion of attack samples incorrectly classified as bona fide)
    attack_indices = np.where(targets == 1)[0]
    if len(attack_indices) > 0:
        apcer = 1 - np.mean(preds[attack_indices])
    else:
        apcer = 0

    # BPCER: Bona Fide Presentation Classification Error Rate (proportion of bona fide samples incorrectly classified as attack)
    bonafide_indices = np.where(targets == 0)[0]
    if len(bonafide_indices) > 0:
        bpcer = np.mean(preds[bonafide_indices])
    else:
        bpcer = 0

    # ACER: Average Classification Error Rate (average of APCER and BPCER)
    acer = (apcer + bpcer) / 2

    # Initialize result dictionary
    metrics = {
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer
    }

    # Calculate BPCER at different APCER thresholds if provided
    if apcer_thresholds is not None:
        # Get bonafide and morph scores
        bonafide_scores = outputs[bonafide_indices]
        morph_scores = outputs[attack_indices]

        for apcer_threshold in apcer_thresholds:
            # Find threshold that gives the desired APCER
            if len(morph_scores) > 0:
                # Sort morph scores in ascending order
                sorted_morph_scores = np.sort(morph_scores)
                # Find the index corresponding to the APCER threshold
                idx = int(np.ceil(apcer_threshold * len(sorted_morph_scores))) - 1
                if idx < 0:
                    idx = 0
                # Get the threshold value
                threshold_at_apcer = sorted_morph_scores[idx]

                # Calculate BPCER at this threshold
                bpcer_at_apcer = np.mean(bonafide_scores > threshold_at_apcer) if len(bonafide_scores) > 0 else 0
                metrics[f"bpcer_{int(apcer_threshold*100)}"] = bpcer_at_apcer
            else:
                metrics[f"bpcer_{int(apcer_threshold*100)}"] = 0

    return metrics

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
        Dictionary of evaluation results and dictionary of targets/outputs for visualization
    """
    # Start timing for the entire evaluation
    eval_start_time = time.time()

    # Set default APCER thresholds if not provided
    if apcer_thresholds is None:
        apcer_thresholds = [0.05, 0.10, 0.20]
    results = {}
    all_metrics = []
    targets_outputs_dict = {}
    class_separated_dict = {}  # New dictionary to store class-separated outputs

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Create metrics and plots directories directly in the output directory
        # This avoids creating epoch-specific directories
        metrics_dir = os.path.join(output_dir, "metrics")
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    # Evaluate on each dataset
    for dataset_loader in test_loaders:
        for method_loader in test_loaders[dataset_loader]:
            # Start timing for this dataset
            dataset_start_time = time.time()

            output_dict = []
            target_dict = []

            # Collect model outputs and targets
            for data in tqdm(test_loaders[dataset_loader][method_loader], desc=f"Evaluating {dataset_loader}_{method_loader}"):
                # Handle different data formats: tuple, dictionary, or list
                if isinstance(data, tuple):
                    img = data[0].to(device, non_blocking=True).float()
                    target = data[1].to(device, non_blocking=True).long()
                elif isinstance(data, list):
                    img = data[0].to(device, non_blocking=True).float()
                    target = data[1].to(device, non_blocking=True).long()
                elif isinstance(data, dict):
                    img = data['img'].to(device, non_blocking=True).float()
                    target = data['label'].to(device, non_blocking=True).long()
                else:
                    raise TypeError(f"Unsupported data type: {type(data)}")

                with torch.no_grad():
                    output = model(img)

                # Handle multi-class output if needed
                if output.shape[1] > 2:
                    output = torch.cat((output[:, 0].unsqueeze(1), output[:, 1:].sum(1).unsqueeze(1)), dim=1)

                output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
                target_dict += target.cpu().data.numpy().tolist()

            # End timing for this dataset
            dataset_end_time = time.time()
            dataset_time = dataset_end_time - dataset_start_time

            # Store targets and outputs for visualization
            dataset_method = f"{dataset_loader}_{method_loader}"
            targets_outputs_dict[dataset_method] = (target_dict, output_dict)

            # Separate outputs by class (bonafide vs morph)
            bonafide_outputs = [output_dict[i] for i in range(len(target_dict)) if target_dict[i] == 0]
            morph_outputs = [output_dict[i] for i in range(len(target_dict)) if target_dict[i] == 1]

            # Store class-separated outputs
            class_separated_dict[dataset_method] = {
                "bonafide": bonafide_outputs,
                "morph": morph_outputs
            }

            # Calculate class-specific statistics
            bonafide_stats = {
                "mean": np.mean(bonafide_outputs) if bonafide_outputs else 0,
                "std": np.std(bonafide_outputs) if bonafide_outputs else 0,
                "min": np.min(bonafide_outputs) if bonafide_outputs else 0,
                "max": np.max(bonafide_outputs) if bonafide_outputs else 0,
                "count": len(bonafide_outputs)
            }

            morph_stats = {
                "mean": np.mean(morph_outputs) if morph_outputs else 0,
                "std": np.std(morph_outputs) if morph_outputs else 0,
                "min": np.min(morph_outputs) if morph_outputs else 0,
                "max": np.max(morph_outputs) if morph_outputs else 0,
                "count": len(morph_outputs)
            }

            # Calculate metrics
            accuracy = calculate_accuracy(target_dict, output_dict)
            precision, recall, f1 = calculate_precision_recall_f1(target_dict, output_dict)
            specificity = calculate_specificity(target_dict, output_dict)
            mcc = calculate_mcc(target_dict, output_dict)
            auc = calculate_auc_roc(target_dict, output_dict)
            eer = calculate_eer(target_dict, output_dict)

            # Calculate confidence intervals
            n_samples = len(target_dict)
            accuracy_ci = calculate_confidence_interval(accuracy, n_samples)
            precision_ci = calculate_confidence_interval(precision, n_samples)
            recall_ci = calculate_confidence_interval(recall, n_samples)
            specificity_ci = calculate_confidence_interval(specificity, n_samples)

            # Calculate APCER, BPCER, ACER and BPCER at different APCER thresholds
            apcer_metrics = calculate_apcer_bpcer_acer(target_dict, output_dict,
                                                      threshold=classification_threshold,
                                                      apcer_thresholds=apcer_thresholds)

            # Store results
            results[dataset_method] = {
                "accuracy": accuracy,
                "accuracy_ci_lower": accuracy_ci[0],
                "accuracy_ci_upper": accuracy_ci[1],
                "precision": precision,
                "precision_ci_lower": precision_ci[0],
                "precision_ci_upper": precision_ci[1],
                "recall": recall,
                "recall_ci_lower": recall_ci[0],
                "recall_ci_upper": recall_ci[1],
                "specificity": specificity,
                "specificity_ci_lower": specificity_ci[0],
                "specificity_ci_upper": specificity_ci[1],
                "f1": f1,
                "mcc": mcc,
                "auc": auc,
                "eer": eer,
                "bonafide_stats": bonafide_stats,
                "morph_stats": morph_stats,
                "time": dataset_time
            }

            # Add APCER metrics to results
            results[dataset_method].update(apcer_metrics)

            # Add to metrics list for CSV
            metrics_entry = {
                "dataset": dataset_method,
                "accuracy": accuracy,
                "accuracy_ci_lower": accuracy_ci[0],
                "accuracy_ci_upper": accuracy_ci[1],
                "precision": precision,
                "precision_ci_lower": precision_ci[0],
                "precision_ci_upper": precision_ci[1],
                "recall": recall,
                "recall_ci_lower": recall_ci[0],
                "recall_ci_upper": recall_ci[1],
                "specificity": specificity,
                "specificity_ci_lower": specificity_ci[0],
                "specificity_ci_upper": specificity_ci[1],
                "f1": f1,
                "mcc": mcc,
                "auc": auc,
                "eer": eer,
                "bonafide_mean": bonafide_stats["mean"],
                "bonafide_std": bonafide_stats["std"],
                "bonafide_count": bonafide_stats["count"],
                "morph_mean": morph_stats["mean"],
                "morph_std": morph_stats["std"],
                "morph_count": morph_stats["count"],
                "time": dataset_time
            }

            # Add APCER metrics to the entry
            metrics_entry.update(apcer_metrics)

            # Add to all_metrics list
            all_metrics.append(metrics_entry)

            # Print results if verbose
            if verbose:
                print(f"{dataset_method}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Bonafide samples: {bonafide_stats['count']}, Mean score: {bonafide_stats['mean']:.4f}")
                print(f"  Morph samples: {morph_stats['count']}, Mean score: {morph_stats['mean']:.4f}")

    # Calculate mean metrics
    if len(all_metrics) > 0:
        # Start with basic metrics
        mean_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "accuracy_ci_lower": np.mean([m["accuracy_ci_lower"] for m in all_metrics]),
            "accuracy_ci_upper": np.mean([m["accuracy_ci_upper"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "precision_ci_lower": np.mean([m["precision_ci_lower"] for m in all_metrics]),
            "precision_ci_upper": np.mean([m["precision_ci_upper"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "recall_ci_lower": np.mean([m["recall_ci_lower"] for m in all_metrics]),
            "recall_ci_upper": np.mean([m["recall_ci_upper"] for m in all_metrics]),
            "specificity": np.mean([m["specificity"] for m in all_metrics]),
            "specificity_ci_lower": np.mean([m["specificity_ci_lower"] for m in all_metrics]),
            "specificity_ci_upper": np.mean([m["specificity_ci_upper"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "mcc": np.mean([m["mcc"] for m in all_metrics]),
            "auc": np.mean([m["auc"] for m in all_metrics]),
            "eer": np.mean([m["eer"] for m in all_metrics]),
            "bonafide_mean": np.mean([m["bonafide_mean"] for m in all_metrics]),
            "bonafide_std": np.mean([m["bonafide_std"] for m in all_metrics]),
            "morph_mean": np.mean([m["morph_mean"] for m in all_metrics]),
            "morph_std": np.mean([m["morph_std"] for m in all_metrics])
        }

        # Add APCER metrics to mean_metrics
        for key in all_metrics[0].keys():
            if key not in mean_metrics and key != "dataset" and key != "bonafide_count" and key != "morph_count":
                mean_metrics[key] = np.mean([m[key] for m in all_metrics])
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

    # Save results to CSV and JSON
    if output_dir and model_name:
        # Create DataFrame from metrics
        df = pd.DataFrame(all_metrics)

        # Add mean row
        mean_row = {"dataset": "mean"}
        mean_row.update(mean_metrics)
        mean_row["total_time"] = eval_time
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        # Generate a standardized filename for CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_dataset and model_type and epoch_number:
            base_filename = f"{model_dataset}_{model_type}_epoch_{epoch_number}_{timestamp}"
        else:
            base_filename = f"{model_name}_{timestamp}"

        # Save metrics CSV (only one version)
        metrics_csv_path = os.path.join(output_dir, "metrics", f"{base_filename}_metrics.csv")
        df.to_csv(metrics_csv_path, index=False)

        # Save detailed CSV with all raw scores
        detailed_data = []
        for dataset_name, (targets, outputs) in targets_outputs_dict.items():
            for i in range(len(targets)):
                # Basic information
                entry = {
                    "dataset": dataset_name,
                    "sample_id": i,
                    "target": targets[i],
                    "output": outputs[i],
                    "prediction": 1 if outputs[i] > classification_threshold else 0,
                    "correct": 1 if (outputs[i] > classification_threshold) == targets[i] else 0
                }

                # Add additional details
                entry["model_dataset"] = model_dataset
                entry["model_type"] = model_type
                entry["epoch_number"] = epoch_number
                entry["timestamp"] = timestamp
                entry["error_magnitude"] = abs(outputs[i] - targets[i])
                entry["bonafide_score"] = 1 - outputs[i]  # 1 - morph score
                entry["classification_threshold"] = classification_threshold

                # Add model parameters if available
                if model_params:
                    for key, value in model_params.items():
                        entry[f"model_{key}"] = value

                # Add training parameters if available
                if training_params:
                    for key, value in training_params.items():
                        entry[f"training_{key}"] = value

                detailed_data.append(entry)

        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(output_dir, "metrics", f"{base_filename}_detailed.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)

        # Create combined files only when evaluating multiple datasets
        if len(results) > 1:
            combined_metrics_csv_path = os.path.join(output_dir, "metrics", f"combined_metrics_{timestamp}.csv")
            combined_detailed_csv_path = os.path.join(output_dir, "metrics", f"combined_detailed_{timestamp}.csv")
            df.to_csv(combined_metrics_csv_path, index=False)
            detailed_df.to_csv(combined_detailed_csv_path, index=False)

        if verbose:
            print(f"Metrics saved to {metrics_csv_path}")
            print(f"Detailed scores saved to {detailed_csv_path}")

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
