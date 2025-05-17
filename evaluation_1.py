"""
Basic metric calculation functions for SelfMAD project.

This module provides functions for calculating various metrics for model evaluation.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def calculate_accuracy(targets, outputs, threshold=0.5):
    """Calculate accuracy from model outputs and targets."""
    preds = (np.array(outputs) > threshold).astype(int)
    return accuracy_score(targets, preds)

def calculate_precision(targets, outputs, threshold=0.5):
    """Calculate precision from model outputs and targets."""
    preds = (np.array(outputs) > threshold).astype(int)
    return precision_score(targets, preds)

def calculate_recall(targets, outputs, threshold=0.5):
    """Calculate recall from model outputs and targets."""
    preds = (np.array(outputs) > threshold).astype(int)
    return recall_score(targets, preds)

def calculate_f1(targets, outputs, threshold=0.5):
    """Calculate F1 score from model outputs and targets."""
    preds = (np.array(outputs) > threshold).astype(int)
    return f1_score(targets, preds)

def calculate_auc(targets, outputs):
    """Calculate AUC from model outputs and targets."""
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_auc. Returning default value 0.5.")
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

        # Find the threshold where FPR and FNR are closest
        eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2

        return eer
    except Exception as e:
        print(f"Error calculating EER: {str(e)}. Returning default value 0.5.")
        return 0.5

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
        Dictionary of metrics
    """
    try:
        # Convert to numpy arrays
        targets = np.array(targets)
        outputs = np.array(outputs)

        # Check if arrays are empty or have only one class
        if len(targets) == 0 or len(outputs) == 0:
            print("Warning: Empty arrays passed to calculate_apcer_bpcer_acer. Returning default values.")
            return {
                "apcer": 0,
                "bpcer": 0,
                "acer": 0,
                "bonafide_mean": 0,
                "morph_mean": 0,
                "bonafide_std": 0,
                "morph_std": 0,
                "bonafide_count": 0,
                "morph_count": 0
            }

        # Get indices for bonafide and morph samples
        bonafide_idx = (targets == 0)
        morph_idx = (targets == 1)

        # Get scores for bonafide and morph samples
        bonafide_scores = outputs[bonafide_idx]
        morph_scores = outputs[morph_idx]

        # Count bonafide and morph samples
        n_bonafide = len(bonafide_scores)
        n_morph = len(morph_scores)

        # Check if both classes are present
        if n_bonafide == 0 or n_morph == 0:
            print(f"Warning: Missing one class in calculate_apcer_bpcer_acer. Bonafide: {n_bonafide}, Morph: {n_morph}. Returning default values.")
            return {
                "apcer": 0,
                "bpcer": 0,
                "acer": 0,
                "bonafide_mean": np.mean(bonafide_scores) if n_bonafide > 0 else 0,
                "morph_mean": np.mean(morph_scores) if n_morph > 0 else 0,
                "bonafide_std": np.std(bonafide_scores) if n_bonafide > 0 else 0,
                "morph_std": np.std(morph_scores) if n_morph > 0 else 0,
                "bonafide_count": n_bonafide,
                "morph_count": n_morph
            }

        # Calculate mean scores
        bonafide_mean = np.mean(bonafide_scores)
        morph_mean = np.mean(morph_scores)

        # Calculate standard deviations
        bonafide_std = np.std(bonafide_scores)
        morph_std = np.std(morph_scores)

        # Calculate APCER and BPCER at the given threshold
        # APCER: proportion of morph samples incorrectly classified as bonafide
        # BPCER: proportion of bonafide samples incorrectly classified as morph
        apcer = np.mean(morph_scores <= threshold)
        bpcer = np.mean(bonafide_scores > threshold)

        # Calculate ACER: average of APCER and BPCER
        acer = (apcer + bpcer) / 2

        # Initialize results dictionary
        results = {
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": acer,
            "bonafide_mean": bonafide_mean,
            "morph_mean": morph_mean,
            "bonafide_std": bonafide_std,
            "morph_std": morph_std,
            "bonafide_count": n_bonafide,
            "morph_count": n_morph
        }
    except Exception as e:
        print(f"Error in calculate_apcer_bpcer_acer: {str(e)}. Returning default values.")
        return {
            "apcer": 0,
            "bpcer": 0,
            "acer": 0,
            "bonafide_mean": 0,
            "morph_mean": 0,
            "bonafide_std": 0,
            "morph_std": 0,
            "bonafide_count": 0,
            "morph_count": 0
        }

    try:
        # Calculate BPCER at different APCER thresholds if provided
        if apcer_thresholds and 'results' in locals() and n_bonafide > 0 and n_morph > 0:
            # Sort scores
            sorted_morph_scores = np.sort(morph_scores)

            for apcer_threshold in apcer_thresholds:
                try:
                    # Calculate the index corresponding to the APCER threshold
                    idx = int(np.ceil(apcer_threshold * n_morph)) - 1
                    idx = max(0, min(idx, n_morph - 1))  # Ensure index is within bounds

                    # Get the score threshold
                    score_threshold = sorted_morph_scores[idx]

                    # Calculate BPCER at this threshold
                    bpcer_at_apcer = np.mean(bonafide_scores > score_threshold)

                    # Add to results
                    results[f"bpcer_at_apcer_{apcer_threshold:.2f}"] = bpcer_at_apcer
                except Exception as e:
                    print(f"Error calculating BPCER at APCER {apcer_threshold}: {str(e)}. Using default value 0.")
                    results[f"bpcer_at_apcer_{apcer_threshold:.2f}"] = 0
        elif apcer_thresholds and 'results' in locals():
            # Add default values for BPCER at different APCER thresholds
            for apcer_threshold in apcer_thresholds:
                results[f"bpcer_at_apcer_{apcer_threshold:.2f}"] = 0
    except Exception as e:
        print(f"Error calculating BPCER at different APCER thresholds: {str(e)}. Using default values.")
        if 'results' in locals() and apcer_thresholds:
            for apcer_threshold in apcer_thresholds:
                results[f"bpcer_at_apcer_{apcer_threshold:.2f}"] = 0

    return results if 'results' in locals() else {
        "apcer": 0,
        "bpcer": 0,
        "acer": 0,
        "bonafide_mean": 0,
        "morph_mean": 0,
        "bonafide_std": 0,
        "morph_std": 0,
        "bonafide_count": 0,
        "morph_count": 0
    }

def calculate_metrics(targets, outputs, threshold=0.5, apcer_thresholds=None):
    """
    Calculate all metrics for model evaluation.

    Args:
        targets: Ground truth labels
        outputs: Model prediction scores
        threshold: Threshold for binary classification
        apcer_thresholds: List of APCER thresholds for BPCER calculation

    Returns:
        Dictionary of metrics
    """
    # Default metrics dictionary with all expected keys
    default_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "auc": 0.5,
        "eer": 0.5,
        "apcer": 0,
        "bpcer": 0,
        "acer": 0,
        "bonafide_mean": 0,
        "morph_mean": 0,
        "bonafide_std": 0,
        "morph_std": 0,
        "bonafide_count": 0,
        "morph_count": 0
    }

    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to calculate_metrics. Returning default values.")
        return default_metrics

    try:
        # Calculate basic metrics
        metrics = default_metrics.copy()  # Start with default values

        # Update with calculated values
        metrics["accuracy"] = calculate_accuracy(targets, outputs, threshold)
        metrics["precision"] = calculate_precision(targets, outputs, threshold)
        metrics["recall"] = calculate_recall(targets, outputs, threshold)
        metrics["f1"] = calculate_f1(targets, outputs, threshold)
        metrics["auc"] = calculate_auc(targets, outputs)
        metrics["eer"] = calculate_eer(targets, outputs)

        # Calculate APCER, BPCER, ACER, and BPCER at different APCER thresholds
        try:
            apcer_bpcer_metrics = calculate_apcer_bpcer_acer(targets, outputs, threshold, apcer_thresholds)
            # Add APCER, BPCER, ACER, and BPCER at different APCER thresholds
            metrics.update(apcer_bpcer_metrics)
        except Exception as e:
            print(f"Error calculating APCER metrics: {str(e)}. Using default values for these metrics.")
            # APCER metrics will remain as default values

        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}. Returning default values.")
        return default_metrics

def separate_by_class(targets, outputs):
    """
    Separate outputs by class for visualization.

    Args:
        targets: Ground truth labels (0 for bonafide, 1 for morph)
        outputs: Model prediction scores

    Returns:
        Dictionary with bonafide and morph scores
    """
    # Check if arrays are empty
    if len(targets) == 0 or len(outputs) == 0:
        print("Warning: Empty arrays passed to separate_by_class. Returning empty lists.")
        return {
            "bonafide": [],
            "morph": []
        }

    try:
        # Convert to numpy arrays
        targets = np.array(targets)
        outputs = np.array(outputs)

        # Get indices for bonafide and morph samples
        bonafide_idx = (targets == 0)
        morph_idx = (targets == 1)

        # Get scores for bonafide and morph samples
        bonafide_scores = outputs[bonafide_idx].tolist()
        morph_scores = outputs[morph_idx].tolist()

        return {
            "bonafide": bonafide_scores,
            "morph": morph_scores
        }
    except Exception as e:
        print(f"Error separating classes: {str(e)}. Returning empty lists.")
        return {
            "bonafide": [],
            "morph": []
        }
