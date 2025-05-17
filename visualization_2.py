"""
Advanced visualization functions for SelfMAD project.

This module provides advanced plotting functions for visualizing model performance.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from itertools import cycle
from visualization_1 import generate_plot_filename

def plot_det_curve(targets, outputs, output_dir=None, model_name=None, dataset_name=None,
                  model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot Detection Error Tradeoff (DET) curve.

    Args:
        targets: Ground truth labels
        outputs: Model prediction scores
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        dataset_name: Name of the dataset (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        eval_dataset: Dataset being evaluated
    """
    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(targets, outputs)
    
    # Calculate DET curve (FRR vs. FAR)
    frr = 1 - tpr  # False Rejection Rate = 1 - True Positive Rate
    far = fpr      # False Acceptance Rate = False Positive Rate

    # Find the Equal Error Rate (EER)
    eer_idx = np.argmin(np.abs(frr - far))
    eer = (frr[eer_idx] + far[eer_idx]) / 2

    # Plot DET curve
    plt.figure(figsize=(10, 8))
    plt.plot(far, frr, lw=2, label=f'DET curve (EER = {eer:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Detection Error Tradeoff (DET) Curve - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset} (EER = {eer:.3f})"
    elif dataset_name:
        title = f"Detection Error Tradeoff (DET) Curve - {dataset_name} (EER = {eer:.3f})"
    else:
        title = f"Detection Error Tradeoff (DET) Curve (EER = {eer:.3f})"

    plt.title(title)
    plt.grid(True, which="both", ls="-")

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="det_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

def plot_score_distributions(bonafide_scores, morph_scores, output_dir=None, model_name=None, dataset_name=None,
                            model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot score distributions for bonafide and morph samples.

    Args:
        bonafide_scores: List of scores for bonafide samples
        morph_scores: List of scores for morph samples
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        dataset_name: Name of the dataset (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        eval_dataset: Dataset being evaluated
    """
    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Plot score distributions
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    bins = np.linspace(0, 1, 50)
    plt.hist(bonafide_scores, bins=bins, alpha=0.5, label=f'Bonafide (n={len(bonafide_scores)})', color='green')
    plt.hist(morph_scores, bins=bins, alpha=0.5, label=f'Morph (n={len(morph_scores)})', color='red')
    
    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold = 0.5')
    
    plt.xlabel('Score')
    plt.ylabel('Count')
    
    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Score Distributions - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Score Distributions - {dataset_name}"
    else:
        title = "Score Distributions"

    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="score_distributions",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

def plot_score_boxplots(class_separated_dict, output_dir=None, model_name=None,
                       model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot box plots comparing score distributions across datasets.

    Args:
        class_separated_dict: Dictionary mapping dataset names to dictionaries of bonafide and morph scores
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
    """
    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Prepare data for box plots
    bonafide_data = []
    morph_data = []
    dataset_names = []

    for dataset_name, scores in class_separated_dict.items():
        if "bonafide" in scores and "morph" in scores:
            bonafide_data.append(scores["bonafide"])
            morph_data.append(scores["morph"])
            dataset_names.append(dataset_name)

    if not dataset_names:
        return  # No data to plot

    # Plot bonafide scores
    plt.figure(figsize=(12, 8))
    plt.boxplot(bonafide_data, labels=dataset_names)
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    
    # Generate plot title
    if model_dataset and model_type:
        title = f"Bonafide Score Distributions - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "Bonafide Score Distributions"

    plt.title(title)
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="bonafide_boxplots",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Plot morph scores
    plt.figure(figsize=(12, 8))
    plt.boxplot(morph_data, labels=dataset_names)
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    
    # Generate plot title
    if model_dataset and model_type:
        title = f"Morph Score Distributions - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "Morph Score Distributions"

    plt.title(title)
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="morph_boxplots",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

def plot_threshold_analysis(targets, outputs, output_dir=None, model_name=None, dataset_name=None,
                          model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot metrics at different threshold values.

    Args:
        targets: Ground truth labels
        outputs: Model prediction scores
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        dataset_name: Name of the dataset (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        eval_dataset: Dataset being evaluated
    """
    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Calculate metrics at different thresholds
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        preds = (np.array(outputs) > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((preds == 1) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Plot metrics vs. threshold
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    
    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold = 0.5')
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    
    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Threshold Analysis - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Threshold Analysis - {dataset_name}"
    else:
        title = "Threshold Analysis"

    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="threshold_analysis",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

def plot_combined_roc_curves(targets_outputs_dict, output_dir=None, model_name=None,
                           model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot ROC curves for all datasets on a single plot.

    Args:
        targets_outputs_dict: Dictionary mapping dataset names to (targets, outputs) tuples
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
    """
    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    
    # Use different colors for each dataset
    colors = list(mcolors.TABLEAU_COLORS)
    color_cycle = cycle(colors)
    
    for dataset_name, (targets, outputs) in targets_outputs_dict.items():
        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(targets, outputs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, color=next(color_cycle), label=f'{dataset_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # Generate plot title
    if model_dataset and model_type:
        title = f"ROC Curves - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "ROC Curves Comparison"

    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_roc_curves",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()
