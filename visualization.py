"""
Visualization module for SelfMAD project.

This module provides functions for visualizing model performance and evaluation results.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from itertools import cycle

def generate_plot_filename(plot_type, model_dataset=None, model_type=None, epoch_number=None,
                          eval_dataset=None, model_name=None, dataset_name=None):
    """
    Generate a consistent filename for plots.

    Args:
        plot_type: Type of plot (e.g., 'roc_curve', 'pr_curve')
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
        eval_dataset: Dataset being evaluated
        model_name: Name of the model (fallback if model_dataset, model_type, or epoch_number is None)
        dataset_name: Name of the dataset (fallback if eval_dataset is None)

    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_dataset and model_type and epoch_number and eval_dataset:
        return f"{plot_type}_{model_dataset}_{model_type}_epoch_{epoch_number}_{eval_dataset}_{timestamp}.png"
    elif model_name and dataset_name:
        return f"{plot_type}_{model_name}_{dataset_name}_{timestamp}.png"
    elif model_name:
        return f"{plot_type}_{model_name}_{timestamp}.png"
    else:
        return f"{plot_type}_{timestamp}.png"

def plot_training_history(train_metrics, val_metrics=None, output_dir=None, model_name=None,
                     model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot training history metrics.

    Args:
        train_metrics: DataFrame or dict or list of training metrics
        val_metrics: DataFrame or dict or list of validation metrics (optional)
        output_dir: Directory to save plots
        model_name: Name of the model (used for saving plots)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
    """
    # Convert to DataFrame if dict or list
    if isinstance(train_metrics, dict):
        train_metrics = pd.DataFrame(train_metrics)
    elif isinstance(train_metrics, list):
        train_metrics = pd.DataFrame(train_metrics)

    if val_metrics is not None:
        if isinstance(val_metrics, dict):
            val_metrics = pd.DataFrame(val_metrics)
        elif isinstance(val_metrics, list):
            val_metrics = pd.DataFrame(val_metrics)

    # Create output directory if it doesn't exist
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Plot metrics
    metrics_to_plot = ['loss', 'accuracy', 'auc', 'f1', 'eer']
    for metric in metrics_to_plot:
        if metric in train_metrics.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(train_metrics['epoch'], train_metrics[metric], label=f'Train {metric}')

            if val_metrics is not None and metric in val_metrics.columns:
                plt.plot(val_metrics['epoch'], val_metrics[metric], label=f'Val {metric}')

            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} vs. Epoch')
            plt.legend()
            plt.grid(True)

            # Save plot with consistent naming scheme
            if output_dir:
                filename = generate_plot_filename(
                    plot_type=f"{metric}_history",
                    model_dataset=model_dataset,
                    model_type=model_type,
                    epoch_number=epoch_number,
                    model_name=model_name
                )
                plt.savefig(os.path.join(plots_dir, filename))

            plt.close()

def plot_roc_curve(targets, outputs, output_dir=None, model_name=None, dataset_name=None,
                  model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot ROC curve.

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
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"ROC Curve - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset} (AUC = {roc_auc:.3f})"
    elif dataset_name:
        title = f"ROC Curve - {dataset_name} (AUC = {roc_auc:.3f})"
    else:
        title = f"Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.3f})"

    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="roc_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

def plot_pr_curve(targets, outputs, output_dir=None, model_name=None, dataset_name=None,
                 model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot Precision-Recall curve.

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

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(targets, outputs)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Precision-Recall Curve - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset} (AUC = {pr_auc:.3f})"
    elif dataset_name:
        title = f"Precision-Recall Curve - {dataset_name} (AUC = {pr_auc:.3f})"
    else:
        title = f"Precision-Recall Curve (AUC = {pr_auc:.3f})"

    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="pr_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

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

    # Calculate ROC curve (used for DET curve)
    fpr, tpr, _ = roc_curve(targets, outputs)
    fnr = 1 - tpr

    # Plot DET curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, fnr, color='darkorange', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-3, 1.0])
    plt.ylim([1e-3, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Detection Error Tradeoff (DET) Curve - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Detection Error Tradeoff (DET) Curve - {dataset_name}"
    else:
        title = f"Detection Error Tradeoff (DET) Curve"

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

def plot_confusion_matrix(targets, outputs, threshold=0.5, output_dir=None, model_name=None, dataset_name=None,
                         model_dataset=None, model_type=None, epoch_number=None, eval_dataset=None):
    """
    Plot confusion matrix.

    Args:
        targets: Ground truth labels
        outputs: Model prediction scores
        threshold: Threshold for binary classification
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

    # Convert outputs to binary predictions
    preds = (np.array(outputs) > threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(targets, preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Confusion Matrix - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Confusion Matrix - {dataset_name}"
    else:
        title = f"Confusion Matrix"

    plt.title(title)

    # Add class labels
    plt.xticks([0.5, 1.5], ['Bonafide', 'Morph'])
    plt.yticks([0.5, 1.5], ['Bonafide', 'Morph'])

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="confusion_matrix",
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

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.hist(bonafide_scores, bins=30, alpha=0.5, label='Bonafide', color='green')
    plt.hist(morph_scores, bins=30, alpha=0.5, label='Morph', color='red')

    # Add vertical lines for means
    plt.axvline(np.mean(bonafide_scores), color='green', linestyle='dashed', linewidth=2,
                label=f'Bonafide Mean: {np.mean(bonafide_scores):.4f}')
    plt.axvline(np.mean(morph_scores), color='red', linestyle='dashed', linewidth=2,
                label=f'Morph Mean: {np.mean(morph_scores):.4f}')

    # Add labels and title
    plt.xlabel('Score')
    plt.ylabel('Count')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Score Distribution - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Score Distribution - {dataset_name}"
    else:
        title = f"Score Distribution"

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="score_distribution",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            eval_dataset=eval_dataset or dataset_name,
            model_name=model_name,
            dataset_name=dataset_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Create KDE plot
    plt.figure(figsize=(12, 6))

    # Plot KDE
    sns.kdeplot(bonafide_scores, fill=True, label='Bonafide', color='green')
    sns.kdeplot(morph_scores, fill=True, label='Morph', color='red')

    # Add vertical lines for means
    plt.axvline(np.mean(bonafide_scores), color='green', linestyle='dashed', linewidth=2,
                label=f'Bonafide Mean: {np.mean(bonafide_scores):.4f}')
    plt.axvline(np.mean(morph_scores), color='red', linestyle='dashed', linewidth=2,
                label=f'Morph Mean: {np.mean(morph_scores):.4f}')

    # Add labels and title
    plt.xlabel('Score')
    plt.ylabel('Density')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Score Density - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Score Density - {dataset_name}"
    else:
        title = f"Score Density"

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="score_density",
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
        bonafide_data.append(scores["bonafide"])
        morph_data.append(scores["morph"])
        dataset_names.append(dataset_name)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Create box plots
    positions = np.arange(len(dataset_names)) * 3
    width = 0.8

    bp1 = plt.boxplot(bonafide_data, positions=positions, widths=width, patch_artist=True)
    bp2 = plt.boxplot(morph_data, positions=positions+width, widths=width, patch_artist=True)

    # Set colors
    for box in bp1['boxes']:
        box.set(facecolor='green', alpha=0.6)
    for box in bp2['boxes']:
        box.set(facecolor='red', alpha=0.6)

    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Score Distributions Across Datasets')
    plt.xticks(positions + width/2, dataset_names, rotation=45, ha='right')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Bonafide', 'Morph'])
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # Generate plot title
    if model_dataset and model_type:
        title = f"Score Distributions Across Datasets - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
        plt.title(title)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="score_boxplots",
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
    thresholds = np.linspace(0, 1, 100)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for threshold in thresholds:
        preds = (np.array(outputs) > threshold).astype(int)

        # Calculate metrics
        tp = np.sum((preds == 1) & (np.array(targets) == 1))
        tn = np.sum((preds == 0) & (np.array(targets) == 0))
        fp = np.sum((preds == 1) & (np.array(targets) == 0))
        fn = np.sum((preds == 0) & (np.array(targets) == 1))

        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        accuracy.append(acc)

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(prec)

        # Recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall.append(rec)

        # F1 Score
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1.append(f1_score)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot metrics
    plt.plot(thresholds, accuracy, label='Accuracy', linewidth=2)
    plt.plot(thresholds, precision, label='Precision', linewidth=2)
    plt.plot(thresholds, recall, label='Recall', linewidth=2)
    plt.plot(thresholds, f1, label='F1 Score', linewidth=2)

    # Find optimal threshold for F1
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1[optimal_idx]

    # Add vertical line for optimal threshold
    plt.axvline(optimal_threshold, color='black', linestyle='dashed', linewidth=1.5,
                label=f'Optimal Threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.4f})')

    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Metrics vs. Threshold - {model_type} trained on {model_dataset} (Epoch {epoch_number})\nEvaluated on {eval_dataset}"
    elif dataset_name:
        title = f"Metrics vs. Threshold - {dataset_name}"
    else:
        title = f"Metrics vs. Threshold"

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

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

def plot_combined_roc_curve(targets_outputs_dict, output_dir=None, model_name=None,
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

    # Create figure
    plt.figure(figsize=(12, 10))

    # Set up colors and line styles
    colors = list(mcolors.TABLEAU_COLORS)
    line_styles = ['-', '--', '-.', ':']
    style_cycle = cycle([(color, style) for color in colors for style in line_styles])

    # Plot ROC curve for each dataset
    for dataset_name, (targets, outputs) in targets_outputs_dict.items():
        color, style = next(style_cycle)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(targets, outputs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, linestyle=style, lw=2,
                 label=f'{dataset_name} (AUC = {roc_auc:.3f})')

    # Add reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Generate plot title
    if model_dataset and model_type:
        title = f"ROC Curves - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
    else:
        title = f"Receiver Operating Characteristic (ROC) Curves"

    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_roc_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()


def plot_combined_pr_curve(targets_outputs_dict, output_dir=None, model_name=None,
                         model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot Precision-Recall curves for all datasets on a single plot.

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

    # Create figure
    plt.figure(figsize=(12, 10))

    # Set up colors and line styles
    colors = list(mcolors.TABLEAU_COLORS)
    line_styles = ['-', '--', '-.', ':']
    style_cycle = cycle([(color, style) for color in colors for style in line_styles])

    # Plot PR curve for each dataset
    for dataset_name, (targets, outputs) in targets_outputs_dict.items():
        color, style = next(style_cycle)

        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(targets, outputs)
        pr_auc = auc(recall, precision)

        # Plot PR curve
        plt.plot(recall, precision, color=color, linestyle=style, lw=2,
                 label=f'{dataset_name} (AUC = {pr_auc:.3f})')

    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Generate plot title
    if model_dataset and model_type:
        title = f"Precision-Recall Curves - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
    else:
        title = f"Precision-Recall Curves"

    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_pr_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()


def plot_combined_det_curve(targets_outputs_dict, output_dir=None, model_name=None,
                          model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot Detection Error Tradeoff (DET) curves for all datasets on a single plot.

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

    # Create figure
    plt.figure(figsize=(12, 10))

    # Set up colors and line styles
    colors = list(mcolors.TABLEAU_COLORS)
    line_styles = ['-', '--', '-.', ':']
    style_cycle = cycle([(color, style) for color in colors for style in line_styles])

    # Plot DET curve for each dataset
    for dataset_name, (targets, outputs) in targets_outputs_dict.items():
        color, style = next(style_cycle)

        # Calculate ROC curve (used for DET curve)
        fpr, tpr, _ = roc_curve(targets, outputs)
        fnr = 1 - tpr

        # Plot DET curve
        plt.plot(fpr, fnr, color=color, linestyle=style, lw=2, label=dataset_name)

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-3, 1.0])
    plt.ylim([1e-3, 1.0])

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')

    # Generate plot title
    if model_dataset and model_type:
        title = f"Detection Error Tradeoff (DET) Curves - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
    else:
        title = f"Detection Error Tradeoff (DET) Curves"

    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, which="both", ls="-")

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_det_curve",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()


def plot_combined_score_distributions(class_separated_dict, output_dir=None, model_name=None,
                                model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot score distributions for all datasets on a single plot.

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

    # Create figure for histograms
    plt.figure(figsize=(15, 10))

    # Create subplots
    plt.subplot(2, 1, 1)
    plt.title('Bonafide Score Distributions')

    # Set up colors
    colors = list(mcolors.TABLEAU_COLORS)
    color_cycle = cycle(colors)

    # Plot bonafide distributions for each dataset
    for dataset_name, scores in class_separated_dict.items():
        color = next(color_cycle)
        if scores["bonafide"]:
            sns.kdeplot(scores["bonafide"], label=dataset_name, color=color)

    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot morph distributions
    plt.subplot(2, 1, 2)
    plt.title('Morph Score Distributions')

    # Reset color cycle
    color_cycle = cycle(colors)

    # Plot morph distributions for each dataset
    for dataset_name, scores in class_separated_dict.items():
        color = next(color_cycle)
        if scores["morph"]:
            sns.kdeplot(scores["morph"], label=dataset_name, color=color)

    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate plot title for bonafide subplot
    if model_dataset and model_type:
        title = f"Bonafide Score Distributions - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
        plt.subplot(2, 1, 1)
        plt.title(title)

        # Update morph subplot title
        plt.subplot(2, 1, 2)
        title = f"Morph Score Distributions - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
        plt.title(title)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_score_distributions",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()


def plot_dataset_comparison(results, output_dir=None, model_name=None,
                        model_dataset=None, model_type=None, epoch_number=None):
    """
    Plot comparison of metrics across datasets.

    Args:
        results: Dictionary of evaluation results
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

    # Extract dataset names and metrics
    dataset_names = [name for name in results.keys() if name != "mean"]

    # Metrics to plot
    metrics = ["accuracy", "auc", "eer", "apcer", "bpcer", "acer"]
    metric_titles = ["Accuracy", "AUC", "EER", "APCER", "BPCER", "ACER"]

    # Create figure
    plt.figure(figsize=(15, 10))

    # Create subplots
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        plt.subplot(2, 3, i+1)

        # Extract metric values
        values = [results[dataset][metric] for dataset in dataset_names]

        # Create bar chart
        bars = plt.bar(dataset_names, values)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', rotation=90)

        # Add labels and title
        plt.xlabel('Dataset')
        plt.ylabel(title)
        plt.title(f'{title} Across Datasets')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Generate plot title
    if model_dataset and model_type:
        plt.suptitle(f"Metrics Comparison - {model_type} trained on {model_dataset} (Epoch {epoch_number})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="dataset_comparison",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Create bonafide vs morph mean score comparison
    plt.figure(figsize=(12, 6))

    # Extract mean scores
    bonafide_means = [results[dataset]["bonafide_stats"]["mean"] for dataset in dataset_names]
    morph_means = [results[dataset]["morph_stats"]["mean"] for dataset in dataset_names]

    # Set up bar positions
    x = np.arange(len(dataset_names))
    width = 0.35

    # Create grouped bar chart
    plt.bar(x - width/2, bonafide_means, width, label='Bonafide Mean Score', color='green', alpha=0.7)
    plt.bar(x + width/2, morph_means, width, label='Morph Mean Score', color='red', alpha=0.7)

    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Mean Score')
    plt.title('Bonafide vs Morph Mean Scores Across Datasets')
    plt.xticks(x, dataset_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # Generate plot title
    if model_dataset and model_type:
        title = f"Bonafide vs Morph Mean Scores - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
        plt.title(title)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="mean_score_comparison",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Create BPCER at different APCER thresholds comparison
    # Check if BPCER at different APCER thresholds are available
    bpcer_metrics = []
    for dataset in dataset_names:
        for key in results[dataset].keys():
            if key.startswith("bpcer_") and key not in bpcer_metrics:
                bpcer_metrics.append(key)

    if bpcer_metrics:
        plt.figure(figsize=(14, 8))

        # Sort BPCER metrics by threshold value
        bpcer_metrics.sort(key=lambda x: int(x.split("_")[1]))

        # Create labels for the legend
        bpcer_labels = [f'BPCER@APCER={key.split("_")[1]}%' for key in bpcer_metrics]

        # Set up bar positions
        x = np.arange(len(dataset_names))
        width = 0.8 / len(bpcer_metrics)

        # Create grouped bar chart
        for i, (metric, label) in enumerate(zip(bpcer_metrics, bpcer_labels)):
            values = [results[dataset][metric] for dataset in dataset_names]
            offset = i * width - (len(bpcer_metrics) - 1) * width / 2
            plt.bar(x + offset, values, width, label=label)

        # Add labels and title
        plt.xlabel('Dataset')
        plt.ylabel('BPCER Value')
        plt.title('BPCER at Different APCER Thresholds Across Datasets')
        plt.xticks(x, dataset_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Generate plot title
        if model_dataset and model_type:
            title = f"BPCER at Different APCER Thresholds - {model_type} trained on {model_dataset} (Epoch {epoch_number})"
            plt.title(title)

        # Save plot with consistent naming scheme
        if output_dir:
            filename = generate_plot_filename(
                plot_type="bpcer_comparison",
                model_dataset=model_dataset,
                model_type=model_type,
                epoch_number=epoch_number,
                model_name=model_name
            )
            plt.savefig(os.path.join(plots_dir, filename))

        plt.close()

def visualize_evaluation_results(results, targets_outputs_dict, class_separated_dict=None, output_dir=None, model_name=None,
                           model_dataset=None, model_type=None, epoch_number=None):
    """
    Visualize evaluation results.

    Args:
        results: Dictionary of evaluation results
        targets_outputs_dict: Dictionary mapping dataset names to (targets, outputs) tuples
        class_separated_dict: Dictionary mapping dataset names to dictionaries of bonafide and morph scores
        output_dir: Directory to save visualizations
        model_name: Name of the model (used for saving visualizations)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Generate visualizations for each dataset
    for dataset_name, (targets, outputs) in targets_outputs_dict.items():
        # Plot ROC curve
        plot_roc_curve(targets, outputs, output_dir, model_name, dataset_name,
                      model_dataset, model_type, epoch_number, dataset_name)

        # Plot Precision-Recall curve
        plot_pr_curve(targets, outputs, output_dir, model_name, dataset_name,
                     model_dataset, model_type, epoch_number, dataset_name)

        # Plot DET curve
        plot_det_curve(targets, outputs, output_dir, model_name, dataset_name,
                      model_dataset, model_type, epoch_number, dataset_name)

        # Plot confusion matrix
        plot_confusion_matrix(targets, outputs, 0.5, output_dir, model_name, dataset_name,
                             model_dataset, model_type, epoch_number, dataset_name)

        # Plot threshold analysis
        plot_threshold_analysis(targets, outputs, output_dir, model_name, dataset_name,
                               model_dataset, model_type, epoch_number, dataset_name)

        # Plot score distributions if class_separated_dict is provided
        if class_separated_dict and dataset_name in class_separated_dict:
            bonafide_scores = class_separated_dict[dataset_name]["bonafide"]
            morph_scores = class_separated_dict[dataset_name]["morph"]

            if bonafide_scores and morph_scores:
                plot_score_distributions(bonafide_scores, morph_scores, output_dir, model_name, dataset_name,
                                        model_dataset, model_type, epoch_number, dataset_name)

    # Generate cross-dataset visualizations
    if results and len(results) > 1:
        plot_dataset_comparison(results, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

    # Generate boxplot comparison if class_separated_dict is provided
    if class_separated_dict and len(class_separated_dict) > 0:
        plot_score_boxplots(class_separated_dict, output_dir, model_name,
                           model_dataset, model_type, epoch_number)

    # Generate combined plots if multiple datasets are present
    if len(targets_outputs_dict) > 1:
        # Combined ROC curve
        plot_combined_roc_curve(targets_outputs_dict, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

        # Combined PR curve
        plot_combined_pr_curve(targets_outputs_dict, output_dir, model_name,
                              model_dataset, model_type, epoch_number)

        # Combined DET curve
        plot_combined_det_curve(targets_outputs_dict, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

        # Combined score distributions
        if class_separated_dict and len(class_separated_dict) > 1:
            plot_combined_score_distributions(class_separated_dict, output_dir, model_name,
                                            model_dataset, model_type, epoch_number)

# Function to be called from train__.py and eval__.py
def visualize(results=None, targets_outputs_dict=None, class_separated_dict=None, train_metrics=None, val_metrics=None,
             output_dir=None, model_name=None, model_dataset=None, model_type=None, epoch_number=None):
    """
    Main visualization function to be called from train__.py and eval__.py.

    Args:
        results: Dictionary of evaluation results (optional)
        targets_outputs_dict: Dictionary mapping dataset names to (targets, outputs) tuples (optional)
        class_separated_dict: Dictionary mapping dataset names to dictionaries of bonafide and morph scores (optional)
        train_metrics: DataFrame or dict of training metrics (optional)
        val_metrics: DataFrame or dict of validation metrics (optional)
        output_dir: Directory to save visualizations
        model_name: Name of the model (used for saving visualizations)
        model_dataset: Dataset the model was trained on
        model_type: Model architecture type
        epoch_number: Epoch number or checkpoint
    """
    # Plot training history if available
    if train_metrics is not None:
        plot_training_history(train_metrics, val_metrics, output_dir, model_name,
                             model_dataset, model_type, epoch_number)

    # Visualize evaluation results if available
    if results is not None and targets_outputs_dict is not None:
        visualize_evaluation_results(results, targets_outputs_dict, class_separated_dict,
                                    output_dir, model_name, model_dataset, model_type, epoch_number)
