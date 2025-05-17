"""
Basic visualization functions for SelfMAD project.

This module provides basic plotting functions for visualizing model performance.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from datetime import datetime

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
        str: Filename for the plot
    """
    # Start with the plot type
    filename_parts = [plot_type]

    # Add model information if available
    if model_dataset and model_type:
        filename_parts.append(f"{model_type}_on_{model_dataset}")
        if epoch_number is not None:
            filename_parts.append(f"epoch_{epoch_number}")
    elif model_name:
        filename_parts.append(model_name)

    # Add dataset information if available
    if eval_dataset:
        filename_parts.append(f"eval_on_{eval_dataset}")
    elif dataset_name:
        filename_parts.append(dataset_name)

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts.append(timestamp)

    # Join parts with underscores and add extension
    return "_".join(filename_parts) + ".png"

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

            # Generate plot title
            if model_dataset and model_type:
                title = f"{metric.capitalize()} - {model_type} trained on {model_dataset}"
                if epoch_number is not None:
                    title += f" (Epoch {epoch_number})"
            else:
                title = f"{metric.capitalize()} vs. Epoch"

            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
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

    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"ROC Curve - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset} (AUC = {roc_auc:.3f})"
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
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Precision-Recall Curve - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset} (AUC = {pr_auc:.3f})"
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

    # Calculate confusion matrix
    preds = (np.array(outputs) > threshold).astype(int)
    cm = confusion_matrix(targets, preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    # Generate plot title
    if model_dataset and model_type and eval_dataset:
        title = f"Confusion Matrix - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nEvaluated on {eval_dataset} (Threshold = {threshold:.2f})"
    elif dataset_name:
        title = f"Confusion Matrix - {dataset_name} (Threshold = {threshold:.2f})"
    else:
        title = f"Confusion Matrix (Threshold = {threshold:.2f})"

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
