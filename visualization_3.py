"""
Main visualization interface for SelfMAD project.

This module provides the main interface for visualizing model performance and evaluation results.
It is designed to be used by both SelfMAD-main and SelfMAD-siam repositories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import functions from visualization_1 and visualization_2
from visualization_1 import (
    generate_plot_filename,
    plot_training_history,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix
)
from visualization_2 import (
    plot_det_curve,
    plot_score_distributions,
    plot_score_boxplots,
    plot_threshold_analysis,
    plot_combined_roc_curves,
    plot_combined_det_curves
)
from evaluation_1 import calculate_eer

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

    # Extract metrics for comparison
    datasets = []
    accuracies = []
    aucs = []
    eers = []

    for dataset_name, metrics in results.items():
        if dataset_name != "mean" and isinstance(metrics, dict):
            datasets.append(dataset_name)
            accuracies.append(metrics.get("accuracy", 0))
            aucs.append(metrics.get("auc", 0))
            eers.append(metrics.get("eer", 0))

    if not datasets:
        return  # No data to plot

    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    plt.bar(datasets, accuracies)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')

    # Generate plot title
    if model_dataset and model_type:
        title = f"Accuracy Comparison - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "Accuracy Comparison"

    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y')

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="accuracy_comparison",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Plot AUC comparison
    plt.figure(figsize=(12, 8))
    plt.bar(datasets, aucs)
    plt.xlabel('Dataset')
    plt.ylabel('AUC')

    # Generate plot title
    if model_dataset and model_type:
        title = f"AUC Comparison - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "AUC Comparison"

    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y')

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="auc_comparison",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Plot EER comparison
    plt.figure(figsize=(12, 8))
    plt.bar(datasets, eers)
    plt.xlabel('Dataset')
    plt.ylabel('EER')

    # Generate plot title
    if model_dataset and model_type:
        title = f"EER Comparison - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "EER Comparison"

    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y')

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="eer_comparison",
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

    # Plot bonafide score distributions
    plt.figure(figsize=(14, 10))

    for dataset_name, scores in class_separated_dict.items():
        if "bonafide" in scores and len(scores["bonafide"]) > 0:
            plt.hist(scores["bonafide"], bins=50, alpha=0.5, label=f'{dataset_name} (n={len(scores["bonafide"])})')

    plt.xlabel('Score')
    plt.ylabel('Count')

    # Generate plot title
    if model_dataset and model_type:
        title = f"Bonafide Score Distributions - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "Bonafide Score Distributions"

    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_bonafide_distributions",
            model_dataset=model_dataset,
            model_type=model_type,
            epoch_number=epoch_number,
            model_name=model_name
        )
        plt.savefig(os.path.join(plots_dir, filename))

    plt.close()

    # Plot morph score distributions
    plt.figure(figsize=(14, 10))

    for dataset_name, scores in class_separated_dict.items():
        if "morph" in scores and len(scores["morph"]) > 0:
            plt.hist(scores["morph"], bins=50, alpha=0.5, label=f'{dataset_name} (n={len(scores["morph"])})')

    plt.xlabel('Score')
    plt.ylabel('Count')

    # Generate plot title
    if model_dataset and model_type:
        title = f"Morph Score Distributions - {model_type} trained on {model_dataset}"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
    else:
        title = "Morph Score Distributions"

    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot with consistent naming scheme
    if output_dir:
        filename = generate_plot_filename(
            plot_type="combined_morph_distributions",
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

    # Generate combined visualizations
    if len(targets_outputs_dict) > 1:
        # Plot combined ROC curves
        plot_combined_roc_curves(targets_outputs_dict, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

        # Plot combined DET curves
        plot_combined_det_curves(targets_outputs_dict, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

        # Plot dataset comparison
        plot_dataset_comparison(results, output_dir, model_name,
                              model_dataset, model_type, epoch_number)

        # Plot combined score distributions if class_separated_dict is provided
        if class_separated_dict:
            plot_combined_score_distributions(class_separated_dict, output_dir, model_name,
                                            model_dataset, model_type, epoch_number)

            # Plot score boxplots
            plot_score_boxplots(class_separated_dict, output_dir, model_name,
                               model_dataset, model_type, epoch_number)

def calculate_and_save_metrics(results, targets_outputs_dict, output_dir=None, model_name=None, epoch_number=None):
    """
    Calculate and save metrics without generating plots.

    Args:
        results: Dictionary of evaluation results
        targets_outputs_dict: Dictionary mapping dataset names to (targets, outputs) tuples
        output_dir: Directory to save metrics
        model_name: Name of the model
        epoch_number: Epoch number or checkpoint

    Returns:
        Dictionary of calculated metrics
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

    # Extract metrics from results
    metrics_results = {}
    if results:
        # Save results to CSV
        results_df = pd.DataFrame()
        for dataset_name, metrics in results.items():
            if isinstance(metrics, dict):
                dataset_metrics = {f"{dataset_name}_{k}": v for k, v in metrics.items()}
                metrics_results.update(dataset_metrics)

                # Add to DataFrame
                dataset_df = pd.DataFrame([metrics], index=[dataset_name])
                results_df = pd.concat([results_df, dataset_df])

        # Save to CSV
        if not results_df.empty and output_dir:
            results_csv_path = os.path.join(metrics_dir, f"results_epoch_{epoch_number}.csv")
            results_df.to_csv(results_csv_path)

    # Calculate additional metrics from targets_outputs_dict if needed
    if targets_outputs_dict:
        for dataset_name, (targets, outputs) in targets_outputs_dict.items():
            # Calculate metrics that might not be in results
            preds = (np.array(outputs) > 0.5).astype(int)
            accuracy = accuracy_score(targets, preds)
            auc = roc_auc_score(targets, outputs)
            eer = calculate_eer(targets, outputs)

            # Add to metrics_results
            metrics_results[f"{dataset_name}_accuracy"] = accuracy
            metrics_results[f"{dataset_name}_auc"] = auc
            metrics_results[f"{dataset_name}_eer"] = eer

    # Save metrics to JSON
    if output_dir and metrics_results:
        metrics_json_path = os.path.join(metrics_dir, f"metrics_epoch_{epoch_number}.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_results, f, indent=4, cls=NumpyEncoder)

    return metrics_results

# Function to be called from train__.py and eval__.py
def visualize(results=None, targets_outputs_dict=None, class_separated_dict=None, train_metrics=None, val_metrics=None,
             output_dir=None, model_name=None, model_dataset=None, model_type=None, epoch_number=None, generate_plots=True):
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
        generate_plots: Whether to generate plots or just calculate metrics (default: True)
    """
    # Plot training history if available (always generate these plots)
    if train_metrics is not None:
        plot_training_history(train_metrics, val_metrics, output_dir, model_name,
                             model_dataset, model_type, epoch_number)

    # Calculate and save metrics for every epoch (this always happens)
    metrics_results = None
    if results is not None and targets_outputs_dict is not None:
        metrics_results = calculate_and_save_metrics(results, targets_outputs_dict,
                                                   output_dir, model_name, epoch_number)

        # Generate plots only if requested
        if generate_plots:
            visualize_evaluation_results(results, targets_outputs_dict, class_separated_dict,
                                        output_dir, model_name, model_dataset, model_type, epoch_number)

    # Return the metrics results
    return metrics_results
