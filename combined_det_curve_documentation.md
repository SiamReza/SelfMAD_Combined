# Combined DET Curve Visualization

This document explains the implementation of combined Detection Error Tradeoff (DET) curve visualization for all datasets in the SelfMAD project.

## Overview

The Detection Error Tradeoff (DET) curve is a graphical representation of the trade-off between false acceptance rate (FAR) and false rejection rate (FRR) in a binary classification system. It is commonly used in biometric authentication, fraud detection, and similar domains to visualize the performance of a classifier at different threshold settings.

The combined DET curve functionality allows for visualizing the performance of a model across multiple datasets on a single plot, making it easier to compare performance across different datasets.

## Implementation Details

The combined DET curve functionality has been implemented in the existing visualization framework:

1. A new function `plot_combined_det_curves` has been added to `visualization_2.py`
2. The `visualize_evaluation_results` function in `visualization_3.py` has been updated to call this new function when multiple datasets are present

### Function Signature

```python
def plot_combined_det_curves(targets_outputs_dict, output_dir=None, model_name=None,
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
```

## Features

The combined DET curve visualization includes the following features:

1. **Multiple Datasets**: Plots DET curves for all datasets in the `targets_outputs_dict` on a single plot
2. **Logarithmic Scale**: Uses logarithmic scales for both axes to better visualize low error rates
3. **Equal Error Rate (EER)**: Calculates and displays the EER for each dataset in the legend
4. **Distinct Styling**: Uses different colors and line styles for each dataset to make them easily distinguishable
5. **Consistent Naming**: Uses the same naming convention as other plots in the project
6. **Automatic Generation**: Automatically generated when evaluating multiple datasets

## How It Works

1. The function takes a dictionary mapping dataset names to tuples of (targets, outputs)
2. For each dataset, it calculates the ROC curve and converts it to a DET curve (FRR vs. FAR)
3. It calculates the Equal Error Rate (EER) for each dataset
4. It plots all DET curves on a single plot with different colors and line styles
5. It saves the plot to the specified output directory

## Interpreting the Results

The DET curve plots the False Positive Rate (FAR) against the False Negative Rate (FRR) on logarithmic scales. Each dataset is represented by a different color and line style. The Equal Error Rate (EER) for each dataset is shown in the legend.

A good classifier will have curves that are closer to the bottom-left corner of the plot, indicating lower error rates. The EER is the point where FAR equals FRR, and a lower EER indicates better performance.

## Usage

The combined DET curve visualization is automatically generated when evaluating multiple datasets. No additional code is required to use this functionality.

When the `visualize_evaluation_results` function is called with multiple datasets in the `targets_outputs_dict`, it will automatically generate the combined DET curve along with other combined visualizations.

## Example Output

The combined DET curve will be saved in the `plots` subdirectory of the specified output directory with a filename following the pattern:

```
combined_det_curves_{model_dataset}_{model_type}_epoch_{epoch_number}_{timestamp}.png
```

For example:
```
combined_det_curves_LMA_vit_mae_large_epoch_100_20230615_123456.png
```

## Benefits

The combined DET curve visualization provides several benefits:

1. **Easy Comparison**: Allows for easy comparison of model performance across different datasets
2. **Comprehensive View**: Provides a comprehensive view of model performance at different operating points
3. **Logarithmic Scale**: Makes it easier to visualize and compare performance at low error rates
4. **Equal Error Rate**: Provides a single metric (EER) for each dataset that can be used for comparison
5. **Consistent Styling**: Uses the same styling as other plots in the project for a consistent user experience
