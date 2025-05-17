"""
Test script for combined DET curve visualization.

This script generates sample data and calls the visualization functions to test
the combined DET curve functionality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from visualization_2 import plot_combined_det_curves

def generate_sample_data():
    """
    Generate sample data for testing the combined DET curve visualization.
    
    Returns:
        Dictionary mapping dataset names to (targets, outputs) tuples
    """
    np.random.seed(42)  # For reproducibility
    
    targets_outputs_dict = {}
    
    # Generate sample data for LMA dataset
    lma_targets = np.random.randint(0, 2, size=1000)
    lma_outputs = np.random.random(size=1000)
    # Make outputs more realistic (higher scores for positive samples)
    lma_outputs = lma_outputs * 0.5 + lma_targets * 0.3 + np.random.random(size=1000) * 0.2
    lma_outputs = np.clip(lma_outputs, 0, 1)
    targets_outputs_dict["LMA"] = (lma_targets, lma_outputs)
    
    # Generate sample data for MIPGAN_I dataset
    mipgan_targets = np.random.randint(0, 2, size=1000)
    mipgan_outputs = np.random.random(size=1000)
    # Make outputs more realistic (higher scores for positive samples)
    mipgan_outputs = mipgan_outputs * 0.4 + mipgan_targets * 0.4 + np.random.random(size=1000) * 0.2
    mipgan_outputs = np.clip(mipgan_outputs, 0, 1)
    targets_outputs_dict["MIPGAN_I"] = (mipgan_targets, mipgan_outputs)
    
    # Generate sample data for LMA_MIPGAN_I dataset
    lma_mipgan_targets = np.random.randint(0, 2, size=1000)
    lma_mipgan_outputs = np.random.random(size=1000)
    # Make outputs more realistic (higher scores for positive samples)
    lma_mipgan_outputs = lma_mipgan_outputs * 0.3 + lma_mipgan_targets * 0.5 + np.random.random(size=1000) * 0.2
    lma_mipgan_outputs = np.clip(lma_mipgan_outputs, 0, 1)
    targets_outputs_dict["LMA_MIPGAN_I"] = (lma_mipgan_targets, lma_mipgan_outputs)
    
    return targets_outputs_dict

def main():
    """Main function to test the combined DET curve visualization."""
    # Create output directory
    output_dir = "./output/test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    targets_outputs_dict = generate_sample_data()
    
    # Generate combined DET curve
    plot_combined_det_curves(
        targets_outputs_dict=targets_outputs_dict,
        output_dir=output_dir,
        model_name="test_model",
        model_dataset="test_dataset",
        model_type="test_type",
        epoch_number=1
    )
    
    print("Combined DET curve generation completed.")
    print(f"Check the output directory: {output_dir}/plots")

if __name__ == "__main__":
    main()
