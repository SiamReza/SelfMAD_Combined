#!/usr/bin/env python3
"""
Test script to verify dataset loading.
"""

import os
import sys
import argparse

# Add the SelfMAD-siam and SelfMAD-main directories to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "SelfMAD-siam"))
sys.path.append(os.path.join(script_dir, "SelfMAD-main"))

# Import the dataset utilities
try:
    from utils.dataset import TestMorphDataset, default_datasets
except ImportError:
    print("Error: Could not import dataset utilities from SelfMAD-siam or SelfMAD-main.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--dataset", default="LMA", help="Dataset to test")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Try to load configuration from automation_config.py
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from automation_config import get_config
        config = get_config()
        print("Loaded configuration from automation_config.py")
        
        # Print dataset paths
        print(f"Dataset paths from automation_config.py: {config['dataset_paths']}")
        
        # Test dataset loading using the paths from automation_config.py
        print(f"\nTesting dataset loading for {args.dataset} using paths from automation_config.py...")
        
        # Create config for custom morph datasets
        morph_config = {
            "LMA_path": config["dataset_paths"]["LMA_path"],
            "LMA_UBO_path": config["dataset_paths"]["LMA_UBO_path"],
            "MIPGAN_I_path": config["dataset_paths"]["MIPGAN_I_path"],
            "MIPGAN_II_path": config["dataset_paths"]["MIPGAN_II_path"],
            "MorDiff_path": config["dataset_paths"]["MorDiff_path"],
            "StyleGAN_path": config["dataset_paths"]["StyleGAN_path"]
        }
        
        # Load custom morph test datasets
        custom_test_datasets = default_datasets(args.image_size, datasets="custom_morph", config=morph_config)
        
        # Check if datasets were loaded successfully
        if custom_test_datasets:
            print(f"Successfully loaded custom test datasets: {list(custom_test_datasets.keys())}")
            
            # Print details about each dataset
            for dataset_name, dataset_dict in custom_test_datasets.items():
                for split_name, dataset in dataset_dict.items():
                    print(f"  - {dataset_name} ({split_name}): {len(dataset)} samples")
        else:
            print("No custom test datasets were loaded.")
        
    except (ImportError, AttributeError) as e:
        print(f"Error loading configuration from automation_config.py: {e}")
        
        # Try with default paths
        print("\nTrying with default paths...")
        
        # Try multiple possible locations for datasets
        possible_base_dirs = [
            os.environ.get("DATASETS_DIR"),  # First check environment variable
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"),  # Check relative to script
            os.path.join(".", "datasets"),  # Check in current directory
            os.path.join("..", "datasets"),  # Check one level up
            os.path.abspath("datasets")  # Check absolute path
        ]
        
        # Find the first valid directory
        datasets_dir = None
        for base_dir in possible_base_dirs:
            if base_dir and os.path.exists(base_dir):
                datasets_dir = base_dir
                break
        
        if datasets_dir:
            print(f"Using datasets directory: {datasets_dir}")
            
            # Create config for custom morph datasets
            morph_config = {
                "LMA_path": datasets_dir,
                "LMA_UBO_path": datasets_dir,
                "MIPGAN_I_path": datasets_dir,
                "MIPGAN_II_path": datasets_dir,
                "MorDiff_path": datasets_dir,
                "StyleGAN_path": datasets_dir
            }
            
            # Load custom morph test datasets
            custom_test_datasets = default_datasets(args.image_size, datasets="custom_morph", config=morph_config)
            
            # Check if datasets were loaded successfully
            if custom_test_datasets:
                print(f"Successfully loaded custom test datasets: {list(custom_test_datasets.keys())}")
                
                # Print details about each dataset
                for dataset_name, dataset_dict in custom_test_datasets.items():
                    for split_name, dataset in dataset_dict.items():
                        print(f"  - {dataset_name} ({split_name}): {len(dataset)} samples")
            else:
                print("No custom test datasets were loaded.")
        else:
            print("Could not find a valid datasets directory.")

if __name__ == "__main__":
    main()
