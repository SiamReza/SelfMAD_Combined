#!/usr/bin/env python3
"""
Direct execution script for SelfMAD experiments.
This script runs training and evaluation processes directly, without relying on marker files.
"""

import os
import sys
import subprocess
import glob
import argparse
import threading
import queue
import traceback

# Import the robust model handler functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from robust_model_handler import normalize_path, get_next_serial_number
except ImportError:
    print("Warning: Could not import robust_model_handler. Using fallback functions.")

    def normalize_path(path):
        """Normalize a path to use the correct path separators for the current OS."""
        if path is None:
            return None

        # Convert to absolute path if not already
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        # Normalize path separators
        return os.path.normpath(path)

    def get_next_serial_number(base_dir, prefix):
        """Get the next available serial number for a given prefix."""
        # Create the directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Find all directories that match the prefix
        matching_dirs = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item.startswith(prefix):
                    matching_dirs.append(item)

        # Extract serial numbers from directory names
        serial_numbers = []
        for dir_name in matching_dirs:
            try:
                # Extract the serial number from the end of the directory name
                serial_str = dir_name[len(prefix):]
                if serial_str.isdigit():
                    serial_num = int(serial_str)
                    serial_numbers.append(serial_num)
            except (ValueError, IndexError):
                # Skip directories that don't have a valid serial number
                continue

        # Determine the next serial number
        if not serial_numbers:
            return 1  # Start with 1 if no existing directories
        else:
            return max(serial_numbers) + 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SelfMAD experiments with direct process execution")
    parser.add_argument("--datasets", nargs="+", default=["LMA"], help="Datasets to process")
    parser.add_argument("--run-models", default="both", choices=["siam", "main", "both"], help="Which models to run")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--run-all-models", action="store_true", help="Run all models defined in MAIN_MODELS")

    # SelfMAD-siam parameters
    parser.add_argument("--siam-model", default="vit_mae_large", help="Model for SelfMAD-siam")
    parser.add_argument("--siam-batch-size", type=int, default=32, help="Batch size for SelfMAD-siam")
    parser.add_argument("--siam-epochs", type=int, default=50, help="Number of epochs for SelfMAD-siam")
    parser.add_argument("--siam-learning-rate", type=float, default=0.0005, help="Learning rate for SelfMAD-siam")
    parser.add_argument("--siam-saving-strategy", default="testset_best", help="Saving strategy for SelfMAD-siam")
    parser.add_argument("--siam-train-val-split", type=float, default=0.8, help="Train/val split for SelfMAD-siam")
    parser.add_argument("--siam-early-stopping-patience", type=int, default=5, help="Early stopping patience for SelfMAD-siam")
    parser.add_argument("--siam-early-stopping-monitor", default="val_loss", help="Early stopping monitor for SelfMAD-siam")
    parser.add_argument("--siam-classification-threshold", type=float, default=0.5, help="Classification threshold for SelfMAD-siam")
    parser.add_argument("--siam-apcer-thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2], help="APCER thresholds for SelfMAD-siam")

    # SelfMAD-main parameters
    parser.add_argument("--main-model", default="hrnet_w18", help="Model for SelfMAD-main")
    parser.add_argument("--main-batch-size", type=int, default=32, help="Batch size for SelfMAD-main")
    parser.add_argument("--main-epochs", type=int, default=50, help="Number of epochs for SelfMAD-main")
    parser.add_argument("--main-learning-rate", type=float, default=0.0005, help="Learning rate for SelfMAD-main")
    parser.add_argument("--main-saving-strategy", default="testset_best", help="Saving strategy for SelfMAD-main")
    parser.add_argument("--main-train-val-split", type=float, default=0.8, help="Train/val split for SelfMAD-main")
    parser.add_argument("--main-classification-threshold", type=float, default=0.5, help="Classification threshold for SelfMAD-main")
    parser.add_argument("--main-apcer-thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2], help="APCER thresholds for SelfMAD-main")

    # Multi-model parameters
    parser.add_argument("--main-models", nargs="+", default=["hrnet_w18"], help="List of models to run when run-all-models is True")

    # Try to load configuration from file if it exists
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from automation_config import get_config
        config = get_config()
        print("Loaded configuration from automation_config.py")

        # Update default values with those from config
        parser.set_defaults(**config)
    except (ImportError, AttributeError) as e:
        print(f"Note: Could not load configuration from automation_config.py: {e}")

    args = parser.parse_args()

    # Add model configurations to args if available
    if hasattr(args, "main_model_configs"):
        args.main_model_configs = args.main_model_configs
    else:
        # Default model configurations
        args.main_model_configs = {
            "default": {"batch_size": 32, "learning_rate": 5e-4, "epochs": 2},
            "hrnet_w18": {"batch_size": 32, "learning_rate": 5e-4, "epochs": 2},
            "efficientnet-b4": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2},
            "efficientnet-b7": {"batch_size": 16, "learning_rate": 5e-4, "epochs": 2},
            "swin": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2},
            "resnet": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2}
        }

    return args

def run_command(command, cwd=None):
    """Run a command and wait for it to complete."""
    print(f"Running command: {command}")
    print(f"Working directory: {cwd}")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    # Return the exit code
    return process.returncode

def find_model_file(model_dir, model_type="", dataset=""):
    """Find the model file in the model directory or elsewhere in the output directory.

    Args:
        model_dir (str): The model directory to search in
        model_type (str): The model type ("siam" or "main")
        dataset (str): The dataset name

    Returns:
        str or None: The path to the model file, or None if not found
    """
    print(f"Searching for model file in {model_dir}...")

    # Normalize path separators
    model_dir = normalize_path(model_dir)

    # Check in weights directory first
    weights_dir = os.path.join(model_dir, "weights")
    if os.path.exists(weights_dir):
        print(f"Checking weights directory: {weights_dir}")
        # Look for specific epoch files first
        for i in range(1, 101):  # Check for epoch_1.tar through epoch_100.tar
            epoch_file = os.path.join(weights_dir, f"epoch_{i}.tar")
            if os.path.exists(epoch_file):
                print(f"Found specific epoch file: {epoch_file}")
                return epoch_file

        # If no specific epoch files found, look for any .tar files
        model_files = glob.glob(os.path.join(weights_dir, "*.tar"))
        if model_files:
            # Sort by modification time (newest first)
            model_files.sort(key=os.path.getmtime, reverse=True)
            print(f"Found {len(model_files)} model files in weights directory. Using: {model_files[0]}")
            return model_files[0]

    # Check directly in model directory
    print(f"Checking model directory: {model_dir}")
    model_files = glob.glob(os.path.join(model_dir, "*.tar"))
    if model_files:
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        print(f"Found {len(model_files)} model files in model directory. Using: {model_files[0]}")
        return model_files[0]

    # Check for epoch directories
    epoch_dirs = glob.glob(os.path.join(model_dir, "epoch_*"))
    if epoch_dirs:
        # Sort by epoch number (highest first)
        epoch_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]), reverse=True)
        latest_epoch_dir = epoch_dirs[0]
        print(f"Checking epoch directory: {latest_epoch_dir}")
        model_files = glob.glob(os.path.join(latest_epoch_dir, "*.tar"))
        if model_files:
            # Sort by modification time (newest first)
            model_files.sort(key=os.path.getmtime, reverse=True)
            print(f"Found {len(model_files)} model files in epoch directory. Using: {model_files[0]}")
            return model_files[0]

    # Search recursively in the model directory
    print(f"Searching recursively in model directory: {model_dir}")
    model_files = glob.glob(os.path.join(model_dir, "**", "*.tar"), recursive=True)
    if model_files:
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        print(f"Found {len(model_files)} model files in recursive search. Using: {model_files[0]}")
        return model_files[0]

    # If still not found, search in the parent directory
    if model_type and dataset:
        # Parent directory is the dataset directory
        parent_dir = os.path.dirname(model_dir)
        print(f"Searching in dataset directory: {parent_dir}")

        # With the new directory structure, we just need to look in the dataset directory
        matching_dirs = [parent_dir]

        # Print the matching directories for debugging
        if matching_dirs:
            print(f"Found {len(matching_dirs)} matching directories: {matching_dirs}")

        if matching_dirs:
            # Sort by serial number (highest first) if using serial numbers, otherwise by creation time
            def get_sort_key(dir_path):
                dir_name = os.path.basename(dir_path)
                # Try to extract serial number
                prefix = f"{dataset}_{model_type}_"
                if dir_name.startswith(prefix):
                    serial_str = dir_name[len(prefix):]
                    if serial_str.isdigit():
                        return int(serial_str) * 1000000  # Give high priority to serial numbers
                # Fall back to creation time
                return os.path.getctime(dir_path)

            matching_dirs.sort(key=get_sort_key, reverse=True)

            for dir_path in matching_dirs:
                print(f"Checking directory: {dir_path}")

                # Check in model/weights directory
                weights_dir = os.path.join(dir_path, "model", "weights")
                if os.path.exists(weights_dir):
                    model_files = glob.glob(os.path.join(weights_dir, "*.tar"))
                    if model_files:
                        # Sort by modification time (newest first)
                        model_files.sort(key=os.path.getmtime, reverse=True)
                        print(f"Found {len(model_files)} model files in {weights_dir}. Using: {model_files[0]}")
                        return model_files[0]

                # Check in model directory
                model_dir = os.path.join(dir_path, "model")
                if os.path.exists(model_dir):
                    model_files = glob.glob(os.path.join(model_dir, "*.tar"))
                    if model_files:
                        # Sort by modification time (newest first)
                        model_files.sort(key=os.path.getmtime, reverse=True)
                        print(f"Found {len(model_files)} model files in {model_dir}. Using: {model_files[0]}")
                        return model_files[0]

                # Search recursively in the directory
                model_files = glob.glob(os.path.join(dir_path, "**", "*.tar"), recursive=True)
                if model_files:
                    # Sort by modification time (newest first)
                    model_files.sort(key=os.path.getmtime, reverse=True)
                    print(f"Found {len(model_files)} model files in recursive search of {dir_path}. Using: {model_files[0]}")
                    return model_files[0]

    # If still not found, search for any model file in the output directory
    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
    print(f"Searching in entire output directory: {output_dir}")

    if model_type and dataset:
        # If model_type is "siam", use "vit_mae_large" instead of the provided model_type
        search_model_type = "vit_mae_large" if model_type.lower() == "siam" else model_type
        print(f"Using search model type: {search_model_type}")

        # For SIAM models, check the special directory structure first
        if model_type.lower() == "siam":
            # Try multiple possible SIAM model locations
            siam_patterns = [
                # Pattern 1: output/siam/vit_mae_large/model/weights/*.tar
                os.path.join(output_dir, model_type, search_model_type, "model", "weights", "*.tar"),

                # Pattern 2: output/siam/vit_mae_large/model/*.tar
                os.path.join(output_dir, model_type, search_model_type, "model", "*.tar"),

                # Pattern 3: output/siam/LMA_siam_*/model/weights/*.tar
                os.path.join(output_dir, model_type, f"{dataset}_siam_*", "model", "weights", "*.tar"),

                # Pattern 4: output/siam/LMA_siam_*/model/*.tar
                os.path.join(output_dir, model_type, f"{dataset}_siam_*", "model", "*.tar"),

                # Pattern 5: output/siam/model/weights/*.tar
                os.path.join(output_dir, model_type, "model", "weights", "*.tar"),

                # Pattern 6: output/siam/model/*.tar
                os.path.join(output_dir, model_type, "model", "*.tar"),

                # Pattern 7: SelfMAD-siam/output/siam/vit_mae_large/model/weights/*.tar
                os.path.join("SelfMAD-siam", "output", model_type, search_model_type, "model", "weights", "*.tar"),

                # Pattern 8: SelfMAD-siam/output/siam/vit_mae_large/model/*.tar
                os.path.join("SelfMAD-siam", "output", model_type, search_model_type, "model", "*.tar")
            ]

            # Try each pattern
            for i, pattern in enumerate(siam_patterns):
                print(f"Using SIAM search pattern {i+1}: {pattern}")
                model_files = glob.glob(pattern, recursive=True)

                if model_files:
                    # Sort by modification time (newest first)
                    model_files.sort(key=os.path.getmtime, reverse=True)
                    print(f"Found {len(model_files)} model files using SIAM pattern {i+1}. Using: {model_files[0]}")
                    return model_files[0]

            # If still not found, try a more aggressive recursive search
            print("No model found with specific patterns. Trying recursive search in output directory...")
            recursive_pattern = os.path.join(output_dir, "**", "*.tar")
            print(f"Using recursive search pattern: {recursive_pattern}")
            model_files = glob.glob(recursive_pattern, recursive=True)

            if model_files:
                # Sort by modification time (newest first)
                model_files.sort(key=os.path.getmtime, reverse=True)
                print(f"Found {len(model_files)} model files in recursive search. Using: {model_files[0]}")
                return model_files[0]

        # Look for any .tar file in the output directory that matches the repo type and model
        # Use search_model_type for the actual search pattern
        if model_type.lower() == "main":
            pattern = os.path.join(output_dir, model_type, search_model_type, "**", "*.tar")
        else:
            # Already handled by the SIAM patterns above
            pattern = ""
        print(f"Using search pattern: {pattern}")
        model_files = glob.glob(pattern, recursive=True) if pattern else []

        if model_files:
            # Sort by modification time (newest first)
            model_files.sort(key=os.path.getmtime, reverse=True)
            print(f"Found {len(model_files)} model files in output directory. Using: {model_files[0]}")
            return model_files[0]

    print("No model file found.")
    return None

def process_model(model_type, dataset, args, results_queue, model_name=None):
    """Process a single model (training and evaluation).

    Args:
        model_type (str): The model type ("siam" or "main")
        dataset (str): The dataset name
        args (argparse.Namespace): The command line arguments
        results_queue (queue.Queue): Queue to store results
        model_name (str, optional): Specific model name to use (for multi-model runs)
    """
    try:
        # Determine which model to use based on model type
        if model_type == "siam":
            current_model = model_name if model_name else args.siam_model
        else:  # main
            current_model = model_name if model_name else args.main_model

        # Print clear message about which model is being processed
        print(f"\n=== Processing {model_type.upper()} with model {current_model} for {dataset} ===\n")

        # Create the output directory structure based on model type
        # Use the root output directory, NOT inside the repository directory
        root_dir = os.path.abspath(os.path.dirname(__file__))
        repo_dir = os.path.join(root_dir, args.output_dir, model_type)

        if model_type == "siam":
            # For SIAM models, use the correct directory structure with dataset-specific folders
            # Format: output/siam/vit_mae_large/[dataset]/model/weights/
            # This ensures output is saved at project_root/output/siam/vit_mae_large/[dataset]/
            model_dir = os.path.join(repo_dir, current_model, dataset)  # Add dataset to path
            weights_dir = os.path.join(model_dir, "model", "weights")
            os.makedirs(weights_dir, exist_ok=True)
            print(f"Created SIAM weights directory for dataset {dataset}: {weights_dir}")
        else:  # main
            # For MAIN models, use the correct directory structure with dataset-specific folders
            # Format: output/main/[model_name]/[dataset]/model/
            # This ensures output is saved at project_root/output/main/{model_name}/{dataset}/
            model_dir = os.path.join(repo_dir, current_model, dataset)  # Add dataset to path
            model_weights_dir = os.path.join(model_dir, "model")
            os.makedirs(model_weights_dir, exist_ok=True)
            print(f"Created MAIN model directory for dataset {dataset}: {model_weights_dir}")

        # Create eval directory
        eval_dir = os.path.join(model_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Created eval directory: {eval_dir}")

        # Construct training command
        if model_type == "siam":
            # For SelfMAD-siam, we don't support multiple models yet
            # Use the correct directory structure for SIAM models with dataset-specific folders
            # Format: output/siam/vit_mae_large/[dataset]/model/weights/
            # Use the root output directory, NOT inside the repository directory
            # This ensures output is saved at project_root/output/siam/vit_mae_large/[dataset]/
            root_dir = os.path.abspath(os.path.dirname(__file__))
            siam_model_dir = os.path.join(root_dir, args.output_dir, "siam", args.siam_model, dataset)  # Add dataset to path
            absolute_model_dir = siam_model_dir
            train_command = (
                f"python train__.py "
                f"-n {args.siam_model} "  # Use model name as session name
                f"-m {args.siam_model} "
                f"-b {args.siam_batch_size} "
                f"-e {args.siam_epochs} "
                f"-lr {args.siam_learning_rate} "
                f"-v {args.siam_saving_strategy} "
                f"-t {dataset} "
                f"-s {absolute_model_dir} "  # Use the absolute path to the model directory
                f"-train_val_split {args.siam_train_val_split} "
            )

            # Add early stopping parameters separately to avoid parsing issues
            if hasattr(args, "siam_early_stopping_patience"):
                train_command += f" -early_stopping_patience {args.siam_early_stopping_patience}"

            if hasattr(args, "siam_early_stopping_monitor"):
                train_command += f" -early_stopping_monitor {args.siam_early_stopping_monitor}"

            # Add scheduler parameter if available
            if hasattr(args, "siam_scheduler"):
                train_command += f" -scheduler {args.siam_scheduler}"
        else:  # main
            # Get model-specific configuration if available
            model_configs = getattr(args, "main_model_configs", {})
            model_config = model_configs.get(current_model, model_configs.get("default", {}))

            # Use model-specific parameters if available, otherwise use defaults
            batch_size = model_config.get("batch_size", args.main_batch_size)
            learning_rate = model_config.get("learning_rate", args.main_learning_rate)
            epochs = model_config.get("epochs", args.main_epochs)

            # Use the correct directory structure for MAIN models with dataset-specific folders
            # Format: output/main/[model_name]/[dataset]/model/
            # Use the root output directory, NOT inside the repository directory
            # This ensures output is saved at project_root/output/main/{model_name}/{dataset}/
            root_dir = os.path.abspath(os.path.dirname(__file__))
            main_model_dir = os.path.join(root_dir, args.output_dir, "main", current_model, dataset)  # Add dataset to path
            absolute_model_dir = main_model_dir
            train_command = (
                f"python train__.py "
                f"-n {current_model} "  # Use model name as session name
                f"-m {current_model} "  # Use the specific model
                f"-b {batch_size} "
                f"-e {epochs} "
                f"-lr {learning_rate} "
                f"-v {args.main_saving_strategy} "
                f"-t {dataset} "
                f"-s {absolute_model_dir} "  # Use the absolute path to the model directory
                f"-train_val_split {args.main_train_val_split}"
            )

            # Add early stopping parameters separately to avoid parsing issues
            if hasattr(args, "main_early_stopping_patience"):
                train_command += f" -early_stopping_patience {args.main_early_stopping_patience}"

            if hasattr(args, "main_early_stopping_monitor"):
                train_command += f" -early_stopping_monitor {args.main_early_stopping_monitor}"

        # Print the command for debugging
        print(f"Training command: {train_command}")

        # Run training
        print(f"\n=== Running {model_type.upper()} training for {dataset} ===\n")
        # Use absolute paths in the command and add the python path to include the repository
        repo_dir = os.path.abspath(f"SelfMAD-{model_type}")
        # Modify the command to use the full path to the train__.py script
        train_script = os.path.join(repo_dir, "train__.py")
        # Replace "python train__.py" with the full path
        train_command = train_command.replace("python train__.py", f"python {train_script}")
        # Run the command from the root directory
        train_exit_code = run_command(train_command)
        training_success = (train_exit_code == 0)

        # Run evaluation if training was successful
        evaluation_success = False
        if training_success:
            if model_name:
                print(f"\n=== Running {model_type.upper()} ({model_name}) evaluation for {dataset} ===\n")
            else:
                print(f"\n=== Running {model_type.upper()} evaluation for {dataset} ===\n")

            # Find the model file
            model_path = None

            # Special case for SelfMAD-main, which might save models in eval directory
            if model_type == "main":
                # Always use the model-specific eval directory
                eval_check_dir = os.path.join(model_dir, "eval")

                if os.path.exists(eval_check_dir):
                    print(f"Checking eval directory for SelfMAD-main model: {eval_check_dir}")
                    epoch_dirs = glob.glob(os.path.join(eval_check_dir, "epoch_*"))
                    if epoch_dirs:
                        # Sort by epoch number (highest first)
                        epoch_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]), reverse=True)
                        latest_epoch_dir = epoch_dirs[0]
                        print(f"Checking latest epoch directory: {latest_epoch_dir}")
                        model_files = glob.glob(os.path.join(latest_epoch_dir, "*.tar"))
                        if model_files:
                            # Sort by modification time (newest first)
                            model_files.sort(key=os.path.getmtime, reverse=True)
                            model_path = model_files[0]
                            print(f"Found SelfMAD-main model in eval directory: {model_path}")

            # If not found in eval directory, use the general find_model_file function
            if not model_path:
                model_path = find_model_file(model_dir, model_type, dataset)

            if model_path and os.path.exists(model_path):
                print(f"Found model file: {model_path}")

                # Use the existing eval directory
                # No need to create a nested directory

                # Construct dataset path arguments
                dataset_path_args = ""
                if hasattr(args, "dataset_paths") and isinstance(args.dataset_paths, dict):
                    # Use dataset paths from automation_config.py
                    for path_name, path_value in args.dataset_paths.items():
                        dataset_path_args += f" -{path_name} {path_value}"
                    print(f"Using dataset paths from automation_config.py: {args.dataset_paths}")
                else:
                    # Default dataset paths using os.path.join for cross-platform compatibility
                    # Try to use absolute path to datasets directory
                    try:
                        # First try to get the absolute path to the datasets directory
                        datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
                        if not os.path.exists(datasets_dir):
                            # Fall back to relative path if absolute path doesn't exist
                            datasets_dir = os.path.join("..", "datasets")
                    except Exception as e:
                        print(f"Warning: Could not determine absolute path to datasets directory: {e}")
                        datasets_dir = os.path.join("..", "datasets")

                    print(f"Using datasets directory: {datasets_dir}")
                    dataset_path_args = f" -LMA_path {datasets_dir} -LMA_UBO_path {datasets_dir} -MIPGAN_I_path {datasets_dir} -MIPGAN_II_path {datasets_dir} -MorDiff_path {datasets_dir} -StyleGAN_path {datasets_dir} -LMA_MIPGAN_I_path {datasets_dir}"

                # Copy the model file to the model directory if it's not already there
                # This helps the eval scripts find the model file more easily
                model_filename = os.path.basename(model_path)
                model_link = os.path.join(model_dir, model_filename)

                # Make sure the model directory exists
                os.makedirs(model_dir, exist_ok=True)

                if not os.path.exists(model_link) and os.path.exists(model_path):
                    try:
                        import shutil
                        print(f"Copying model file from {model_path} to {model_link}")
                        shutil.copy2(model_path, model_link)
                        print(f"Successfully copied model file to {model_link}")
                    except Exception as e:
                        print(f"Warning: Could not copy model file: {str(e)}")
                        traceback.print_exc()

                # Construct evaluation command with output directory
                # Pass the eval_dir to ensure consistent output structure
                if model_type == "siam":
                    # For SIAM models, ensure we're using the correct model directory structure with dataset-specific folders
                    # Format: output/siam/vit_mae_large/[dataset]/eval/
                    # Use the root output directory, NOT inside the repository directory
                    root_dir = os.path.abspath(os.path.dirname(__file__))
                    siam_model_dir = os.path.join(root_dir, args.output_dir, "siam", args.siam_model, dataset)  # Add dataset to path
                    siam_eval_dir = os.path.join(siam_model_dir, "eval")
                    os.makedirs(siam_eval_dir, exist_ok=True)

                    eval_command = f"python eval__.py -m {args.siam_model} -p {model_path} -v{dataset_path_args} -o {siam_eval_dir} -train_dataset {dataset}"

                    # Add classification threshold if available
                    if hasattr(args, "siam_classification_threshold"):
                        eval_command += f" -t {args.siam_classification_threshold}"

                    # Add APCER thresholds if available
                    if hasattr(args, "siam_apcer_thresholds"):
                        thresholds_str = " ".join(str(t) for t in args.siam_apcer_thresholds)
                        eval_command += f" -apcer_thresholds {thresholds_str}"
                else:  # main
                    # For MAIN models, use the correct directory structure with dataset-specific folders
                    # Format: output/main/[model_name]/[dataset]/eval/
                    # Use the root output directory, NOT inside the repository directory
                    root_dir = os.path.abspath(os.path.dirname(__file__))
                    main_model_dir = os.path.join(root_dir, args.output_dir, "main", current_model, dataset)  # Add dataset to path
                    main_eval_dir = os.path.join(main_model_dir, "eval")
                    os.makedirs(main_eval_dir, exist_ok=True)

                    eval_command = f"python eval__.py -m {current_model} -p {model_path} -v{dataset_path_args} -o {main_eval_dir} -train_dataset {dataset}"

                    # Add classification threshold if available
                    if hasattr(args, "main_classification_threshold"):
                        eval_command += f" -t {args.main_classification_threshold}"

                    # Add APCER thresholds if available
                    if hasattr(args, "main_apcer_thresholds"):
                        thresholds_str = " ".join(str(t) for t in args.main_apcer_thresholds)
                        eval_command += f" -apcer_thresholds {thresholds_str}"

                # Run evaluation
                # Use absolute paths in the command and add the python path to include the repository
                # Modify the command to use the full path to the eval__.py script
                eval_script = os.path.join(repo_dir, "eval__.py")
                # Replace "python eval__.py" with the full path
                eval_command = eval_command.replace("python eval__.py", f"python {eval_script}")
                # Run the command from the root directory
                eval_exit_code = run_command(eval_command)
                evaluation_success = (eval_exit_code == 0)
            else:
                print(f"Warning: No {model_type.upper()} model found for {dataset}. Skipping evaluation.")
        else:
            print(f"Warning: {model_type.upper()} training failed for {dataset}. Skipping evaluation.")

        # Put results in queue
        results_queue.put((model_type, training_success, evaluation_success))

    except Exception as e:
        print(f"Error processing {model_type.upper()} for {dataset}: {str(e)}")
        traceback.print_exc()
        results_queue.put((model_type, False, False))

def process_dataset(dataset, args):
    """Process a dataset with direct process execution."""
    print(f"\n=== Processing dataset: {dataset} ===\n")

    try:
        # Create a queue for thread results
        results_queue = queue.Queue()
        threads = []

        # Check if we should run all models
        run_all_models = getattr(args, "run_all_models", False)

        # Create threads for selected models
        if args.run_models in ["siam", "both"]:
            siam_thread = threading.Thread(target=process_model, args=("siam", dataset, args, results_queue))
            threads.append(("siam", siam_thread))

        if args.run_models in ["main", "both"]:
            if run_all_models and hasattr(args, "main_models") and args.main_models:
                # Run all models in the list sequentially (not in threads)
                print(f"\n=== Running all models for {dataset} ===\n")
                for model_name in args.main_models:
                    print(f"\n=== Starting model: {model_name} ===\n")
                    # Run directly without threading to ensure proper directory creation
                    process_model("main", dataset, args, results_queue, model_name)
            else:
                # Run just the default model
                main_thread = threading.Thread(target=process_model, args=("main", dataset, args, results_queue))
                threads.append(("main", main_thread))

        # Start threads (only for non-multi-model runs)
        for _, thread in threads:
            thread.start()

        # Wait for threads to complete
        for _, thread in threads:
            thread.join()

        # Get results
        results = {}
        while not results_queue.empty():
            model_type, training_success, evaluation_success = results_queue.get()
            results[model_type] = (training_success, evaluation_success)

        # Summarize results
        print(f"\nProcessing summary for {dataset}:")

        overall_success = False

        if args.run_models in ["siam", "both"]:
            if "siam" in results:
                siam_training_success, siam_evaluation_success = results.get("siam", (False, False))
                print(f"  - SelfMAD-siam: Training {'Completed' if siam_training_success else 'Failed'}, "
                      f"Evaluation {'Completed' if siam_evaluation_success else 'Failed'}")
                if siam_training_success and siam_evaluation_success:
                    overall_success = True

        if args.run_models in ["main", "both"]:
            if run_all_models and hasattr(args, "main_models") and args.main_models:
                # Print results for each model
                for model_name in args.main_models:
                    model_key = f"main_{model_name}"
                    if model_key in results:
                        training_success, evaluation_success = results.get(model_key, (False, False))
                        print(f"  - SelfMAD-main ({model_name}): Training {'Completed' if training_success else 'Failed'}, "
                              f"Evaluation {'Completed' if evaluation_success else 'Failed'}")
                        if training_success and evaluation_success:
                            overall_success = True
            else:
                # Print results for the default model
                if "main" in results:
                    main_training_success, main_evaluation_success = results.get("main", (False, False))
                    print(f"  - SelfMAD-main: Training {'Completed' if main_training_success else 'Failed'}, "
                          f"Evaluation {'Completed' if main_evaluation_success else 'Failed'}")
                    if main_training_success and main_evaluation_success:
                        overall_success = True

        # Return success if at least one selected model completed both training and evaluation
        return overall_success

    except Exception as e:
        print(f"Error processing dataset {dataset}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print("\n=== SelfMAD Automation Configuration ===\n")
    print(f"Datasets: {args.datasets}")
    print(f"Models to run: {args.run_models}")
    print(f"Output directory: {args.output_dir}")
    print(f"Run all models: {args.run_all_models}")

    if args.run_all_models:
        print(f"Models to run: {args.main_models}")

    print("\nSelfMAD-siam parameters:")
    print(f"  Model: {args.siam_model}")
    print(f"  Batch size: {args.siam_batch_size}")
    print(f"  Epochs: {args.siam_epochs}")
    print(f"  Learning rate: {args.siam_learning_rate}")
    print(f"  Saving strategy: {args.siam_saving_strategy}")
    print(f"  Train/val split: {args.siam_train_val_split}")
    print(f"  Early stopping patience: {args.siam_early_stopping_patience}")
    print(f"  Early stopping monitor: {args.siam_early_stopping_monitor}")
    print(f"  Classification threshold: {args.siam_classification_threshold}")
    print(f"  APCER thresholds: {args.siam_apcer_thresholds}")

    print("\nSelfMAD-main parameters:")
    if args.run_all_models:
        print(f"  Models: {args.main_models}")
        print("  Model-specific configurations:")
        for model_name in args.main_models:
            config = args.main_model_configs.get(model_name, args.main_model_configs.get("default", {}))
            print(f"    - {model_name}:")
            print(f"      Batch size: {config.get('batch_size', args.main_batch_size)}")
            print(f"      Learning rate: {config.get('learning_rate', args.main_learning_rate)}")
            print(f"      Epochs: {config.get('epochs', args.main_epochs)}")
    else:
        print(f"  Model: {args.main_model}")
        print(f"  Batch size: {args.main_batch_size}")
        print(f"  Epochs: {args.main_epochs}")
        print(f"  Learning rate: {args.main_learning_rate}")

    print(f"  Saving strategy: {args.main_saving_strategy}")
    print(f"  Train/val split: {args.main_train_val_split}")
    print(f"  Classification threshold: {args.main_classification_threshold}")
    print(f"  APCER thresholds: {args.main_apcer_thresholds}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each dataset
    for dataset in args.datasets:
        success = process_dataset(dataset, args)
        if not success:
            print(f"Warning: Processing failed for dataset {dataset}.")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()
