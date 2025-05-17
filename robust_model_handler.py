"""
Robust model file handling module for SelfMAD project.

This module provides functions for finding, copying, and loading model files
with improved error handling and path normalization.
"""

import os
import glob
import shutil
import torch
import traceback

import logging

# Set up logging
def setup_logger():
    """Set up and return a logger that writes to both file and console.

    This function can be called multiple times and will ensure the logger
    is properly configured each time, appending to the existing log file.
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Get the logger
        logger = logging.getLogger("model_handler")

        # Clear any existing handlers to avoid duplicate logs
        if logger.handlers:
            logger.handlers.clear()

        # Set the logging level
        logger.setLevel(logging.INFO)

        # Create a file handler that appends to the log file
        log_file_path = os.path.join("logs", "model_handler.log")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Test log message to verify logging is working
        logger.info("Logger initialized successfully")
        print(f"Logging to file: {os.path.abspath(log_file_path)}")

        return logger
    except Exception as e:
        print(f"Error setting up logger: {e}")
        # Create a basic logger as fallback
        basic_logger = logging.getLogger("model_handler_basic")
        if not basic_logger.handlers:
            basic_handler = logging.StreamHandler()
            basic_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            basic_handler.setFormatter(basic_format)
            basic_logger.addHandler(basic_handler)
        basic_logger.setLevel(logging.INFO)
        basic_logger.info(f"Using basic logger due to error: {e}")
        return basic_logger

# Initialize the logger
logger = setup_logger()

def normalize_path(path):
    """Normalize a path to use the correct path separators for the current OS.

    Args:
        path (str): The path to normalize

    Returns:
        str: The normalized path
    """
    if path is None:
        return None

    # Convert to absolute path if not already
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Normalize path separators
    return os.path.normpath(path)

def find_model_file(model_dir, model_type="", dataset="", return_abs_path=True):
    """Find the model file in the model directory or elsewhere in the output directory.

    Args:
        model_dir (str): The model directory to search in
        model_type (str): The model type ("siam" or "main")
        dataset (str): The dataset name
        return_abs_path (bool): Whether to return an absolute path

    Returns:
        str or None: The path to the model file, or None if not found
    """
    # Use the global logger but don't reinitialize it
    global logger

    # Log the search
    logger.info(f"Searching for model file in {model_dir}... (model_type={model_type}, dataset={dataset})")

    # Normalize path separators and convert to absolute path
    model_dir = normalize_path(model_dir)

    # Check if model_dir exists
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory {model_dir} does not exist.")
        # Try to create it
        try:
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory {model_dir}")
        except Exception as e:
            logger.error(f"Failed to create model directory {model_dir}: {e}")
            return None

    # List of locations to check for model files
    search_locations = [
        os.path.join(model_dir, "weights"),  # Check in weights directory first
        model_dir,  # Check directly in model directory
        os.path.join(model_dir, "model", "weights"),  # Check in model/weights directory
        os.path.join(model_dir, "model")  # Check in model directory
    ]

    # Check each location for model files
    for location in search_locations:
        if os.path.exists(location):
            logger.info(f"Checking location: {location}")

            # Look for specific epoch files first
            for i in range(1, 101):  # Check for epoch_1.tar through epoch_100.tar
                epoch_file = os.path.join(location, f"epoch_{i}.tar")
                if os.path.exists(epoch_file):
                    logger.info(f"Found specific epoch file: {epoch_file}")
                    return normalize_path(epoch_file) if return_abs_path else epoch_file

            # Look for best.tar or early_stopped_best.tar
            for filename in ["best.tar", "early_stopped_best.tar"]:
                model_file = os.path.join(location, filename)
                if os.path.exists(model_file):
                    logger.info(f"Found model file: {model_file}")
                    return normalize_path(model_file) if return_abs_path else model_file

            # If no specific files found, look for any .tar files
            model_files = glob.glob(os.path.join(location, "*.tar"))
            if model_files:
                # Sort by modification time (newest first)
                model_files.sort(key=os.path.getmtime, reverse=True)
                logger.info(f"Found {len(model_files)} model files in {location}. Using: {model_files[0]}")
                return normalize_path(model_files[0]) if return_abs_path else model_files[0]

    # Search recursively in the model directory
    logger.info(f"Searching recursively in model directory: {model_dir}")
    model_files = glob.glob(os.path.join(model_dir, "**", "*.tar"), recursive=True)
    if model_files:
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"Found {len(model_files)} model files in recursive search. Using: {model_files[0]}")
        return normalize_path(model_files[0]) if return_abs_path else model_files[0]

    # If still not found, search in the parent directory
    if model_type and dataset:
        parent_dir = os.path.dirname(os.path.dirname(model_dir))
        logger.info(f"Searching in parent directory: {parent_dir}")

        # For SIAM models, check the special directory structure first
        if model_type.lower() == "siam":
            # Look for models in output/siam/vit_mae_large/model/weights/
            search_model_type = "vit_mae_large"

            # Only search in the root output directory
            output_dir = os.path.abspath(os.path.join(os.path.dirname(parent_dir), "output"))

            # Try only the root output directory patterns
            siam_patterns = [
                # Primary pattern: output/siam/vit_mae_large/model/weights/*.tar
                os.path.join(output_dir, model_type, search_model_type, "model", "weights", "*.tar"),

                # Alternative pattern: output/siam/vit_mae_large/model/*.tar
                os.path.join(output_dir, model_type, search_model_type, "model", "*.tar")
            ]

            # Try each pattern
            for i, pattern in enumerate(siam_patterns):
                logger.info(f"Using SIAM search pattern {i+1}: {pattern}")
                model_files = glob.glob(pattern, recursive=True)

                if model_files:
                    # Sort by modification time (newest first)
                    model_files.sort(key=os.path.getmtime, reverse=True)
                    logger.info(f"Found {len(model_files)} model files using SIAM pattern {i+1}. Using: {model_files[0]}")
                    return normalize_path(model_files[0]) if return_abs_path else model_files[0]

            # If still not found, throw an error
            error_msg = f"No model files found in root output directory for {model_type}/{search_model_type}. " \
                        f"Please ensure models are saved in the correct location: {output_dir}/{model_type}/{search_model_type}/model/weights/"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Look for any directory matching the dataset and model type
        # If model_type is "siam", use "vit_mae_large" instead of the provided model_type
        search_model_type = "vit_mae_large" if model_type.lower() == "siam" else model_type
        pattern = f"{dataset}_{search_model_type}_*"
        logger.info(f"Using search pattern: {pattern}")
        matching_dirs = glob.glob(os.path.join(parent_dir, pattern))

        if matching_dirs:
            # Sort by creation time (newest first)
            matching_dirs.sort(key=os.path.getctime, reverse=True)

            for dir_path in matching_dirs:
                logger.info(f"Checking directory: {dir_path}")

                # Check in model/weights directory
                weights_dir = os.path.join(dir_path, "model", "weights")
                if os.path.exists(weights_dir):
                    model_files = glob.glob(os.path.join(weights_dir, "*.tar"))
                    if model_files:
                        # Sort by modification time (newest first)
                        model_files.sort(key=os.path.getmtime, reverse=True)
                        logger.info(f"Found {len(model_files)} model files in {weights_dir}. Using: {model_files[0]}")
                        return normalize_path(model_files[0]) if return_abs_path else model_files[0]

                # Check in model directory
                model_dir = os.path.join(dir_path, "model")
                if os.path.exists(model_dir):
                    model_files = glob.glob(os.path.join(model_dir, "*.tar"))
                    if model_files:
                        # Sort by modification time (newest first)
                        model_files.sort(key=os.path.getmtime, reverse=True)
                        logger.info(f"Found {len(model_files)} model files in {model_dir}. Using: {model_files[0]}")
                        return normalize_path(model_files[0]) if return_abs_path else model_files[0]

                # Search recursively in the directory
                model_files = glob.glob(os.path.join(dir_path, "**", "*.tar"), recursive=True)
                if model_files:
                    # Sort by modification time (newest first)
                    model_files.sort(key=os.path.getmtime, reverse=True)
                    logger.info(f"Found {len(model_files)} model files in recursive search of {dir_path}. Using: {model_files[0]}")
                    return normalize_path(model_files[0]) if return_abs_path else model_files[0]

    logger.warning("No model file found.")
    return None

def copy_model_file(source_path, target_dir, target_filename=None):
    """Copy a model file to a target directory with improved error handling.

    Args:
        source_path (str): Path to the source model file
        target_dir (str): Directory to copy the model file to
        target_filename (str, optional): Name to give the copied file. If None, uses the original filename.

    Returns:
        str or None: Path to the copied file, or None if copying failed
    """
    # Use the global logger but don't reinitialize it
    global logger
    # Normalize paths
    source_path = normalize_path(source_path)
    target_dir = normalize_path(target_dir)

    # Check if source file exists
    if not os.path.exists(source_path):
        logger.error(f"Source file {source_path} does not exist.")
        return None

    # Create target directory if it doesn't exist
    try:
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Ensured target directory exists: {target_dir}")
    except Exception as e:
        logger.error(f"Failed to create target directory {target_dir}: {e}")
        return None

    # Determine target filename
    if target_filename is None:
        target_filename = os.path.basename(source_path)

    target_path = os.path.join(target_dir, target_filename)

    # Copy the file
    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Successfully copied model file from {source_path} to {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Failed to copy model file from {source_path} to {target_path}: {e}")
        return None

def get_next_serial_number(base_dir, prefix):
    """Get the next available serial number for a given prefix.

    Args:
        base_dir (str): The base directory to search in
        prefix (str): The prefix to match (e.g., "LMA_main_")

    Returns:
        int: The next available serial number
    """
    # Use the global logger but don't reinitialize it
    global logger
    # Normalize the base directory path
    base_dir = normalize_path(base_dir)

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

def load_model_with_retry(model_path, model_type="", dataset="", max_retries=3):
    """Load a model file with retry mechanism.

    Args:
        model_path (str): Path to the model file
        model_type (str): The model type ("siam" or "main")
        dataset (str): The dataset name
        max_retries (int): Maximum number of retries

    Returns:
        dict or None: The loaded model state, or None if loading failed
    """
    # Use the global logger but don't reinitialize it
    global logger
    # Normalize path
    model_path = normalize_path(model_path)

    # Try to load the model
    for retry in range(max_retries):
        try:
            logger.info(f"Attempt {retry+1}/{max_retries} to load model from {model_path}")

            # Check if file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} does not exist.")

                # Try to find an alternative model file
                alternative_path = find_model_file(os.path.dirname(model_path), model_type, dataset)
                if alternative_path and os.path.exists(alternative_path):
                    logger.info(f"Found alternative model file: {alternative_path}")
                    model_path = alternative_path
                else:
                    logger.error(f"No alternative model file found. Retrying...")
                    continue

            # Load the model
            # Set weights_only=False to allow loading NumPy data types in PyTorch 2.6+
            model_state = torch.load(model_path, weights_only=False)
            logger.info(f"Successfully loaded model from {model_path}")
            return model_state

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            traceback.print_exc()

            # If this is the last retry, try to find an alternative model file
            if retry == max_retries - 1:
                logger.warning(f"All retries failed. Trying to find an alternative model file...")
                alternative_path = find_model_file(os.path.dirname(os.path.dirname(model_path)), model_type, dataset)
                if alternative_path and os.path.exists(alternative_path):
                    try:
                        logger.info(f"Found alternative model file: {alternative_path}")
                        # Set weights_only=False to allow loading NumPy data types in PyTorch 2.6+
                        model_state = torch.load(alternative_path, weights_only=False)
                        logger.info(f"Successfully loaded model from alternative path: {alternative_path}")
                        return model_state
                    except Exception as e2:
                        logger.error(f"Error loading model from alternative path {alternative_path}: {e2}")
                        traceback.print_exc()

    logger.error(f"Failed to load model after {max_retries} retries.")
    return None
