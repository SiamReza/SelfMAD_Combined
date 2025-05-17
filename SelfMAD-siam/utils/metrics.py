from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import logging
import os

# Set up logging
def setup_logger():
    """Set up and return a logger that writes to both file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get the logger
    logger = logging.getLogger("metrics")

    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a file handler that appends to the log file
    file_handler = logging.FileHandler(os.path.join("logs", "metrics.log"), mode='a')
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

    return logger

# Initialize the logger
logger = setup_logger()

def calculate_eer(y_true, y_score):
    """Calculate Equal Error Rate (EER) with robust error handling.

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores

    Returns:
        EER value, or 0.5 if calculation fails
    """
    # Check if there are enough samples and both classes are present
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        logger.warning("Not enough samples or only one class present. Returning EER=0.5")
        return 0.5

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    except Exception as e:
        logger.error(f"Error calculating EER: {e}")
        return 0.5

def calculate_auc(y_true, y_score):
    """Calculate Area Under ROC Curve (AUC) with robust error handling.

    Args:
        y_true: Ground truth labels
        y_score: Prediction scores

    Returns:
        AUC value, or 0.5 if calculation fails
    """
    # Check if there are enough samples and both classes are present
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        logger.warning("Not enough samples or only one class present. Returning AUC=0.5")
        return 0.5

    try:
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        logger.error(f"Error calculating AUC: {e}")
        return 0.5
