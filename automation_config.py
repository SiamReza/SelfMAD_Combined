"""
Configuration file for SelfMAD automation script.

This file contains all configurable parameters for both SelfMAD-siam and SelfMAD-main repositories.
Modify the values in this file to customize the behavior of the automation script.
"""

# Common parameters
# -----------------

# List of datasets to process
DATASETS = ["LMA_MIPGAN_I"]  # Options: "LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"

# Enable combined dataset loading for training
# When True, all datasets in DATASETS will be combined for training
# When False, each dataset will be processed separately
ENABLE_COMBINED_DATASET = False # DO NOT USE TRUE THAT GIVES ERROR

# Which models to run
RUN_MODELS = "both"  # Options: "siam", "main", "both"

# Whether to run all models defined in MAIN_MODELS
# Set to True to run all models, False to run only the model specified in MAIN_MODEL
RUN_ALL_MODELS = False

# Base output directory
OUTPUT_DIR = "./output"

# Dataset paths (relative to the repository root) using os.path.join for cross-platform compatibility
import os
# datasets_dir = os.path.join("..", "datasets")
datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
DATASET_PATHS = {
    "LMA_path": datasets_dir,
    "LMA_UBO_path": datasets_dir,
    "MIPGAN_I_path": datasets_dir,
    "MIPGAN_II_path": datasets_dir,
    "MorDiff_path": datasets_dir,
    "StyleGAN_path": datasets_dir,
    "LMA_MIPGAN_I_path": datasets_dir
}

# SelfMAD-siam parameters
# -----------------------

# Model parameters
SIAM_MODEL = "vit_mae_large"  # Model type (vit_mae_large)
SIAM_BATCH_SIZE = 64          # Batch size for training
SIAM_EPOCHS =120               # Number of epochs for training (reduced for testing)
SIAM_LEARNING_RATE = 5e-3     # Learning rate for training
SIAM_IMAGE_SIZE = 224         # Image size for training (224x224)

# Training parameters
SIAM_SAVING_STRATEGY = "testset_best"  # Strategy for saving models (original, testset_best)
SIAM_TRAIN_VAL_SPLIT = 0.8             # Train/validation split ratio

# Early stopping parameters (SelfMAD-siam only)
SIAM_EARLY_STOPPING_PATIENCE = 100      # Patience for early stopping (increased from 5)
SIAM_EARLY_STOPPING_MONITOR = "val_acc"  # Metric to monitor for early stopping (val_loss, train_loss, val_acc)

# Evaluation parameters (SelfMAD-siam only)
SIAM_CLASSIFICATION_THRESHOLD = 0.5    # Classification threshold for binary classification
SIAM_APCER_THRESHOLDS = [0.05, 0.10, 0.20]  # APCER thresholds for BPCER calculation (5%, 10%, 20%)

# Additional parameters (SelfMAD-siam only)
SIAM_SCHEDULER = "cosine"              # Learning rate scheduler (cosine or linear)

# ViT MAE improvement parameters
SIAM_VIT_UNFREEZE_LAYERS = 6           # Number of encoder layers to unfreeze (0 to freeze all)
SIAM_VIT_USE_ADAMW = True              # Use AdamW optimizer instead of SAM+SGD
SIAM_VIT_ENCODER_WEIGHT_DECAY = 0.05   # Weight decay for encoder parameters
SIAM_VIT_CLASSIFIER_WEIGHT_DECAY = 0.01 # Weight decay for classifier parameters
SIAM_VIT_WARMUP_PERCENTAGE = 0.25       # Percentage of total steps for learning rate warm-up
SIAM_VIT_LABEL_SMOOTHING = 0.1         # Label smoothing factor for CrossEntropyLoss
SIAM_VIT_GRADIENT_ACCUMULATION_STEPS = 2 # Number of steps to accumulate gradients for large models (4-8 recommended)
SIAM_VIT_ADVANCED_AUGMENTATIONS = True # Use advanced augmentations for better generalization

# Resume training parameters (SelfMAD-siam)
SIAM_RESUME_TRAINING = False        # Whether to resume training from a checkpoint
SIAM_RESUME_CHECKPOINT = ""         # Path to the checkpoint to resume from (if empty, will use latest.tar)


# SelfMAD-main parameters
# ----------------------

# Model Selection
# -------------
# Set the model to use for single-model training (when RUN_ALL_MODELS = False)
MAIN_MODEL = "hrnet_w18"  # Options: "hrnet_w18", "efficientnet-b4", "efficientnet-b7", "swin", "resnet"

# List of all models to run when RUN_ALL_MODELS = True
MAIN_MODELS = ["hrnet_w18", "efficientnet-b4", "efficientnet-b7", "swin", "resnet"]

# Model-specific configurations
# These settings override the default parameters when a specific model is used
MAIN_MODEL_CONFIGS = {
    # Default configuration (used when a specific model config is not found)
    "default": {
        "batch_size": 32,
        "learning_rate": 5e-4,
        "epochs": 100
    },
    # Model-specific configurations
    "hrnet_w18": {
        "batch_size": 32,
        "learning_rate": 5e-4,
        "epochs": 100
    },
    "efficientnet-b4": {
        "batch_size": 32,  # Reduced batch size for larger model
        "learning_rate": 5e-4,
        "epochs": 100
    },
    "efficientnet-b7": {
        "batch_size": 32,  # Further reduced batch size for even larger model
        "learning_rate": 5e-4,
        "epochs": 100
    },
    "swin": {
        "batch_size": 32,  # Reduced batch size for larger model
        "learning_rate": 5e-4,
        "epochs": 100
    },
    "resnet": {
        "batch_size": 32,  # Reduced batch size for larger model
        "learning_rate": 5e-4,
        "epochs": 100
    }
}

# Default parameters (used when not overridden by model-specific config)
MAIN_BATCH_SIZE = 32          # Default batch size for training
MAIN_EPOCHS = 100               # Default number of epochs for training
MAIN_LEARNING_RATE = 5e-4     # Default learning rate for training

# Training parameters
MAIN_SAVING_STRATEGY = "testset_best"  # Strategy for saving models (original, testset_best)
MAIN_TRAIN_VAL_SPLIT = 0.8             # Train/validation split ratio

# Resume training parameters (SelfMAD-main)
MAIN_RESUME_TRAINING = False        # Whether to resume training from a checkpoint
MAIN_RESUME_CHECKPOINT = ""         # Path to the checkpoint to resume from (if empty, will use latest.tar)

# Evaluation parameters (SelfMAD-main only)
MAIN_CLASSIFICATION_THRESHOLD = 0.5    # Classification threshold for binary classification
MAIN_APCER_THRESHOLDS = [0.05, 0.10, 0.20]  # APCER thresholds for BPCER calculation (5%, 10%, 20%)

# Function to get all parameters as a dictionary
def get_config():
    """Get all configuration parameters as a dictionary."""
    config = {
        # Common parameters
        "datasets": DATASETS,
        "enable_combined_dataset": ENABLE_COMBINED_DATASET,
        "run_models": RUN_MODELS,
        "run_all_models": RUN_ALL_MODELS,
        "output_dir": OUTPUT_DIR,
        "dataset_paths": DATASET_PATHS,

        # SelfMAD-siam parameters
        "siam_model": SIAM_MODEL,
        "siam_batch_size": SIAM_BATCH_SIZE,
        "siam_epochs": SIAM_EPOCHS,
        "siam_learning_rate": SIAM_LEARNING_RATE,
        "siam_saving_strategy": SIAM_SAVING_STRATEGY,
        "siam_train_val_split": SIAM_TRAIN_VAL_SPLIT,
        "siam_early_stopping_patience": SIAM_EARLY_STOPPING_PATIENCE,
        "siam_early_stopping_monitor": SIAM_EARLY_STOPPING_MONITOR,
        "siam_classification_threshold": SIAM_CLASSIFICATION_THRESHOLD,
        "siam_apcer_thresholds": SIAM_APCER_THRESHOLDS,

        # SelfMAD-main parameters
        "main_model": MAIN_MODEL,
        "main_batch_size": MAIN_BATCH_SIZE,
        "main_epochs": MAIN_EPOCHS,
        "main_learning_rate": MAIN_LEARNING_RATE,
        "main_saving_strategy": MAIN_SAVING_STRATEGY,
        "main_train_val_split": MAIN_TRAIN_VAL_SPLIT,
        "main_classification_threshold": MAIN_CLASSIFICATION_THRESHOLD,
        "main_apcer_thresholds": MAIN_APCER_THRESHOLDS,

        # Multi-model parameters
        "main_models": MAIN_MODELS,
        "main_model_configs": MAIN_MODEL_CONFIGS,
    }

    # Add advanced parameters if they are defined
    for param in dir():
        if param.startswith("SIAM_") and param not in ["SIAM_MODEL", "SIAM_BATCH_SIZE", "SIAM_EPOCHS",
                                                      "SIAM_LEARNING_RATE", "SIAM_SAVING_STRATEGY",
                                                      "SIAM_TRAIN_VAL_SPLIT", "SIAM_EARLY_STOPPING_PATIENCE",
                                                      "SIAM_EARLY_STOPPING_MONITOR", "SIAM_CLASSIFICATION_THRESHOLD",
                                                      "SIAM_APCER_THRESHOLDS", "SIAM_IMAGE_SIZE",
                                                      "SIAM_VIT_UNFREEZE_LAYERS", "SIAM_VIT_USE_ADAMW",
                                                      "SIAM_VIT_ENCODER_WEIGHT_DECAY", "SIAM_VIT_CLASSIFIER_WEIGHT_DECAY",
                                                      "SIAM_VIT_WARMUP_PERCENTAGE", "SIAM_VIT_LABEL_SMOOTHING",
                                                      "SIAM_VIT_GRADIENT_ACCUMULATION_STEPS", "SIAM_VIT_ADVANCED_AUGMENTATIONS"]:
            config[param.lower()] = globals()[param]
        elif param.startswith("MAIN_") and param not in ["MAIN_MODEL", "MAIN_BATCH_SIZE", "MAIN_EPOCHS",
                                                        "MAIN_LEARNING_RATE", "MAIN_SAVING_STRATEGY",
                                                        "MAIN_TRAIN_VAL_SPLIT", "MAIN_CLASSIFICATION_THRESHOLD",
                                                        "MAIN_APCER_THRESHOLDS"]:
            config[param.lower()] = globals()[param]

    # Add ViT MAE improvement parameters to config
    config["siam_image_size"] = SIAM_IMAGE_SIZE
    config["siam_vit_unfreeze_layers"] = SIAM_VIT_UNFREEZE_LAYERS
    config["siam_vit_use_adamw"] = SIAM_VIT_USE_ADAMW
    config["siam_vit_encoder_weight_decay"] = SIAM_VIT_ENCODER_WEIGHT_DECAY
    config["siam_vit_classifier_weight_decay"] = SIAM_VIT_CLASSIFIER_WEIGHT_DECAY
    config["siam_vit_warmup_percentage"] = SIAM_VIT_WARMUP_PERCENTAGE
    config["siam_vit_label_smoothing"] = SIAM_VIT_LABEL_SMOOTHING
    config["siam_vit_gradient_accumulation_steps"] = SIAM_VIT_GRADIENT_ACCUMULATION_STEPS
    config["siam_vit_advanced_augmentations"] = SIAM_VIT_ADVANCED_AUGMENTATIONS

    # Add resume training parameters to config
    config["siam_resume_training"] = SIAM_RESUME_TRAINING
    config["siam_resume_checkpoint"] = SIAM_RESUME_CHECKPOINT
    config["main_resume_training"] = MAIN_RESUME_TRAINING
    config["main_resume_checkpoint"] = MAIN_RESUME_CHECKPOINT

    return config
