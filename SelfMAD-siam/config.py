import os
import json
import argparse

class Config:
    def __init__(self):
        # Dataset paths
        self.datasets = {
            "LMA": os.path.join("..", "datasets"),
            "LMA_UBO": os.path.join("..", "datasets"),
            "MIPGAN_I": os.path.join("..", "datasets"),
            "MIPGAN_II": os.path.join("..", "datasets"),
            "MorDiff": os.path.join("..", "datasets"),
            "StyleGAN": os.path.join("..", "datasets"),
            "LMA_MIPGAN_I": os.path.join("..", "datasets")
        }

        # Training configuration
        self.train_dataset = "LMA"  # Default dataset for training
        self.test_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]
        self.train_val_split = 0.8  # 80% training, 20% validation

        # Model configuration
        self.model = "vit_mae_large"
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 5e-4
        self.image_size = 224  # ViT-MAE default image size

        # Output configuration
        self.save_path = "./output/model"
        self.csv_output_path = "./output/train"
        self.eval_output_path = "./output/eval"

        # Early stopping configuration
        self.early_stopping_patience = 5
        self.early_stopping_monitor = "val_loss"  # Options: "val_loss", "train_loss", "val_acc"

        # Evaluation configuration
        self.classification_threshold = 0.5  # Default threshold for binary classification
        self.apcer_thresholds = [0.05, 0.10, 0.20]  # APCER thresholds for BPCER calculation (5%, 10%, 20%)

    def update_from_args(self, args):
        """Update config from command line arguments"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def update_from_json(self, json_path):
        """Update config from JSON file"""
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def save_to_json(self, json_path):
        """Save config to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

def get_config():
    """Get configuration with command line arguments"""
    parser = argparse.ArgumentParser(description='Morph Detection Configuration')
    parser.add_argument('--train_dataset', type=str, help='Dataset for training')
    parser.add_argument('--model', type=str, help='Model type (vit_mae_large)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--train_val_split', type=float, help='Train/validation split ratio')
    parser.add_argument('--save_path', type=str, help='Path to save models')
    parser.add_argument('--config_file', type=str, default='./configs/morph_config.json',
                        help='Path to config JSON file')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--early_stopping_monitor', type=str, help='Metric to monitor for early stopping (val_loss, train_loss, val_acc)')

    args = parser.parse_args()

    config = Config()

    # Update from JSON if provided
    if args.config_file and os.path.exists(args.config_file):
        config.update_from_json(args.config_file)

    # Update from command line args
    config.update_from_args(args)

    return config
