import os
import torch
import torch.nn as nn
import numpy as np
import os
import random
import pandas as pd
import json
import time
import re
from utils.selfMAD import selfMAD_Dataset
from utils.custom_dataset import MorphDataset, CombinedMorphDataset
from utils.scheduler import LinearDecayLR, CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import argparse
from utils.logs import log
from datetime import datetime
from tqdm import tqdm
from utils.model import Detector
import json
from utils.metrics import calculate_eer, calculate_auc
from eval__ import default_datasets, prep_dataloaders, evaluate
import sys
sys.path.append('..')
from evaluation_1 import calculate_precision, calculate_recall, calculate_f1, calculate_auc, calculate_eer, calculate_apcer_bpcer_acer
from evaluation_2 import evaluate
from visualization_3 import visualize

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):  # Add specific check for float32
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):  # General floating types
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_next_serial_number(base_dir, prefix):
    """Get the next available serial number for a given prefix.

    Args:
        base_dir (str): The base directory to search in
        prefix (str): The prefix to match (e.g., "LMA_siam_")

    Returns:
        int: The next available serial number
    """
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

def main(args):

    assert args["model"] in ["efficientnet-b4", "efficientnet-b7", "swin", "resnet", "hrnet_w18", "hrnet_w32", "hrnet_w44", "hrnet_w64", "vit_mae_large"]
    assert args["train_dataset"] in ["FF++", "SMDD", "LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]
    assert args["saving_strategy"] in ["original", "testset_best"]

    # Create centralized output directory structure
    # Check if a session name was provided (which might already have a serial number)
    if args.get("session_name"):
        model_name = args["session_name"]
        print(f"Using provided session name: {model_name}")
    else:
        # Get the next serial number
        # Use absolute path to the root output directory
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        base_dir = os.path.join(parent_dir, "output", "siam")
        prefix = f"{args['train_dataset']}_siam_"
        serial_number = get_next_serial_number(base_dir, prefix)
        model_name = f"{args['train_dataset']}_siam_{serial_number}"
        print(f"Generated session name with serial number: {model_name}")

    # Function to check if a path is inside a repository output directory
    def is_repository_output_dir(path):
        """Check if the path is inside a repository output directory."""
        norm_path = os.path.normpath(path)
        return (
            norm_path.startswith(os.path.normpath("output")) or
            norm_path.startswith(os.path.normpath("./output")) or
            norm_path.startswith(os.path.normpath("SelfMAD-siam/output")) or
            norm_path.startswith(os.path.normpath("SelfMAD-main/output"))
        )

    # Determine the output directory
    if args.get("save_path"):
        # Use the provided save path
        output_dir = args["save_path"]

        # Check if the path is inside a repository output directory
        if is_repository_output_dir(output_dir):
            # Convert to absolute path relative to the parent directory
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(parent_dir, "output", "siam", model_name)
            print(f"Warning: Redirecting output to root directory: {output_dir}")
    else:
        # Use a path relative to the parent directory (outside the repository)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(parent_dir, "output", "siam", model_name)

    # Ensure we're using an absolute path
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    print(f"Using output directory: {output_dir}")

    model_dir = os.path.join(output_dir, "model")
    train_dir = os.path.join(output_dir, "train")
    eval_dir = os.path.join(output_dir, "eval")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "config"), exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "plots"), exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = {
        'train': [],
        'val': [],
        'batch': []
    }

    # FOR REPRODUCIBILITY
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = {
        "session_name": args["session_name"],
        "train_dataset": args["train_dataset"],
        "model": args["model"],
        "epochs": args["epochs"],
        "batch_size": args["batch_size"],
        "learning_rate": args["lr"],
        "image_size": 224 if args["model"] == "vit_mae_large" else (384 if "hrnet" in args["model"] else 380),
        "saving_strategy": args["saving_strategy"],
        "scheduler": args.get("scheduler", "cosine"),
    }

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']

    # Check if using custom morph dataset
    if args["train_dataset"] in ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]:
        # Create directory for CSV files if it doesn't exist
        # Always use the root output directory for CSV files
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if args.get("save_path"):
            # If we have a custom save path, extract the base output directory
            # This ensures we're using the same root directory for all outputs
            if os.path.isabs(args.get("save_path")):
                # If it's an absolute path, extract the output directory
                path_parts = args.get("save_path").split(os.sep)
                if "output" in path_parts:
                    # Find the output directory in the path
                    output_index = path_parts.index("output")
                    base_output_dir = os.path.join(*path_parts[:output_index+1])
                else:
                    # If "output" is not in the path, use the parent directory's output
                    base_output_dir = os.path.join(parent_dir, "output")
            else:
                # If it's a relative path, use the parent directory's output
                base_output_dir = os.path.join(parent_dir, "output")
        else:
            # Otherwise use the parent directory's output
            base_output_dir = os.path.join(parent_dir, "output")

        # Create the CSV output path in the train subdirectory
        csv_output_path = os.path.join(base_output_dir, "train")

        # Ensure we're using an absolute path
        if not os.path.isabs(csv_output_path):
            csv_output_path = os.path.abspath(csv_output_path)

        print(f"Using base output directory: {base_output_dir}")
        print(f"Using CSV output path: {csv_output_path}")

        os.makedirs(csv_output_path, exist_ok=True)
        print(f"Using CSV output path: {csv_output_path}")

        # CSV path for this dataset
        csv_path = os.path.join(csv_output_path, f"{args['train_dataset']}_split.csv")

        # Check if the CSV file exists and contains Windows-style paths
        recreate_csv = False
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'image_path' in df.columns and any('\\\\' in str(path) for path in df['image_path']):
                    print(f"Warning: CSV file {csv_path} contains Windows-style paths. Recreating it.")
                    os.remove(csv_path)
                    print(f"Deleted CSV file {csv_path}")
                    recreate_csv = True
            except Exception as e:
                print(f"Error checking CSV file {csv_path}: {e}")
                # If there's an error reading the CSV, delete it and recreate it
                try:
                    os.remove(csv_path)
                    print(f"Deleted CSV file {csv_path} due to error")
                except:
                    pass
                recreate_csv = True
        else:
            recreate_csv = True
            print(f"Creating new dataset split for {args['train_dataset']}")

        # Try to get configuration from automation_config.py if it exists
        try:
            import sys
            sys.path.append('..')
            from automation_config import get_config
            config = get_config()
            enable_combined_dataset = config.get("enable_combined_dataset", False)
            datasets = config.get("datasets", [args["train_dataset"]])
        except (ImportError, AttributeError):
            # Default values if config is not available
            enable_combined_dataset = False
            datasets = [args["train_dataset"]]

        # Check if we need to create a new dataset split
        if recreate_csv:
            if enable_combined_dataset and len(datasets) > 1:
                # Create a temporary combined dataset to generate the train/val split
                temp_dataset = CombinedMorphDataset(
                    dataset_names=datasets,
                    phase='train',  # Doesn't matter for split creation
                    image_size=image_size,
                    train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                    csv_path=None  # Don't load from CSV
                )

                # Save the split to CSV
                temp_dataset.save_to_csv(csv_path)

                # Now create the val dataset and save it to the same CSV
                temp_val_dataset = CombinedMorphDataset(
                    dataset_names=datasets,
                    phase='val',
                    image_size=image_size,
                    train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                    csv_path=None  # Don't load from CSV
                )
            else:
                # Create a temporary dataset to generate the train/val split
                temp_dataset = MorphDataset(
                    dataset_name=args["train_dataset"],
                    phase='train',  # Doesn't matter for split creation
                    image_size=image_size,
                    train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                    csv_path=None  # Don't load from CSV
                )

                # Save the split to CSV
                temp_dataset.save_to_csv(csv_path)

                # Now create the val dataset and save it to the same CSV
                temp_val_dataset = MorphDataset(
                    dataset_name=args["train_dataset"],
                    phase='val',
                    image_size=image_size,
                    train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                    csv_path=None  # Don't load from CSV
                )

            # Save the val split to the same CSV
            temp_val_dataset.save_to_csv(csv_path)

            print(f"Dataset split created and saved to {csv_path}")

        # Create custom datasets for training and validation from the CSV
        if enable_combined_dataset and len(datasets) > 1:
            print(f"Using combined dataset with datasets: {datasets}")
            train_dataset = CombinedMorphDataset(
                dataset_names=datasets,
                phase='train',
                image_size=image_size,
                train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                csv_path=csv_path
            )

            val_dataset = CombinedMorphDataset(
                dataset_names=datasets,
                phase='val',
                image_size=image_size,
                train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                csv_path=csv_path
            )
        else:
            train_dataset = MorphDataset(
                dataset_name=args["train_dataset"],
                phase='train',
                image_size=image_size,
                train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                csv_path=csv_path
            )

            val_dataset = MorphDataset(
                dataset_name=args["train_dataset"],
                phase='val',
                image_size=image_size,
                train_val_split=args["train_val_split"] if "train_val_split" in args else 0.8,
                csv_path=csv_path
            )
    else:
        # Use original SelfMAD dataset loading
        train_datapath = args["SMDD_path"] if args["train_dataset"] == "SMDD" else args["FF_path"]
        train_dataset = selfMAD_Dataset(phase='train', image_size=image_size, datapath=train_datapath)
        if args["saving_strategy"] == "original": # valset is always FF++
            val_dataset = selfMAD_Dataset(phase='val', image_size=image_size, datapath=args["FF_path"])

    # Create train data loader with reduced number of workers for better error handling
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=0,  # Use single process to avoid multiprocessing issues with file not found errors
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )

    # Create validation loader with reduced number of workers
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=0,  # Use single process to avoid multiprocessing issues with file not found errors
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )

    # Try to load original test datasets, but handle the case where they're not available
    try:
        original_test_datasets = default_datasets(image_size, datasets="original", config={
            "FRLL_path": args.get("FRLL_path", ""),
            "FRGC_path": args.get("FRGC_path", ""),
            "FERET_path": args.get("FERET_path", "")
        })

        # Create test data loaders with single process for better error handling
        original_test_loaders = {}
        for dataset_name, dataset_dict in original_test_datasets.items():
            original_test_loaders[dataset_name] = {}
            for split_name, dataset in dataset_dict.items():
                original_test_loaders[dataset_name][split_name] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                    num_workers=0,  # Use single process to avoid multiprocessing issues
                    pin_memory=True
                )
    except FileNotFoundError as e:
        print(f"Warning: Original test datasets not available: {e}")
        original_test_datasets = {}
        original_test_loaders = {}

    # Load custom morph test datasets
    try:
        # Create config for custom morph datasets using os.path.join for cross-platform compatibility
        datasets_dir = os.path.join("..", "datasets")
        morph_config = {
            "LMA_path": args.get("LMA_path", datasets_dir),
            "LMA_UBO_path": args.get("LMA_UBO_path", datasets_dir),
            "MIPGAN_I_path": args.get("MIPGAN_I_path", datasets_dir),
            "MIPGAN_II_path": args.get("MIPGAN_II_path", datasets_dir),
            "MorDiff_path": args.get("MorDiff_path", datasets_dir),
            "StyleGAN_path": args.get("StyleGAN_path", datasets_dir),
            "LMA_MIPGAN_I_path": args.get("LMA_MIPGAN_I_path", datasets_dir)
        }

        # Load custom morph test datasets
        custom_test_datasets = default_datasets(image_size, datasets="custom_morph", config=morph_config)

        # Create test data loaders with single process for better error handling
        custom_test_loaders = {}
        for dataset_name, dataset_dict in custom_test_datasets.items():
            custom_test_loaders[dataset_name] = {}
            for split_name, dataset in dataset_dict.items():
                custom_test_loaders[dataset_name][split_name] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                    num_workers=0,  # Use single process to avoid multiprocessing issues
                    pin_memory=True
                )

        # If we have custom datasets, use them as the main test datasets
        if custom_test_datasets:
            test_datasets = custom_test_datasets
            test_loaders = custom_test_loaders
        else:
            # Fall back to original datasets if no custom datasets are available
            test_datasets = original_test_datasets
            test_loaders = original_test_loaders

    except Exception as e:
        print(f"Warning: Custom morph test datasets not available: {e}")
        # Fall back to original datasets
        test_datasets = original_test_datasets
        test_loaders = original_test_loaders

    # If no test datasets are available at all
    if not test_datasets:
        print("No test datasets available. Training will continue without test evaluation.")
        test_datasets = {}
        test_loaders = {}

    model=Detector(model=args["model"], lr=args["lr"])
    model=model.to(device)
    n_epoch=cfg['epochs']
    # Choose scheduler based on args
    # Try to get configuration from automation_config.py if it exists
    try:
        import sys
        sys.path.append('..')
        from automation_config import get_config
        cfg = get_config()
        warmup_percentage = cfg.get("siam_vit_warmup_percentage", 0.1)
    except (ImportError, AttributeError):
        # Default value if config is not available
        warmup_percentage = 0.1

    if args.get("model") == "vit_mae_large":
        # For ViT MAE, use warm-up followed by cosine annealing
        warmup_steps = int(warmup_percentage * n_epoch * len(train_loader))  # Configurable percentage of total steps
        total_steps = n_epoch * len(train_loader)

        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            model.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Create main scheduler
        main_scheduler = CosineAnnealingLR(
            model.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )

        # Combine into sequential scheduler
        lr_scheduler = SequentialLR(
            model.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    elif args.get("scheduler", "cosine") == "linear":
        lr_scheduler = LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    else:  # default to cosine
        lr_scheduler = CosineAnnealingLR(model.optimizer, T_max=n_epoch, eta_min=0.0)

    # Create model save directory
    save_path = os.path.join(model_dir, "weights")
    logs_path = os.path.join(model_dir, "logs")

    # Ensure directories exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # Save configuration
    with open(os.path.join(model_dir, "config.txt"), "w") as f:
        f.write(str(cfg))

    # Initialize logger
    logger = log(path=logs_path, file="losses.logs")

    # Use BCE loss for binary classification
    # Note: BCELoss doesn't support label_smoothing parameter
    criterion=nn.BCELoss()
    if args["saving_strategy"] == "original":
        last_val_auc=0
        weight_dict={}
        n_weight=5
    elif args["saving_strategy"] == "testset_best":
        best_mean = None
        best_epoch = None

    # Early stopping configuration
    early_stopping_patience = args.get("early_stopping_patience", 5)
    early_stopping_monitor = args.get("early_stopping_monitor", "val_loss")
    early_stopping_counter = 0
    best_monitor_value = float('inf') if 'loss' in early_stopping_monitor else -float('inf')
    best_model_state = None

    # Initialize variables for resuming training
    start_epoch = 0

    # Check if we should resume training
    if args.get("resume_training", False):
        # Determine checkpoint path
        resume_checkpoint = args.get("resume_checkpoint", "")
        if not resume_checkpoint:
            resume_checkpoint = os.path.join(save_path, "latest.tar")

        # Check if checkpoint exists
        if os.path.exists(resume_checkpoint):
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            try:
                # Set weights_only=False to allow loading NumPy data types in PyTorch 2.6+
                checkpoint = torch.load(resume_checkpoint, weights_only=False)

                # Log checkpoint contents for debugging
                logger.info(f"Checkpoint contains: {list(checkpoint.keys())}")

                # Load model and optimizer state
                model.load_state_dict(checkpoint["model"])
                model.optimizer.load_state_dict(checkpoint["optimizer"])

                # Set starting epoch
                start_epoch = checkpoint["epoch"] + 1

                # Load scheduler state if available
                if "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None and hasattr(lr_scheduler, 'load_state_dict'):
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

                # Load other training state variables
                if args["saving_strategy"] == "testset_best":
                    best_mean = checkpoint.get("best_mean", None)
                    best_epoch = checkpoint.get("best_epoch", None)
                elif args["saving_strategy"] == "original":
                    last_val_auc = checkpoint.get("last_val_auc", 0)
                    weight_dict = checkpoint.get("weight_dict", {})

                # Load early stopping counter if available
                if "early_stopping_counter" in checkpoint:
                    early_stopping_counter = checkpoint["early_stopping_counter"]

                # Load metrics tracker if available
                if "metrics_tracker" in checkpoint:
                    metrics_tracker = checkpoint["metrics_tracker"]

                logger.info(f"Resuming from epoch {start_epoch} / {n_epoch}")
                logger.info(f"Loaded optimizer state with param groups: {len(model.optimizer.param_groups)}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.warning("Starting training from scratch.")
                start_epoch = 0
        else:
            logger.warning(f"Checkpoint not found at {resume_checkpoint}. Starting training from scratch.")

    for epoch in range(start_epoch, n_epoch):
        # TRAIN LOOP ##################################################
        np.random.seed(seed + epoch)
        train_loss=0.
        train_outputs = []
        train_targets = []
        model.train(mode=True)

        # Gradient accumulation settings for ViT MAE
        # Try to get configuration from automation_config.py if it exists
        try:
            import sys
            sys.path.append('..')
            from automation_config import get_config
            cfg = get_config()
            accumulation_steps = cfg.get("siam_vit_gradient_accumulation_steps", 4) if args.get("model") == "vit_mae_large" else 1
        except (ImportError, AttributeError):
            # Default value if config is not available
            accumulation_steps = 4 if args.get("model") == "vit_mae_large" else 1

        # Start timing for the epoch
        epoch_start_time = time.time()

        for i, data in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, n_epoch))):
            # Start timing for the batch
            batch_start_time = time.time()

            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()

            # Convert target to float for BCE loss (expects float targets)
            if args.get("model") == "vit_mae_large":
                # For binary classification with BCE loss, convert to float and reshape to match output
                target_float = target.float().unsqueeze(1)  # Shape: [batch_size, 1]

                # For ViT MAE, use gradient accumulation
                output = model(img)  # Shape: [batch_size, 1]
                loss = criterion(output, target_float) / accumulation_steps  # Scale loss
                loss.backward()

                # Add the full loss (not scaled) to train_loss for reporting
                train_loss += (loss.item() * accumulation_steps)

                # Only update weights after accumulation_steps
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    model.optimizer.step()
                    model.optimizer.zero_grad()

                    # Step the scheduler after each effective batch
                    lr_scheduler.step()
            else:
                # For other models, use SAM optimizer as before
                output = model.training_step(img, target)
                loss = criterion(output, target)
                train_loss += loss.item()

            # Collect outputs and targets for metrics
            if args.get("model") == "vit_mae_large":
                # For ViT-MAE with sigmoid activation, use the output directly
                # Squeeze to remove the extra dimension and get a flat list
                train_outputs.extend(output.squeeze(1).detach().cpu().numpy().tolist())
            else:
                # For other models using softmax
                train_outputs.extend(output.softmax(1)[:, 1].detach().cpu().numpy().tolist())
            train_targets.extend(target.detach().cpu().numpy().tolist())

            # End timing for the batch
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            # Get current learning rate
            current_lr = model.optimizer.param_groups[0]['lr']

            # Add batch metrics
            batch_metrics = {
                'epoch': epoch + 1,
                'batch': i + 1,
                'loss': loss.item() * (accumulation_steps if args.get("model") == "vit_mae_large" else 1),
                'lr': current_lr,
                'time': batch_time
            }

            # Add to metrics tracker
            metrics_tracker['batch'].append(batch_metrics)

        # For non-ViT models, step the scheduler once per epoch
        if args.get("model") != "vit_mae_large":
            lr_scheduler.step()

        # End timing for the epoch
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Calculate training metrics
        train_metrics = {
            'epoch': epoch + 1,
            'loss': train_loss / len(train_loader),
            'lr': model.optimizer.param_groups[0]['lr'],
            'time': epoch_time
        }

        # Calculate additional metrics
        if len(train_outputs) > 0 and len(train_targets) > 0:
            train_auc = calculate_auc(train_targets, train_outputs)
            train_eer = calculate_eer(train_targets, train_outputs)
            precision = calculate_precision(train_targets, train_outputs)
            recall = calculate_recall(train_targets, train_outputs)
            f1 = calculate_f1(train_targets, train_outputs)

            # Add to metrics dictionary
            train_metrics.update({
                'auc': train_auc,
                'eer': train_eer,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Add to metrics tracker
        metrics_tracker['train'].append(train_metrics)

        # Save training metrics to CSV (single file with all epochs)
        train_df = pd.DataFrame(metrics_tracker['train'])
        train_df.to_csv(os.path.join(train_dir, 'train_metrics.csv'), index=False)

        # Save batch metrics to CSV (single file with all batches)
        batch_df = pd.DataFrame(metrics_tracker['batch'])
        batch_df.to_csv(os.path.join(train_dir, 'batch_metrics.csv'), index=False)

        log_text="Epoch {}/{} | train loss: {:.4f} |".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        )
        # VAL LOOP ##################################################
        model.train(mode=False)
        val_outputs = []
        val_targets = []
        val_loss = 0.0
        np.random.seed(seed)

        # Start timing for validation
        val_start_time = time.time()

        for data in tqdm(val_loader, desc="Running validation"):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()

            # Convert target to float for BCE loss (expects float targets)
            target_float = target.float().unsqueeze(1)  # Shape: [batch_size, 1]

            with torch.no_grad():
                output=model(img)  # Shape: [batch_size, 1]
                loss=criterion(output, target_float)
                val_loss += loss.item()

            # Process outputs based on model type
            if args.get("model") == "vit_mae_large":
                # For ViT-MAE with sigmoid activation, use the output directly
                # Squeeze to remove the extra dimension and get a flat list
                val_outputs.extend(output.squeeze(1).cpu().data.numpy().tolist())
            else:
                # For other models using softmax
                val_outputs.extend(output.softmax(1)[:,1].cpu().data.numpy().tolist())
            val_targets.extend(target.cpu().data.numpy().tolist())

        # End timing for validation
        val_end_time = time.time()
        val_time = val_end_time - val_start_time

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_auc = calculate_auc(val_targets, val_outputs)
        val_eer = calculate_eer(val_targets, val_outputs)
        val_acc = sum([1 if val_outputs[i] >= 0.5 and val_targets[i] == 1 or val_outputs[i] < 0.5 and val_targets[i] == 0 else 0 for i in range(len(val_outputs))]) / len(val_outputs)

        # Calculate additional metrics
        precision = calculate_precision(val_targets, val_outputs)
        recall = calculate_recall(val_targets, val_outputs)
        f1 = calculate_f1(val_targets, val_outputs)

        # Calculate APCER metrics
        apcer_metrics = calculate_apcer_bpcer_acer(val_targets, val_outputs)
        apcer = apcer_metrics["apcer"]
        bpcer = apcer_metrics["bpcer"]
        acer = apcer_metrics["acer"]

        # Create validation metrics dictionary
        val_metrics = {
            'epoch': epoch + 1,
            'loss': val_loss,
            'auc': val_auc,
            'eer': val_eer,
            'acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'apcer': apcer,
            'bpcer': bpcer,
            'acer': acer,
            'time': val_time
        }

        # Add to metrics tracker
        metrics_tracker['val'].append(val_metrics)

        # Save validation metrics to CSV (single file with all epochs)
        val_df = pd.DataFrame(metrics_tracker['val'])
        val_df.to_csv(os.path.join(train_dir, 'val_metrics.csv'), index=False)

        # Save combined metrics (single file with all epochs)
        combined_df = pd.merge(
            pd.DataFrame(metrics_tracker['train']),
            pd.DataFrame(metrics_tracker['val']),
            on='epoch',
            suffixes=('_train', '_val')
        )
        combined_df.to_csv(os.path.join(train_dir, 'combined_metrics.csv'), index=False)

        # Generate visualizations
        visualize(
            train_metrics=metrics_tracker['train'],
            val_metrics=metrics_tracker['val'],
            output_dir=train_dir,
            model_name=model_name,
            model_dataset=args["train_dataset"],
            model_type=args["model"],
            epoch_number=epoch + 1
        )

        log_text+=" val loss: {:.4f}, val auc: {:.4f}, val eer: {:.4f}, val acc: {:.4f} |".format(
                        val_loss,
                        val_auc,
                        val_eer,
                        val_acc
        )

        # Early stopping logic
        current_monitor_value = None
        if early_stopping_monitor == "val_loss":
            current_monitor_value = val_loss
        elif early_stopping_monitor == "train_loss":
            current_monitor_value = train_loss / len(train_loader)
        elif early_stopping_monitor == "val_acc":
            current_monitor_value = val_acc

        improved = False
        if 'loss' in early_stopping_monitor:
            # For loss metrics, lower is better
            if current_monitor_value < best_monitor_value:
                best_monitor_value = current_monitor_value
                improved = True
        else:
            # For accuracy metrics, higher is better
            if current_monitor_value > best_monitor_value:
                best_monitor_value = current_monitor_value
                improved = True

        if improved:
            early_stopping_counter = 0
            best_model_state = {
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch
            }
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                log_text += f" Early stopping triggered after {epoch+1} epochs |"
                logger.info(log_text)
                # Save the best model if we're stopping early
                if best_model_state is not None:
                    save_model_path = os.path.join(save_path, f"early_stopped_best.tar")
                    torch.save(best_model_state, save_model_path)

                # Create marker file for early stopping
                marker_info = {
                    "early_stopped": True,
                    "epochs_completed": epoch + 1,
                    "total_epochs": n_epoch,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                with open(os.path.join(save_path, "training_complete.marker"), "w") as f:
                    json.dump(marker_info, f, cls=NumpyEncoder)

                break
        # TEST LOOP ###################################################
        if test_loaders:
            model.train(mode=False)

            # Get model parameters
            model_params = {
                "num_parameters": sum(p.numel() for p in model.parameters())
            }

            # Get training parameters
            training_params = {
                "learning_rate": model.optimizer.param_groups[0]['lr'],
                "batch_size": args["batch_size"],
                "epochs": n_epoch,
                "current_epoch": epoch + 1
            }

            # Use the new evaluation module
            results_dataset, targets_outputs_dict, class_separated_dict = evaluate(
                model=model,
                test_loaders=test_loaders,
                device=device,
                output_dir=os.path.join(eval_dir, f"epoch_{epoch+1}"),
                model_name=f"{model_name}_epoch_{epoch+1}",
                model_dataset=args["train_dataset"],
                model_type=args["model"],
                epoch_number=epoch + 1,
                verbose=True,
                model_params=model_params,
                training_params=training_params
            )

            # Determine if this is the best model
            is_best_model = False
            if args["saving_strategy"] == "testset_best":
                if test_loaders:
                    # If test datasets are available, use them to determine the best model
                    if best_mean is None or results_dataset['mean']['eer'] < best_mean:
                        is_best_model = True
            elif args["saving_strategy"] == "original":
                if val_auc >= last_val_auc:
                    is_best_model = True

            # Use the visualization module to generate plots (only for the best model)
            test_metrics = visualize(
                results=results_dataset,
                targets_outputs_dict=targets_outputs_dict,
                class_separated_dict=class_separated_dict,
                output_dir=os.path.join(eval_dir, f"epoch_{epoch+1}"),
                model_name=f"{model_name}_epoch_{epoch+1}",
                model_dataset=args["train_dataset"],
                model_type=args["model"],
                epoch_number=epoch + 1,
                # Only generate plots if this is the best model
                generate_plots=is_best_model
            )

            # Create a best model marker if this is the best model
            if is_best_model:
                # Create a marker file
                with open(os.path.join(eval_dir, "best_model.txt"), "w") as f:
                    f.write(f"Best model: epoch {epoch+1}\n")
                    f.write(f"Test metrics: {json.dumps(results_dataset['mean'], indent=4, cls=NumpyEncoder)}\n")

                # Create a dedicated directory for the best model's plots
                best_model_dir = os.path.join(eval_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)

                # Copy the plots from the current epoch to the best_model directory
                source_dir = os.path.join(eval_dir, f"epoch_{epoch+1}", "plots")
                if os.path.exists(source_dir):
                    import shutil
                    # Remove existing plots in best_model_dir
                    if os.path.exists(os.path.join(best_model_dir, "plots")):
                        shutil.rmtree(os.path.join(best_model_dir, "plots"))
                    # Copy plots
                    shutil.copytree(source_dir, os.path.join(best_model_dir, "plots"))

            # Create a list to store all dataset metrics for this epoch
            epoch_metrics = []

            # Add metrics for each dataset
            for dataset in results_dataset:
                if dataset != 'mean':  # Skip mean, it will be handled separately
                    # Create a row for this dataset
                    dataset_row = {
                        'epoch': epoch + 1,
                        'dataset': dataset,
                        'auc': results_dataset[dataset]['auc'],
                        'eer': results_dataset[dataset]['eer'],
                        'accuracy': results_dataset[dataset]['accuracy']
                    }

                    # Add to the list of metrics for this epoch
                    epoch_metrics.append(dataset_row)

                    # Add to log text
                    log_text += f" {dataset}: auc: {results_dataset[dataset]['auc']:.4f}, eer: {results_dataset[dataset]['eer']:.4f} |"

            # Add mean metrics if available
            if 'mean' in results_dataset:
                # Create a row for mean metrics
                mean_row = {
                    'epoch': epoch + 1,
                    'dataset': 'mean',
                    'auc': results_dataset['mean']['auc'],
                    'eer': results_dataset['mean']['eer'],
                    'accuracy': results_dataset['mean']['accuracy']
                }

                # Add to the list of metrics for this epoch
                epoch_metrics.append(mean_row)

                # Add to log text
                log_text += f" Mean: auc: {results_dataset['mean']['auc']:.4f}, eer: {results_dataset['mean']['eer']:.4f} |"

            # Initialize test metrics tracker if it doesn't exist
            if 'test_metrics' not in metrics_tracker:
                metrics_tracker['test_metrics'] = []

            # Add the metrics to the tracker
            metrics_tracker['test_metrics'].extend(epoch_metrics)

            # Save to CSV (single file with all epochs)
            test_df = pd.DataFrame(metrics_tracker['test_metrics'])
            test_df.to_csv(os.path.join(train_dir, 'test_metrics.csv'), index=False)

            # Use the results for model saving
            results_original_dataset = results_dataset
        else:
            results_original_dataset = {'mean': {'eer': float('inf')}}
            log_text += " No test datasets available |"
        # for dataset in results_mordiff_dataset:
        #     log_text += f" {dataset}: auc: {results_mordiff_dataset[dataset]['auc']:.4f}, eer: {results_mordiff_dataset[dataset]['eer']:.4f}"
        # results_mordiff_dataset = evaluate(model, test_loaders_mordiff, device, calculate_means=False)
        # SAVE MODEL ###################################################
        if args["saving_strategy"] == "original":
            if len(weight_dict)<n_weight:
                save_model_path=os.path.join(save_path,"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                weight_dict[save_model_path]=val_auc
                torch.save({
                        "model":model.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])

            elif val_auc>=last_val_auc:
                save_model_path=os.path.join(save_path,"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                for k in weight_dict:
                    if weight_dict[k]==last_val_auc:
                        del weight_dict[k]
                        os.remove(k)
                        weight_dict[save_model_path]=val_auc
                        break
                torch.save({
                        "model":model.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])
        elif args["saving_strategy"] == "testset_best":
            if test_loaders:
                # If test datasets are available, use them to determine the best model
                if best_mean is None or results_original_dataset['mean']['eer'] < best_mean:
                    best_mean = results_original_dataset['mean']['eer']
                    # remove previous best model
                    if best_epoch is not None and os.path.exists(os.path.join(save_path, "epoch_{}.tar".format(best_epoch))):
                        os.remove(os.path.join(save_path, "epoch_{}.tar".format(best_epoch)))
                    best_epoch = epoch + 1
                    save_model_path=os.path.join(save_path,"epoch_{}.tar".format(best_epoch))
                    torch.save({
                            "model":model.state_dict(),
                            "optimizer":model.optimizer.state_dict(),
                            "epoch":epoch
                        },save_model_path)
            else:
                # If no test datasets are available, save based on validation metrics
                if best_mean is None or val_eer < best_mean:
                    best_mean = val_eer
                    # remove previous best model
                    if best_epoch is not None and os.path.exists(os.path.join(save_path, "epoch_{}.tar".format(best_epoch))):
                        os.remove(os.path.join(save_path, "epoch_{}.tar".format(best_epoch)))
                    best_epoch = epoch + 1
                    save_model_path=os.path.join(save_path,"epoch_{}.tar".format(best_epoch))
                    torch.save({
                            "model":model.state_dict(),
                            "optimizer":model.optimizer.state_dict(),
                            "epoch":epoch
                        },save_model_path)

        # Always save the latest checkpoint (for resuming training)
        latest_checkpoint_path = os.path.join(save_path, "latest.tar")
        torch.save({
            "model": model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
            "epoch": epoch,
            "lr_scheduler": lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
            "best_mean": best_mean if args["saving_strategy"] == "testset_best" else None,
            "best_epoch": best_epoch if args["saving_strategy"] == "testset_best" else None,
            "last_val_auc": last_val_auc if args["saving_strategy"] == "original" else None,
            "weight_dict": weight_dict if args["saving_strategy"] == "original" else None,
            "early_stopping_counter": early_stopping_counter,
            "metrics_tracker": metrics_tracker
        }, latest_checkpoint_path)

    # Create marker file for normal completion (only if we didn't early stop)
    if early_stopping_counter < early_stopping_patience:
        marker_info = {
            "early_stopped": False,
            "epochs_completed": n_epoch,
            "total_epochs": n_epoch,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        with open(os.path.join(save_path, "training_complete.marker"), "w") as f:
            json.dump(marker_info, f, cls=NumpyEncoder)
        logger.info(log_text)

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # required
    parser.add_argument('-n',dest='session_name', type=str, required=True)
    # specified in train_config.json
    parser.add_argument('-m',dest='model',type=str,required=False)
    parser.add_argument('-b', dest='batch_size', type=int, required=False)
    parser.add_argument('-e', dest='epochs', type=int, required=False)
    parser.add_argument('-v', dest='saving_strategy', type=str, required=False)
    parser.add_argument('-t', dest='train_dataset', type=str, required=False)
    parser.add_argument('-s', dest='save_path', type=str, required=False)
    parser.add_argument('-lr', dest='lr', type=float, required=False)
    parser.add_argument('-FRLL_path', type=str, required=False)
    parser.add_argument('-FRGC_path', type=str, required=False)
    # Resume training parameters
    parser.add_argument('-resume', dest='resume_training', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('-resume_checkpoint', dest='resume_checkpoint', type=str, default='', help='Path to checkpoint to resume from (if empty, will use latest.tar)')
    parser.add_argument('-FERET_path', type=str, required=False)
    # parser.add_argument('-MorDIFF_f_path', type=str, required=False)
    # parser.add_argument('-MorDIFF_bf_path', type=str, required=False)
    parser.add_argument('-SMDD_path', type=str, required=False)
    parser.add_argument('-FF_path', type=str, required=False)

    # Custom morph dataset parameters
    parser.add_argument('-LMA_path', type=str, required=False)
    parser.add_argument('-LMA_UBO_path', type=str, required=False)
    parser.add_argument('-MIPGAN_I_path', type=str, required=False)
    parser.add_argument('-MIPGAN_II_path', type=str, required=False)
    parser.add_argument('-MorDiff_path', type=str, required=False)
    parser.add_argument('-StyleGAN_path', type=str, required=False)
    parser.add_argument('-train_val_split', type=float, required=False)
    parser.add_argument('-csv_output_path', type=str, required=False)
    parser.add_argument('-early_stopping_patience', type=int, required=False, help='Patience for early stopping')
    parser.add_argument('-early_stopping_monitor', type=str, required=False, help='Metric to monitor for early stopping (val_loss, train_loss, val_acc)')
    parser.add_argument('-scheduler', type=str, required=False, help='Learning rate scheduler (cosine or linear)')
    args=parser.parse_args()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base training configuration
    try:
        train_config_path = os.path.join(script_dir, "configs", "train_config.json")
        print(f"Loading train config from: {train_config_path}")
        train_config = json.load(open(train_config_path))
    except FileNotFoundError:
        print(f"Warning: train_config.json not found at {train_config_path}. Using default values.")
        train_config = {
            "model": "vit_mae_large",
            "batch_size": 128,
            "epochs": 1,
            "saving_strategy": "testset_best",
            "train_dataset": "LMA",
            "save_path": "",
            "lr": 0.1
        }

    # Override with command line arguments
    for key in vars(args):
        if vars(args)[key] is not None:
            train_config[key] = vars(args)[key]

    # Load original data configuration
    try:
        data_config_path = os.path.join(script_dir, "configs", "data_config.json")
        print(f"Loading data config from: {data_config_path}")
        data_config = json.load(open(data_config_path))
    except FileNotFoundError:
        print(f"Warning: data_config.json not found at {data_config_path}. Using default values.")
        data_config = {
            "FRLL_path": "",
            "FRGC_path": "",
            "FERET_path": "",
            "FF_path": "",
            "SMDD_path": ""
        }

    for key in data_config:
        if key in vars(args) and vars(args)[key] is None:
            train_config[key] = data_config[key]

    # Load morph dataset configuration if needed
    if train_config.get("train_dataset") in ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN"]:
        try:
            morph_config_path = os.path.join(script_dir, "configs", "morph_config.json")
            print(f"Loading morph config from: {morph_config_path}")
            morph_config = json.load(open(morph_config_path))
            for key in morph_config:
                if key in vars(args) and vars(args)[key] is None and key not in train_config:
                    train_config[key] = morph_config[key]
        except FileNotFoundError:
            print(f"Warning: morph_config.json not found at {morph_config_path}. Using default values.")

    main(train_config)
