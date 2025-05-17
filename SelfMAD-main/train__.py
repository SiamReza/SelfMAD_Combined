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
from utils.dataset import MorphDataset, TestMorphDataset, CombinedMorphDataset
from utils.scheduler import LinearDecayLR
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
        prefix (str): The prefix to match (e.g., "LMA_main_")

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

    assert args["model"] in ["efficientnet-b4", "efficientnet-b7", "swin", "resnet", "hrnet_w18", "hrnet_w32", "hrnet_w44", "hrnet_w64"]
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
        base_dir = os.path.join(parent_dir, "output", "main")
        prefix = f"{args['train_dataset']}_main_"
        serial_number = get_next_serial_number(base_dir, prefix)
        model_name = f"{args['train_dataset']}_main_{serial_number}"
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
        # Use the provided save path as the model directory
        model_dir = args["save_path"]

        # Check if the path is inside a repository output directory
        if is_repository_output_dir(model_dir):
            # Convert to absolute path relative to the parent directory
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(parent_dir, "output", "main", model_name)
            model_dir = os.path.join(output_dir, args["model"])
            print(f"Warning: Redirecting output to root directory: {model_dir}")

        print(f"Using custom save path: {model_dir}")
    else:
        # Use a path relative to the parent directory (outside the repository)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(parent_dir, "output", "main", model_name)
        model_dir = os.path.join(output_dir, args["model"])
        print(f"Using default directory structure: {model_dir}")

    # Ensure we're using an absolute path
    if not os.path.isabs(model_dir):
        model_dir = os.path.abspath(model_dir)

    print(f"Using model directory: {model_dir}")

    # Create the model directory
    os.makedirs(model_dir, exist_ok=True)

    # Create all necessary subdirectories according to the required structure
    model_weights_dir = os.path.join(model_dir, "model")
    eval_dir = os.path.join(model_dir, "eval")
    train_dir = os.path.join(model_dir, "train")

    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    print(f"Created subdirectories: {model_weights_dir}, {eval_dir}, {train_dir}")

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
        "image_size": 384 if "hrnet" in args["model"] else 380,
        "saving_strategy": args["saving_strategy"],
    }

    device = torch.device('cuda')

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']

    # Check if using original datasets or custom morph datasets
    custom_morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN", "LMA_MIPGAN_I"]

    if args["train_dataset"] in custom_morph_datasets:
        # Use custom morph dataset
        dataset_name = args["train_dataset"]

        # Create output directory for CSV files
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
        csv_output_dir = os.path.join(base_output_dir, "train")

        # Ensure we're using an absolute path
        if not os.path.isabs(csv_output_dir):
            csv_output_dir = os.path.abspath(csv_output_dir)

        print(f"Using base output directory: {base_output_dir}")
        print(f"Using CSV output directory: {csv_output_dir}")

        os.makedirs(csv_output_dir, exist_ok=True)
        csv_path = os.path.join(csv_output_dir, f"{dataset_name}_split.csv")

        # Check if the CSV file exists and contains Windows-style paths
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'image_path' in df.columns and any('\\\\' in str(path) for path in df['image_path']):
                    print(f"Warning: CSV file {csv_path} contains Windows-style paths. Recreating it.")
                    os.remove(csv_path)
                    print(f"Deleted CSV file {csv_path}")
            except Exception as e:
                print(f"Error checking CSV file {csv_path}: {e}")
                # If there's an error reading the CSV, delete it and recreate it
                try:
                    os.remove(csv_path)
                    print(f"Deleted CSV file {csv_path} due to error")
                except:
                    pass

        # Try to get configuration from automation_config.py if it exists
        try:
            import sys
            sys.path.append('..')
            from automation_config import get_config
            config = get_config()
            enable_combined_dataset = config.get("enable_combined_dataset", False)
            datasets = config.get("datasets", [dataset_name])
        except (ImportError, AttributeError):
            # Default values if config is not available
            enable_combined_dataset = False
            datasets = [dataset_name]

        # Create train dataset
        if enable_combined_dataset and len(datasets) > 1:
            print(f"Using combined dataset with datasets: {datasets}")
            train_dataset = CombinedMorphDataset(
                dataset_names=datasets,
                phase='train',
                image_size=image_size,
                train_val_split=args.get("train_val_split", 0.8),
                csv_path=csv_path
            )

            # Create validation dataset
            if args["saving_strategy"] == "original":
                # For original strategy, use FF++ as validation
                val_dataset = selfMAD_Dataset(phase='val', image_size=image_size, datapath=args["FF_path"])
            else:
                # For testset_best strategy, use combined morph validation set
                val_dataset = CombinedMorphDataset(
                    dataset_names=datasets,
                    phase='val',
                    image_size=image_size,
                    train_val_split=args.get("train_val_split", 0.8),
                    csv_path=csv_path
                )
        else:
            train_dataset = MorphDataset(
                dataset_name=dataset_name,
                phase='train',
                image_size=image_size,
                train_val_split=args.get("train_val_split", 0.8),
                csv_path=csv_path
            )

            # Create validation dataset
            if args["saving_strategy"] == "original":
                # For original strategy, use FF++ as validation
                val_dataset = selfMAD_Dataset(phase='val', image_size=image_size, datapath=args["FF_path"])
            else:
                # For testset_best strategy, use custom morph validation set
                val_dataset = MorphDataset(
                    dataset_name=dataset_name,
                    phase='val',
                    image_size=image_size,
                    train_val_split=args.get("train_val_split", 0.8),
                    csv_path=csv_path
                )
    else:
        # Use original datasets (FF++ or SMDD)
        train_datapath = args["SMDD_path"] if args["train_dataset"] == "SMDD" else args["FF_path"]
        train_dataset = selfMAD_Dataset(phase='train', image_size=image_size, datapath=train_datapath)
        # For both strategies, use FF++ as validation
        val_dataset = selfMAD_Dataset(phase='val', image_size=image_size, datapath=args["FF_path"])

    # Create train data loader with reduced number of workers for better error handling
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size//2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,  # Use single process to avoid multiprocessing issues with file not found errors
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn
    )

    # Create validation data loader with reduced number of workers
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,  # Use single process to avoid multiprocessing issues with file not found errors
        pin_memory=True,
        worker_init_fn=val_dataset.worker_init_fn
    )

    # Skip loading original test datasets as they are not available
    print("Skipping evaluation on original datasets (FRGC, FERET, FRLL)...")
    test_datasets = {}
    test_loaders = {}

    # Load custom morph test datasets if available
    custom_test_datasets = {}
    custom_test_loaders = {}

    for dataset_name in custom_morph_datasets:
        # Create test dataset directly, TestMorphDataset will find the correct path
        # TestMorphDataset has its own path determination logic that checks multiple possible locations
        test_dataset = TestMorphDataset(dataset_name=dataset_name, image_size=image_size)

        # Check if the dataset has any samples
        if len(test_dataset) > 0:
            custom_test_datasets[dataset_name] = {"test": test_dataset}
            print(f"Successfully loaded test dataset for {dataset_name} with {len(test_dataset)} samples")
        else:
            print(f"Warning: No test samples found for {dataset_name}. Skipping this dataset.")

    if custom_test_datasets:
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
    # test_datasets_mordiff = default_datasets(image_size, datasets="MorDIFF", config={
    #     "MorDIFF_f_path": args["MorDIFF_f_path"],
    #     "MorDIFF_bf_path": args["MorDIFF_bf_path"]
    # })
    # test_loaders_mordiff = prep_dataloaders(test_datasets_mordiff, batch_size)

    model=Detector(model=args["model"], lr=args["lr"])
    model=model.to('cuda')
    n_epoch=cfg['epochs']
    lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))

    # Set save_path to the model directory for model saving
    save_path = model_weights_dir

    # Print the paths for debugging
    print(f"Model directory: {model_dir}")
    print(f"Model weights directory: {model_weights_dir}")
    print(f"Eval directory: {eval_dir}")
    print(f"Train directory: {train_dir}")

    # Save configuration
    with open(os.path.join(model_dir, "config.txt"), "w") as f:
        f.write(str(cfg))

    # Initialize logger
    logger = log(path=model_weights_dir, file="losses.logs")

    criterion=nn.CrossEntropyLoss()
    if args["saving_strategy"] == "original":
        last_val_auc=0
        weight_dict={}
        n_weight=5
    elif args["saving_strategy"] == "testset_best":
        best_mean = None
        best_epoch = None

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

        # Start timing for the epoch
        epoch_start_time = time.time()

        for batch_idx, data in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, n_epoch))):
            # Start timing for the batch
            batch_start_time = time.time()

            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            train_loss+=loss.item()

            # Collect outputs and targets for metrics
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
                'batch': batch_idx + 1,
                'loss': loss.item(),
                'lr': current_lr,
                'time': batch_time
            }

            # Add to metrics tracker
            metrics_tracker['batch'].append(batch_metrics)

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
            with torch.no_grad():
                output=model(img)
                loss=criterion(output,target)
                val_loss += loss.item()
            val_outputs.extend(output.softmax(1)[:,1].cpu().data.numpy().tolist())
            val_targets.extend(target.cpu().data.numpy().tolist())

        # End timing for validation
        val_end_time = time.time()
        val_time = val_end_time - val_start_time

        # Calculate validation metrics
        val_auc = calculate_auc(val_targets, val_outputs)
        val_eer = calculate_eer(val_targets, val_outputs)

        # Check if validation loader is empty
        if len(val_loader) > 0:
            val_loss = val_loss / len(val_loader)
        else:
            val_loss = 0.0
            print("Warning: Validation loader is empty. Setting validation loss to 0.")

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
            model_name=model_name
        )

        log_text+=" val loss: {:.4f}, val auc: {:.4f}, val eer: {:.4f} |".format(
                        val_loss,
                        val_auc,
                        val_eer
        )
        # TEST LOOP ###################################################
        model.train(mode=False)

        # Skip evaluation on original datasets as they are not available
        results_original_dataset = {}

        # Evaluate on custom morph datasets if available
        if custom_test_datasets:
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

            # Use the new evaluation module
            try:
                # Handle different return value formats from evaluate function
                eval_result = evaluate(
                    model=model,
                    test_loaders=custom_test_loaders,
                    device=device,
                    output_dir=os.path.join(eval_dir, f"epoch_{epoch+1}"),
                    model_name=f"{model_name}_epoch_{epoch+1}",
                    verbose=True,
                    classification_threshold=0.5,  # Default threshold
                    apcer_thresholds=[0.05, 0.10, 0.20]  # Default APCER thresholds
                )

                # Check the number of returned values and handle accordingly
                if isinstance(eval_result, tuple):
                    if len(eval_result) >= 2:
                        results_custom_dataset = eval_result[0]
                        targets_outputs_dict = eval_result[1]
                    else:
                        results_custom_dataset = eval_result[0]
                        targets_outputs_dict = {}
                else:
                    # If only one value is returned
                    results_custom_dataset = eval_result
                    targets_outputs_dict = {}

                # Ensure results_custom_dataset has a 'mean' key with at least 'eer' and 'auc' subkeys
                if 'mean' not in results_custom_dataset:
                    print("Warning: 'mean' key not found in evaluation results. Creating default mean metrics.")
                    # Calculate mean metrics from available datasets
                    if results_custom_dataset:
                        # Get all metrics except 'dataset' from the first dataset
                        first_dataset = next(iter(results_custom_dataset.values()))
                        mean_metrics = {k: 0.0 for k in first_dataset if k != 'dataset'}

                        # Calculate mean for each metric
                        for dataset, metrics in results_custom_dataset.items():
                            for k in mean_metrics:
                                if k in metrics:
                                    mean_metrics[k] += metrics[k]

                        # Divide by number of datasets
                        for k in mean_metrics:
                            mean_metrics[k] /= len(results_custom_dataset)

                        # Add to results
                        results_custom_dataset['mean'] = mean_metrics
                    else:
                        # No datasets available, create default mean metrics
                        results_custom_dataset['mean'] = {
                            'eer': 0.5,
                            'auc': 0.5,
                            'accuracy': 0.0
                        }
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
                results_custom_dataset = {'mean': {'eer': 0.5, 'auc': 0.5, 'accuracy': 0.0}}
                targets_outputs_dict = {}

            # Determine if this is the best model
            is_best_model = False
            if args["saving_strategy"] == "testset_best":
                # Check if results_custom_dataset has 'mean' key and 'eer' subkey
                if 'mean' in results_custom_dataset and 'eer' in results_custom_dataset['mean']:
                    current_eer = results_custom_dataset['mean']['eer']
                    if best_mean is None or current_eer < best_mean:
                        is_best_model = True
            elif args["saving_strategy"] == "original":
                if val_auc >= last_val_auc:
                    is_best_model = True

            # Use the visualization module to generate plots (only for the best model)
            visualize(
                results=results_custom_dataset,
                targets_outputs_dict=targets_outputs_dict,
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
                    # Convert NumPy types to Python native types before serializing
                    mean_metrics = {}
                    for key, value in results_custom_dataset['mean'].items():
                        if isinstance(value, np.float32) or isinstance(value, np.float64):
                            mean_metrics[key] = float(value)
                        elif isinstance(value, np.int32) or isinstance(value, np.int64):
                            mean_metrics[key] = int(value)
                        elif isinstance(value, np.ndarray):
                            mean_metrics[key] = value.tolist()
                        else:
                            mean_metrics[key] = value
                    f.write(f"Test metrics: {json.dumps(mean_metrics, indent=4)}\n")

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

            print(f"Evaluation results saved to {os.path.join(eval_dir, f'epoch_{epoch+1}')}")

            # Create a list to store all dataset metrics for this epoch
            epoch_metrics = []

            # Add metrics for each dataset
            for dataset in results_custom_dataset:
                if dataset != 'mean':  # Skip mean, it will be handled separately
                    # Create a row for this dataset
                    dataset_row = {
                        'epoch': epoch + 1,
                        'dataset': dataset,
                        'auc': results_custom_dataset[dataset]['auc'],
                        'eer': results_custom_dataset[dataset]['eer'],
                        'accuracy': results_custom_dataset[dataset]['accuracy']
                    }

                    # Add to the list of metrics for this epoch
                    epoch_metrics.append(dataset_row)

                    # Add to log text
                    log_text += f" {dataset}: auc: {results_custom_dataset[dataset]['auc']:.4f}, eer: {results_custom_dataset[dataset]['eer']:.4f} |"

            # Add mean metrics if available
            if 'mean' in results_custom_dataset:
                # Create a row for mean metrics
                mean_row = {
                    'epoch': epoch + 1,
                    'dataset': 'mean',
                    'auc': results_custom_dataset['mean']['auc'],
                    'eer': results_custom_dataset['mean']['eer'],
                    'accuracy': results_custom_dataset['mean']['accuracy']
                }

                # Add to the list of metrics for this epoch
                epoch_metrics.append(mean_row)

                # Add to log text
                log_text += f" mean: auc: {results_custom_dataset['mean']['auc']:.4f}, eer: {results_custom_dataset['mean']['eer']:.4f} |"

            # Initialize test metrics tracker if it doesn't exist
            if 'test_metrics' not in metrics_tracker:
                metrics_tracker['test_metrics'] = []

            # Add the metrics to the tracker
            metrics_tracker['test_metrics'].extend(epoch_metrics)

            # Save to CSV (single file with all epochs)
            test_df = pd.DataFrame(metrics_tracker['test_metrics'])
            test_df.to_csv(os.path.join(train_dir, 'test_metrics.csv'), index=False)

            # Add custom dataset results to original results for saving strategy
            if args["saving_strategy"] == "testset_best":
                if 'mean' in results_custom_dataset:
                    results_original_dataset = {'mean': results_custom_dataset['mean']}
                else:
                    # Add default values if 'mean' key is not found
                    print("Warning: 'mean' key not found in results_custom_dataset. Using default values.")
                    results_original_dataset = {'mean': {'eer': 0.5, 'auc': 0.5}}
        else:
            # No custom test datasets available, but we still need to ensure results_original_dataset has the correct structure
            if args["saving_strategy"] == "testset_best":
                print("Warning: No custom test datasets available. Using default values for results_original_dataset.")
                results_original_dataset = {'mean': {'eer': 0.5, 'auc': 0.5}}
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
            # Check if results_original_dataset has 'mean' key and 'eer' subkey
            if 'mean' in results_original_dataset and 'eer' in results_original_dataset['mean']:
                current_eer = results_original_dataset['mean']['eer']
                if best_mean is None or current_eer < best_mean:
                    best_mean = current_eer
                    # remove previous best model
                    if best_epoch is not None and os.path.exists(os.path.join(save_path, "epoch_{}.tar".format(best_epoch))):
                        os.remove(os.path.join(save_path, "epoch_{}.tar".format(best_epoch)))
                    best_epoch = epoch + 1
                    save_model_path=os.path.join(save_path, "epoch_{}.tar".format(best_epoch))
                    torch.save({
                            "model":model.state_dict(),
                            "optimizer":model.optimizer.state_dict(),
                            "epoch":epoch
                        },save_model_path)
            else:
                print("Warning: 'mean' key or 'eer' subkey not found in results_original_dataset. Skipping best model saving for this epoch.")

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
            "metrics_tracker": metrics_tracker
        }, latest_checkpoint_path)

        logger.info(log_text)

    # Create marker file for training completion
    marker_info = {
        "epochs_completed": n_epoch,
        "total_epochs": n_epoch,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(os.path.join(model_dir, "training_complete.marker"), "w") as f:
        # Use standard Python types to ensure JSON serialization works
        json.dump(marker_info, f)

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
    parser.add_argument('-LMA_path', type=str, required=False)
    parser.add_argument('-LMA_UBO_path', type=str, required=False)
    parser.add_argument('-MIPGAN_I_path', type=str, required=False)
    parser.add_argument('-MIPGAN_II_path', type=str, required=False)
    parser.add_argument('-MorDiff_path', type=str, required=False)
    parser.add_argument('-StyleGAN_path', type=str, required=False)
    parser.add_argument('-train_val_split', type=float, required=False)
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
            "model": "hrnet_w18",
            "batch_size": 64,
            "epochs": 1,
            "saving_strategy": "testset_best",
            "train_dataset": "LMA",
            "save_path": "",
            "lr": 0.0005
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

    main(train_config)
