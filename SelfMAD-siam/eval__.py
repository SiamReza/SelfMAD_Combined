import torch
import numpy as np
import os
from tqdm import tqdm
from utils.model import Detector
import numpy as np
from utils.dataset import PartialMorphDataset
from utils.custom_dataset import TestMorphDataset
# from utils.dataset import MorDIFF
from utils.metrics import calculate_eer, calculate_auc
import json
import argparse
from datetime import datetime
import sys
import os
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_1 import calculate_auc, calculate_eer, calculate_apcer_bpcer_acer
from evaluation_2 import evaluate
from visualization_1 import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from visualization_2 import plot_det_curve, plot_score_distributions, plot_threshold_analysis
from visualization_3 import visualize

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main(eval_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state_path = eval_config["model_path"]

    # Normalize path (replace backslashes with forward slashes)
    model_state_path = os.path.normpath(model_state_path).replace('\\', '/')

    # Check if we should evaluate the best model
    if eval_config.get("evaluate_best", False):
        # Find the best model based on the marker file
        model_dir = os.path.dirname(model_state_path)
        best_model_marker = os.path.join(model_dir, "eval", "best_model.txt")
        if os.path.exists(best_model_marker):
            with open(best_model_marker, 'r') as f:
                marker_content = f.read()
                # Extract epoch number from marker file
                import re
                match = re.search(r"Best model: epoch (\d+)", marker_content)
                if match:
                    best_epoch = int(match.group(1))
                    # Update model_state_path to use the best model
                    model_state_path = os.path.join(model_dir, f"epoch_{best_epoch}.tar")
                    print(f"Using best model from epoch {best_epoch}")
        else:
            print("Best model marker not found. Using the specified model.")

    try:
        # Check if model file exists
        if not os.path.exists(model_state_path):
            # Try relative to current directory
            alt_path = os.path.join(".", model_state_path)
            alt_path = os.path.normpath(alt_path).replace('\\', '/')
            if os.path.exists(alt_path):
                model_state_path = alt_path
                print(f"Using model file at: {model_state_path}")
            else:
                print(f"Warning: Model file not found: {model_state_path} or {alt_path}")
                raise FileNotFoundError(f"Model file not found: {model_state_path}")

        # Load model state first to detect architecture
        try:
            # Set weights_only=False to allow loading NumPy data types in PyTorch 2.6+
            model_state = torch.load(model_state_path, map_location=device, weights_only=False)

            # Detect model architecture from state dictionary
            if 'model' in model_state:
                model_state_dict = model_state['model']
            else:
                model_state_dict = model_state

            # Check for EfficientNet architecture
            if any(key.startswith('net._conv_stem') for key in model_state_dict):
                print(f"Detected EfficientNet architecture in model state dictionary")
                eval_config["model_type"] = "efficientnet-b4"
            # Check for EfficientNet-B7 (has more blocks than B4)
            elif any(key.startswith('net._blocks.30') for key in model_state_dict):
                print(f"Detected EfficientNet-B7 architecture in model state dictionary")
                eval_config["model_type"] = "efficientnet-b7"
            # Check for HRNet architecture
            elif any(key.startswith('net.conv1') for key in model_state_dict):
                print(f"Detected HRNet architecture in model state dictionary")
                eval_config["model_type"] = "hrnet_w18"
            # Check for Swin architecture
            elif any(key.startswith('net.features') for key in model_state_dict):
                print(f"Detected Swin architecture in model state dictionary")
                eval_config["model_type"] = "swin"
            # Check for ResNet architecture
            elif any(key.startswith('net.layer1') for key in model_state_dict):
                print(f"Detected ResNet architecture in model state dictionary")
                eval_config["model_type"] = "resnet"
            # Check for ViT-MAE architecture
            elif any(key.startswith('net.encoder') for key in model_state_dict):
                print(f"Detected ViT-MAE architecture in model state dictionary")
                eval_config["model_type"] = "vit_mae_large"

            print(f"Using model_type: {eval_config['model_type']}")

            # Create model with detected or specified architecture
            model = Detector(model=eval_config["model_type"])

            # Load the state dictionary
            if 'model' in model_state:
                model.load_state_dict(model_state['model'])
            else:
                # Try loading directly if 'model' key is not present
                model.load_state_dict(model_state)
        except Exception as e:
            print(f"Error loading model state: {str(e)}")
            raise

        model.train(mode=False)
        model.to(device)
        print(f"Model loaded successfully from {model_state_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    image_size = 224 if eval_config["model_type"] == "vit_mae_large" else (384 if "hrnet" in eval_config["model_type"] else 380)
    batch_size = 32

    # Extract model name and create output directory structure
    model_path_parts = os.path.basename(model_state_path).split('.')
    model_name = model_path_parts[0]

    # Remove epoch number if present to avoid epoch-specific directories
    if model_name.startswith("epoch_"):
        # Extract the base model name from the directory structure
        model_dir = os.path.dirname(model_state_path)
        base_dir = os.path.dirname(model_dir)
        model_name = os.path.basename(base_dir)

    repo_name = "siam"  # This is the SelfMAD-siam repository

    # Create centralized output directory structure
    # If output directory is provided via command line, use that instead
    if args.output_dir:
        eval_dir = args.output_dir
    else:
        output_dir = os.path.join("output", repo_name, model_name)
        eval_dir = os.path.join(output_dir, "eval")

    os.makedirs(eval_dir, exist_ok=True)

    # Skip original datasets evaluation since they're not available
    print("Skipping evaluation on original datasets (FRGC, FERET, FRLL) as they are not available...")

    # Extract model information from the model name
    model_dataset = eval_config.get("train_dataset", "unknown")
    model_type = eval_config.get("model_type", "unknown")

    # Try to extract dataset from model path if not provided
    if model_dataset == "unknown":
        # Try to extract from model path
        if "_siam_" in model_state_path:
            try:
                model_dataset = os.path.basename(model_state_path).split("_siam_")[0]
            except:
                pass

        # If still unknown, try to extract from directory structure
        if model_dataset == "unknown":
            try:
                # Get the parent directory of the model file
                model_dir = os.path.dirname(model_state_path)
                # Get the parent directory of the model directory
                parent_dir = os.path.dirname(model_dir)
                # Get the basename of the parent directory
                parent_name = os.path.basename(parent_dir)
                # Extract dataset name from parent directory name
                if "_siam_" in parent_name:
                    model_dataset = parent_name.split("_siam_")[0]
            except:
                pass

    print(f"Model dataset identified as: {model_dataset}")
    print(f"Model type identified as: {model_type}")

    # Initialize epoch_number to None before trying to extract it
    epoch_number = None
    if "epoch_" in model_state_path:
        try:
            epoch_number = int(os.path.basename(model_state_path).split("epoch_")[1].split(".")[0])
        except (IndexError, ValueError):
            epoch_number = None

    # If epoch number not in path, try to get it from model state
    if epoch_number is None and 'epoch' in model_state:
        epoch_number = model_state['epoch']

    # Get model parameters
    model_params = {
        "num_parameters": sum(p.numel() for p in model.parameters())
    }

    # CUSTOM MORPH DATASETS
    if any(key in eval_config for key in ["LMA_path", "LMA_UBO_path", "MIPGAN_I_path", "MIPGAN_II_path", "MorDiff_path", "StyleGAN_path"]):
        custom_test_datasets = default_datasets(image_size, datasets="custom_morph", config=eval_config)
        if custom_test_datasets:  # Only proceed if we have datasets
            custom_test_loaders = prep_dataloaders(custom_test_datasets, batch_size)

            # Skip printing dataset details to keep console output clean

            # Use the new evaluation module for custom datasets
            custom_results, custom_targets_outputs_dict, custom_class_separated_dict = evaluate(
                model=model,
                test_loaders=custom_test_loaders,
                device=device,
                output_dir=os.path.join(eval_dir, "standalone_eval"),
                model_name=f"{model_name}_standalone_eval",
                model_dataset=model_dataset,
                model_type=model_type,
                epoch_number=epoch_number,
                verbose=eval_config["verbose"],
                classification_threshold=args.classification_threshold,
                apcer_thresholds=args.apcer_thresholds,
                model_params=model_params
            )

            # Use the visualization module to generate plots for custom datasets
            # Determine whether to generate plots based on command-line arguments
            generate_plots = eval_config.get("generate_plots", True)
            # If evaluating the best model, always generate plots unless explicitly set to False
            if eval_config.get("evaluate_best", False) and "generate_plots" not in eval_config:
                generate_plots = True

            visualize(
                results=custom_results,
                targets_outputs_dict=custom_targets_outputs_dict,
                class_separated_dict=custom_class_separated_dict,
                output_dir=os.path.join(eval_dir, "standalone_eval"),
                model_name=f"{model_name}_standalone_eval",
                model_dataset=model_dataset,
                model_type=model_type,
                epoch_number=epoch_number,
                generate_plots=generate_plots
            )

            print(f"Standalone evaluation results saved to {os.path.join(eval_dir, 'standalone_eval')}")
        else:
            print("No custom morph datasets found for evaluation.")
    else:
        print("No custom morph dataset paths provided for evaluation.")
        custom_test_datasets = {}

    # Create marker file for evaluation completion
    marker_info = {
        "datasets_evaluated": [],
        "custom_datasets_evaluated": list(custom_test_datasets.keys()) if 'custom_test_datasets' in locals() and custom_test_datasets else [],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(os.path.join(eval_dir, "evaluation_complete.marker"), "w") as f:
        json.dump(marker_info, f, cls=NumpyEncoder)

    # Create a combined metrics file that includes results from all datasets
    combined_metrics = {}
    if 'custom_results' in locals() and custom_results:
        combined_metrics.update(custom_results)

    # Save combined metrics to a JSON file
    with open(os.path.join(eval_dir, "combined_results.json"), "w") as f:
        json.dump(combined_metrics, f, cls=NumpyEncoder)

    print(f"Evaluation complete marker created at {os.path.join(eval_dir, 'evaluation_complete.marker')}")
    print(f"Combined results saved to {os.path.join(eval_dir, 'combined_results.json')}")

    # MORDIFF DATASET
    # test_datasets = default_datasets(image_size, datasets="MorDIFF", config=eval_config)
    # test_loaders = prep_dataloaders(test_datasets, batch_size)

    # if eval_config["verbose"]:
    #     for dataset in test_datasets:
    #         print(f'-----{dataset}:')
    #         for method in test_datasets[dataset]:
    #             print('-', method)
    #             print("real", test_datasets[dataset][method].labels.count(0))
    #             print("fake", test_datasets[dataset][method].labels.count(1))

    # evaluate_simple(model, test_loaders, device, calculate_means=False, verbose=True)


def default_datasets(image_size, datasets="original", config=None):
    assert datasets in ["original", "custom_morph"
                        # "MorDIFF"
                        ]
    if datasets == "original":
        FRGC_datapath = config["FRGC_path"]
        FERET_datapath = config["FERET_path"]
        FRLL_datapath = config["FRLL_path"]

        test_datasets = {
            "FRGC": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='stylegan')
            },
            "FERET": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='stylegan')
            },
            "FRLL": {
                "amsl": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='amsl'),
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='stylegan'),
                "wm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='webmorph')
            }
        }
        return test_datasets
    elif datasets == "custom_morph":
        # Create test datasets for our custom morph datasets
        morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN"]
        test_datasets = {}

        for dataset_name in morph_datasets:
            # Check if dataset path is in config
            path_key = f"{dataset_name}_path"
            if path_key in config and config[path_key] is not None:
                # Create test dataset
                test_datasets[dataset_name] = {
                    "test": TestMorphDataset(dataset_name=dataset_name, image_size=image_size)
                }

        return test_datasets
    # if datasets == "MorDIFF":
    #     test_datasets = {
    #         "MorDIFF": {
    #             "MorDIFF": MorDIFF(datapath_fake=config["MorDIFF_f_path"],
    #                                 datapath_real=config["MorDIFF_bf_path"],
    #                                 image_size=image_size)
    #         }
    #     }
    #     return test_datasets

def prep_dataloaders(test_datasets, batch_size):
    test_loaders = {
        dataset: {
            method: torch.utils.data.DataLoader(test_datasets[dataset][method],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
            for method in test_datasets[dataset]
        }
        for dataset in test_datasets
    }
    return test_loaders

def evaluate_simple(model, test_loaders, device, calculate_means=True, verbose=False, multi=False):
    """
    Simple evaluation function that calculates AUC and EER metrics.
    This is different from the more comprehensive evaluate() function imported from evaluation_2.py.
    """
    results = {}
    if calculate_means:
        total_eers, total_aucs = [], []
    for dataset_loader in test_loaders:
        for method_loader in test_loaders[dataset_loader]:
            output_dict = []
            target_dict = []
            for data in tqdm(test_loaders[dataset_loader][method_loader], desc=f"Evaluating {dataset_loader}_{method_loader}"):
                # Handle both tuple format (img, target) and dictionary format {'img': img, 'label': target}
                if isinstance(data, tuple):
                    img = data[0].to(device, non_blocking=True).float()
                    target = data[1].to(device, non_blocking=True).long()
                else:
                    img = data['img'].to(device, non_blocking=True).float()
                    target = data['label'].to(device, non_blocking=True).long()
                with torch.no_grad():
                    output = model(img)
                if multi:
                    output = torch.cat((output[:, 0].unsqueeze(1), output[:, 1:].sum(1).unsqueeze(1)), dim=1)
                output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
                target_dict += target.cpu().data.numpy().tolist()
            eer = calculate_eer(target_dict, output_dict)
            auc = calculate_auc(target_dict, output_dict)
            if calculate_means:
                total_eers.append(eer)
                total_aucs.append(auc)
            if verbose:
                print(f"{dataset_loader}_{method_loader} auc: {auc:.4f}, eer: {eer:.4f}")
            results[f"{dataset_loader}_{method_loader}"] = {"auc": auc, "eer": eer}
    if calculate_means:
        if verbose:
            print(f"Total mean auc: {np.mean(total_aucs):.4f}, mean eer: {np.mean(total_eers):.4f}")
        results["mean"] = {"auc": np.mean(total_aucs), "eer": np.mean(total_eers)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Print more information')
    parser.add_argument('-m', dest='model_type', type=str, required=True, help='Type of the model, e.g. hrnet_w18')
    parser.add_argument('-p', dest='model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('-o', dest='output_dir', type=str, required=False, help='Output directory for evaluation results')
    parser.add_argument('-t', dest='classification_threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    parser.add_argument('-apcer_thresholds', type=float, nargs='+', default=[0.05, 0.10, 0.20],
                        help='APCER thresholds for BPCER calculation (default: 0.05 0.10 0.20)')
    parser.add_argument('--evaluate_best', action='store_true', help='Evaluate the best model instead of the specified epoch')
    parser.add_argument('--generate_plots', action='store_true', help='Generate plots during evaluation (default: True for best model, False otherwise)')
    parser.add_argument('-FRLL_path', type=str, required=False)
    parser.add_argument('-FRGC_path', type=str, required=False)
    parser.add_argument('-FERET_path', type=str, required=False)
    parser.add_argument('-train_dataset', type=str, required=False, help='Dataset the model was trained on')
    # parser.add_argument('-MorDIFF_f_path', type=str, required=False)
    # parser.add_argument('-MorDIFF_bf_path', type=str, required=False)

    # Custom morph dataset parameters
    parser.add_argument('-LMA_path', type=str, required=False)
    parser.add_argument('-LMA_UBO_path', type=str, required=False)
    parser.add_argument('-MIPGAN_I_path', type=str, required=False)
    parser.add_argument('-MIPGAN_II_path', type=str, required=False)
    parser.add_argument('-MorDiff_path', type=str, required=False)
    parser.add_argument('-StyleGAN_path', type=str, required=False)
    args = parser.parse_args()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base data configuration
    try:
        data_config_path = os.path.join(script_dir, "configs", "data_config.json")
        print(f"Loading data config from: {data_config_path}")
        eval_config = json.load(open(data_config_path))
    except FileNotFoundError:
        print(f"Warning: data_config.json not found at {data_config_path}. Using default values.")
        eval_config = {
            "FRLL_path": "",
            "FRGC_path": "",
            "FERET_path": "",
            "FF_path": "",
            "SMDD_path": "",
            "verbose": True,
            "classification_threshold": 0.5,
            "apcer_thresholds": [0.05, 0.1, 0.2]
        }

    # Override with command line arguments
    for key in vars(args):
        if vars(args)[key] is not None:
            eval_config[key] = vars(args)[key]

    # Load morph dataset configuration if needed
    if any(key.startswith(prefix) for prefix in ["LMA", "MIPGAN", "MorDiff", "StyleGAN"] for key in eval_config):
        try:
            morph_config_path = os.path.join(script_dir, "configs", "morph_config.json")
            print(f"Loading morph config from: {morph_config_path}")
            morph_config = json.load(open(morph_config_path))
            for key in morph_config:
                if key not in eval_config:
                    eval_config[key] = morph_config[key]
        except FileNotFoundError:
            print(f"Warning: morph_config.json not found at {morph_config_path}. Using default values.")

    main(eval_config)
