#!/usr/bin/env python3
"""
Script to run training for all models specified in automation_config.py.
This script allows running all 5 models mentioned in the original SelfMAD paper:
1. HRNet-W18
2. EfficientNet-b4
3. EfficientNet-b7
4. Swin_base
5. ResNet-152
"""

import os
import sys
import subprocess
import argparse

# Define the models to run if not imported from automation_config.py
DEFAULT_MODELS = ["hrnet_w18", "efficientnet-b4", "efficientnet-b7", "swin", "resnet"]

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "default": {"batch_size": 32, "learning_rate": 5e-4, "epochs": 2},
    "hrnet_w18": {"batch_size": 32, "learning_rate": 5e-4, "epochs": 2},
    "efficientnet-b4": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2},
    "efficientnet-b7": {"batch_size": 16, "learning_rate": 5e-4, "epochs": 2},
    "swin": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2},
    "resnet": {"batch_size": 24, "learning_rate": 5e-4, "epochs": 2}
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run training for all models specified in automation_config.py")

    # Try to import from automation_config.py
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from automation_config import MAIN_MODELS, MAIN_MODEL_CONFIGS, DATASETS, OUTPUT_DIR

        parser.add_argument("--models", nargs="+", default=MAIN_MODELS,
                           help="Models to process")
        parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to process")
        parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Base output directory")

        # Use model configs from automation_config.py
        model_configs = MAIN_MODEL_CONFIGS

    except (ImportError, AttributeError) as e:
        print(f"Note: Could not load all configurations from automation_config.py: {e}")
        print("Using default values for missing configurations.")

        # Try to import just the datasets
        try:
            from automation_config import DATASETS, OUTPUT_DIR
            parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to process")
            parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to process")
            parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Base output directory")
        except (ImportError, AttributeError):
            # Use all defaults if automation_config.py can't be imported
            parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to process")
            parser.add_argument("--datasets", nargs="+", default=["StyleGAN", "MIPGAN_II"],
                               help="Datasets to process")
            parser.add_argument("--output-dir", default="./output", help="Base output directory")

        # Use default model configs
        model_configs = DEFAULT_MODEL_CONFIGS

    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")

    args = parser.parse_args()

    # Store model configs in args for later use
    args.model_configs = model_configs

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

def find_model_file(model_dir):
    """Find the most recent model file in the model directory."""
    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".tar"):
                model_files.append(os.path.join(root, file))

    if not model_files:
        return None

    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def main():
    """Main function."""
    args = parse_args()

    print("\n=== Running training for all models ===\n")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip evaluation: {args.skip_eval}")

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create main directory
    main_dir = os.path.join(args.output_dir, "main")
    os.makedirs(main_dir, exist_ok=True)

    # Process each model
    for model in args.models:
        print(f"\n=== Processing model: {model} ===\n")

        # Get model-specific configuration
        model_config = args.model_configs.get(model, args.model_configs.get("default", {"batch_size": 32, "learning_rate": 5e-4, "epochs": 2}))
        batch_size = model_config["batch_size"]
        learning_rate = model_config["learning_rate"]
        epochs = model_config.get("epochs", 2)

        # Process each dataset
        for dataset in args.datasets:
            print(f"\n=== Processing dataset: {dataset} for model: {model} ===\n")

            # Create base output directory for this dataset
            base_output_dir = os.path.join(args.output_dir, "main", f"{dataset}_main_1")
            os.makedirs(base_output_dir, exist_ok=True)

            # Create model-specific output directory
            model_output_dir = os.path.join(base_output_dir, model)
            os.makedirs(model_output_dir, exist_ok=True)

            # Create session name
            session_name = f"{dataset}_{model}"

            # Construct training command
            train_command = (
                f"python train__.py "
                f"-n {session_name} "
                f"-m {model} "
                f"-b {batch_size} "
                f"-e {epochs} "
                f"-lr {learning_rate} "
                f"-v testset_best "
                f"-t {dataset} "
                f"-s {model_output_dir} "
                f"-train_val_split 0.8"
            )

            # Run training
            print(f"\n=== Running training for {dataset} with model {model} ===\n")
            cwd = "SelfMAD-main"
            train_exit_code = run_command(train_command, cwd=cwd)

            if train_exit_code != 0:
                print(f"Warning: Training failed for dataset {dataset} with model {model}.")
                continue

            # Skip evaluation if requested
            if args.skip_eval:
                continue

            # Find the model file
            model_path = find_model_file(model_output_dir)

            if not model_path:
                print(f"Warning: No model file found for dataset {dataset} with model {model}.")
                continue

            # Create eval directory
            eval_dir = os.path.join(model_output_dir, 'eval')
            os.makedirs(eval_dir, exist_ok=True)

            # Construct evaluation command
            eval_command = (
                f"python eval__.py "
                f"-m {model} "
                f"-p {model_path} "
                f"-v "
                f"-o {eval_dir} "
                f"-t 0.5 "
                f"-apcer_thresholds 0.05 0.10 0.20"
            )

            # Run evaluation
            print(f"\n=== Running evaluation for {dataset} with model {model} ===\n")
            eval_exit_code = run_command(eval_command, cwd=cwd)

            if eval_exit_code != 0:
                print(f"Warning: Evaluation failed for dataset {dataset} with model {model}.")

    print("\n=== All models processed ===\n")

if __name__ == "__main__":
    main()
