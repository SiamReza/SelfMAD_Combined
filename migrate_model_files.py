#!/usr/bin/env python3
"""
Script to migrate model files from repository directories to the root output directory.
This script helps fix the output folder structure issue by copying model files from
SelfMAD-siam/output and SelfMAD-main/output to the root output directory.
"""

import os
import shutil
import glob
import argparse
import sys

def migrate_models(dry_run=False, remove_source=False):
    """
    Migrate model files from repository directories to the root output directory.

    Args:
        dry_run (bool): If True, only print what would be done without actually copying files.
        remove_source (bool): If True, remove source files after copying.
    """
    print("Starting model file migration...")

    # Create root output directories if they don't exist
    if not dry_run:
        os.makedirs("output/siam/vit_mae_large/model/weights", exist_ok=True)
        os.makedirs("output/main", exist_ok=True)
        os.makedirs("output/train", exist_ok=True)

    # Find all model files in SelfMAD-siam
    siam_models = glob.glob("SelfMAD-siam/output/siam/vit_mae_large/model/weights/*.tar")
    print(f"Found {len(siam_models)} model files in SelfMAD-siam")

    for model_path in siam_models:
        # Get the filename
        filename = os.path.basename(model_path)
        # Create the destination path
        dest_path = os.path.join("output/siam/vit_mae_large/model/weights", filename)

        # Check if the destination file already exists
        if os.path.exists(dest_path):
            print(f"Skipping {model_path} as {dest_path} already exists")
            continue

        # Copy the file
        print(f"{'Would copy' if dry_run else 'Copying'} {model_path} to {dest_path}")
        if not dry_run:
            try:
                shutil.copy2(model_path, dest_path)
                print(f"Successfully copied {filename}")

                # Remove source file if requested
                if remove_source:
                    print(f"Removing source file: {model_path}")
                    os.remove(model_path)
            except Exception as e:
                print(f"Error copying {model_path}: {e}")

    # Find all model files in SelfMAD-main
    main_models = glob.glob("SelfMAD-main/output/main/**/*.tar", recursive=True)
    print(f"Found {len(main_models)} model files in SelfMAD-main")

    for model_path in main_models:
        try:
            # Extract the model name from the path
            path_parts = model_path.split(os.path.sep)
            # Find the index of "main" in the path
            main_index = path_parts.index("main")
            if main_index + 1 < len(path_parts):
                model_name = path_parts[main_index + 1]
            else:
                print(f"Skipping {model_path} as it doesn't have a model name in the path")
                continue

            # Create the destination directory
            dest_dir = os.path.join("output/main", model_name, "model")
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)

            # Get the filename
            filename = os.path.basename(model_path)

            # Create the destination path
            dest_path = os.path.join(dest_dir, filename)

            # Check if the destination file already exists
            if os.path.exists(dest_path):
                print(f"Skipping {model_path} as {dest_path} already exists")
                continue

            # Copy the file
            print(f"{'Would copy' if dry_run else 'Copying'} {model_path} to {dest_path}")
            if not dry_run:
                shutil.copy2(model_path, dest_path)
                print(f"Successfully copied {filename}")

                # Remove source file if requested
                if remove_source:
                    print(f"Removing source file: {model_path}")
                    os.remove(model_path)
        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    # Also copy any CSV files or other training data
    train_files = glob.glob("SelfMAD-siam/output/train/**/*.*", recursive=True)
    train_files.extend(glob.glob("SelfMAD-main/output/train/**/*.*", recursive=True))
    print(f"Found {len(train_files)} training files")

    for file_path in train_files:
        try:
            # Get the relative path within the train directory
            if "SelfMAD-siam/output/train/" in file_path:
                rel_path = file_path.split("SelfMAD-siam/output/train/")[1]
            elif "SelfMAD-main/output/train/" in file_path:
                rel_path = file_path.split("SelfMAD-main/output/train/")[1]
            else:
                print(f"Skipping {file_path} as it doesn't match expected path structure")
                continue

            # Create the destination path
            dest_path = os.path.join("output/train", rel_path)

            # Create the destination directory if needed
            if not dry_run:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Check if the destination file already exists
            if os.path.exists(dest_path):
                print(f"Skipping {file_path} as {dest_path} already exists")
                continue

            # Copy the file
            print(f"{'Would copy' if dry_run else 'Copying'} {file_path} to {dest_path}")
            if not dry_run:
                shutil.copy2(file_path, dest_path)
                print(f"Successfully copied {os.path.basename(file_path)}")

                # Remove source file if requested
                if remove_source:
                    print(f"Removing source file: {file_path}")
                    os.remove(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Optionally clean up empty directories in repository output folders
    if remove_source and not dry_run:
        print("Cleaning up empty directories in repository output folders...")

        # Clean up SelfMAD-siam output directory
        if os.path.exists("SelfMAD-siam/output"):
            for root, dirs, files in os.walk("SelfMAD-siam/output", topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if not os.listdir(dir_path):  # Check if directory is empty
                        print(f"Removing empty directory: {dir_path}")
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            print(f"Error removing directory {dir_path}: {e}")

        # Clean up SelfMAD-main output directory
        if os.path.exists("SelfMAD-main/output"):
            for root, dirs, files in os.walk("SelfMAD-main/output", topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if not os.listdir(dir_path):  # Check if directory is empty
                        print(f"Removing empty directory: {dir_path}")
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            print(f"Error removing directory {dir_path}: {e}")

    print("Migration completed!")

def main():
    parser = argparse.ArgumentParser(description="Migrate model files from repository directories to the root output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done without actually copying files")
    parser.add_argument("--remove-source", action="store_true", help="Remove source files after copying")
    args = parser.parse_args()

    migrate_models(dry_run=args.dry_run, remove_source=args.remove_source)

if __name__ == "__main__":
    main()
