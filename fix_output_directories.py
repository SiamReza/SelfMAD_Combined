#!/usr/bin/env python3
"""
Script to fix the output directory structure in run_experiments_direct.py.
This script modifies run_experiments_direct.py to ensure that all output files
are saved in the correct locations.
"""

import os
import re
import shutil
import argparse

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def fix_run_experiments_direct(dry_run=False):
    """
    Fix the output directory handling in run_experiments_direct.py.
    
    Args:
        dry_run (bool): If True, only print what would be changed without actually modifying files.
    """
    file_path = "run_experiments_direct.py"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    # Create a backup
    if not dry_run:
        backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Modify how the save path is constructed for SIAM models
    siam_pattern = r"(siam_model_dir = os\.path\.join\(args\.output_dir, \"siam\", args\.siam_model\))"
    siam_replacement = r"# Use a path relative to the root directory, not the repository directory\n    root_dir = os.path.abspath(os.path.dirname(__file__))\n    siam_model_dir = os.path.join(root_dir, args.output_dir, \"siam\", args.siam_model)"
    
    if re.search(siam_pattern, content):
        print(f"Found SIAM model directory pattern in {file_path}")
        if not dry_run:
            content = re.sub(siam_pattern, siam_replacement, content)
            print("Updated SIAM model directory construction")
    else:
        print(f"Warning: SIAM model directory pattern not found in {file_path}")
    
    # Fix 2: Modify how the save path is constructed for MAIN models
    main_pattern = r"(main_model_dir = os\.path\.join\(args\.output_dir, \"main\", current_model\))"
    main_replacement = r"# Use a path relative to the root directory, not the repository directory\n    root_dir = os.path.abspath(os.path.dirname(__file__))\n    main_model_dir = os.path.join(root_dir, args.output_dir, \"main\", current_model)"
    
    if re.search(main_pattern, content):
        print(f"Found MAIN model directory pattern in {file_path}")
        if not dry_run:
            content = re.sub(main_pattern, main_replacement, content)
            print("Updated MAIN model directory construction")
    else:
        print(f"Warning: MAIN model directory pattern not found in {file_path}")
    
    # Fix 3: Modify how the command is executed to avoid changing the working directory
    cmd_pattern = r"(cwd = f\"SelfMAD-\{model_type\}\"\n\s+train_exit_code = run_command\(train_command, cwd=cwd\))"
    cmd_replacement = r"# Use absolute paths in the command instead of changing the working directory\n    repo_dir = os.path.abspath(f\"SelfMAD-{model_type}\")\n    train_command = f\"cd {repo_dir} && {train_command}\"\n    train_exit_code = run_command(train_command)"
    
    if re.search(cmd_pattern, content):
        print(f"Found command execution pattern in {file_path}")
        if not dry_run:
            content = re.sub(cmd_pattern, cmd_replacement, content)
            print("Updated command execution to use absolute paths")
    else:
        print(f"Warning: Command execution pattern not found in {file_path}")
    
    # Fix 4: Modify the evaluation command execution
    eval_pattern = r"(eval_exit_code = run_command\(eval_command, cwd=cwd\))"
    eval_replacement = r"# Use absolute paths in the command instead of changing the working directory\n                eval_command = f\"cd {repo_dir} && {eval_command}\"\n                eval_exit_code = run_command(eval_command)"
    
    if re.search(eval_pattern, content):
        print(f"Found evaluation command pattern in {file_path}")
        if not dry_run:
            content = re.sub(eval_pattern, eval_replacement, content)
            print("Updated evaluation command execution to use absolute paths")
    else:
        print(f"Warning: Evaluation command pattern not found in {file_path}")
    
    # Write the modified content back to the file
    if not dry_run:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated {file_path}")
    else:
        print("Dry run completed. No changes were made.")
    
    return True

def fix_train_scripts(dry_run=False):
    """
    Fix the output directory handling in SelfMAD-siam/train__.py and SelfMAD-main/train__.py.
    
    Args:
        dry_run (bool): If True, only print what would be changed without actually modifying files.
    """
    # Fix SelfMAD-siam/train__.py
    siam_train_path = "SelfMAD-siam/train__.py"
    
    if not os.path.exists(siam_train_path):
        print(f"Warning: {siam_train_path} not found")
    else:
        # Create a backup
        if not dry_run:
            backup_file(siam_train_path)
        
        with open(siam_train_path, 'r') as f:
            siam_content = f.read()
        
        # Fix the output directory construction
        siam_pattern = r"(output_dir = os\.path\.join\(\"output\", \"siam\", model_name\))"
        siam_replacement = r"if args.get(\"save_path\"):\n        # Use the provided absolute path\n        output_dir = args[\"save_path\"]\n    else:\n        # Use a path relative to the parent directory (outside the repository)\n        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))\n        output_dir = os.path.join(parent_dir, \"output\", \"siam\", model_name)"
        
        if re.search(siam_pattern, siam_content):
            print(f"Found output directory pattern in {siam_train_path}")
            if not dry_run:
                siam_content = re.sub(siam_pattern, siam_replacement, siam_content)
                print(f"Updated output directory construction in {siam_train_path}")
        else:
            print(f"Warning: Output directory pattern not found in {siam_train_path}")
        
        # Write the modified content back to the file
        if not dry_run:
            with open(siam_train_path, 'w') as f:
                f.write(siam_content)
            print(f"Updated {siam_train_path}")
    
    # Fix SelfMAD-main/train__.py
    main_train_path = "SelfMAD-main/train__.py"
    
    if not os.path.exists(main_train_path):
        print(f"Warning: {main_train_path} not found")
    else:
        # Create a backup
        if not dry_run:
            backup_file(main_train_path)
        
        with open(main_train_path, 'r') as f:
            main_content = f.read()
        
        # Fix the output directory construction
        main_pattern = r"(output_dir = os\.path\.join\(\"output\", \"main\", model_name\))"
        main_replacement = r"if args.get(\"save_path\"):\n        # Use the provided absolute path\n        output_dir = args[\"save_path\"]\n    else:\n        # Use a path relative to the parent directory (outside the repository)\n        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\"))\n        output_dir = os.path.join(parent_dir, \"output\", \"main\", model_name)"
        
        if re.search(main_pattern, main_content):
            print(f"Found output directory pattern in {main_train_path}")
            if not dry_run:
                main_content = re.sub(main_pattern, main_replacement, main_content)
                print(f"Updated output directory construction in {main_train_path}")
        else:
            print(f"Warning: Output directory pattern not found in {main_train_path}")
        
        # Write the modified content back to the file
        if not dry_run:
            with open(main_train_path, 'w') as f:
                f.write(main_content)
            print(f"Updated {main_train_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Fix output directory structure in SelfMAD Combined")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be changed without actually modifying files")
    args = parser.parse_args()
    
    print("Fixing output directory structure...")
    
    # Fix run_experiments_direct.py
    fix_run_experiments_direct(dry_run=args.dry_run)
    
    # Fix train scripts
    fix_train_scripts(dry_run=args.dry_run)
    
    print("Done!")

if __name__ == "__main__":
    main()
