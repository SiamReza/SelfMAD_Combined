#!/usr/bin/env python3
"""
Setup script to copy the modified files to the correct locations.

This script copies the modified evaluation files to the correct locations
and ensures that the run_experiments_modified.py script uses the correct paths.
"""

import os
import shutil
import sys

def copy_file(src, dst):
    """Copy a file from src to dst, creating directories if needed."""
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Copy the file
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False

def main():
    """Main function to copy the modified files."""
    print("Setting up modified files...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the files to copy
    files_to_copy = [
        # Visualization files
        ("visualization_1.py", "visualization_1.py"),
        ("visualization_2.py", "visualization_2.py"),
        ("visualization_3.py", "visualization_3.py"),
        
        # Evaluation files
        ("evaluation_1.py", "evaluation_1.py"),
        ("evaluation_2.py", "evaluation_2.py"),
        
        # Robust model handler
        ("robust_model_handler.py", "robust_model_handler.py"),
        
        # Modified evaluation scripts
        ("SelfMAD-siam/eval__modified.py", "SelfMAD-siam/eval__.py"),
        ("SelfMAD-main/eval__modified.py", "SelfMAD-main/eval__.py"),
        
        # Documentation files
        ("vis_eval_doc.md", "vis_eval_doc.md"),
        ("model_path_fix.md", "model_path_fix.md"),
        ("implementation_summary.md", "implementation_summary.md"),
        ("error_handling_log.md", "error_handling_log.md")
    ]
    
    # Copy each file
    success = True
    for src, dst in files_to_copy:
        src_path = os.path.join(current_dir, src)
        dst_path = os.path.join(current_dir, dst)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source file {src_path} does not exist.")
            success = False
            continue
        
        if not copy_file(src_path, dst_path):
            success = False
    
    # Rename run_experiments_modified.py to run_experiments.py
    run_experiments_src = os.path.join(current_dir, "run_experiments_modified.py")
    run_experiments_dst = os.path.join(current_dir, "run_experiments.py")
    
    if os.path.exists(run_experiments_src):
        try:
            # Create a backup of the original file
            if os.path.exists(run_experiments_dst):
                backup_path = os.path.join(current_dir, "run_experiments_backup.py")
                shutil.copy2(run_experiments_dst, backup_path)
                print(f"Created backup of original run_experiments.py at {backup_path}")
            
            # Copy the modified file
            shutil.copy2(run_experiments_src, run_experiments_dst)
            print(f"Copied {run_experiments_src} to {run_experiments_dst}")
        except Exception as e:
            print(f"Error copying {run_experiments_src} to {run_experiments_dst}: {e}")
            success = False
    else:
        print(f"Warning: Source file {run_experiments_src} does not exist.")
        success = False
    
    # Print summary
    if success:
        print("\nAll files copied successfully.")
        print("\nYou can now run the experiments using:")
        print("  python run_experiments.py")
    else:
        print("\nSome files could not be copied. Please check the errors above.")
        print("\nYou can still run the experiments using:")
        print("  python run_experiments_modified.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
