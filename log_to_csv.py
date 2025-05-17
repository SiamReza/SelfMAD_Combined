#!/usr/bin/env python
"""
Utility script to convert losses.log files to CSV format.

This script can be used to convert existing losses.log files to CSV format
for backward compatibility with the new CSV logging system.
"""

import os
import re
import csv
import argparse
from datetime import datetime

def convert_log_to_csv(log_file, csv_file=None):
    """
    Convert a losses.log file to CSV format.
    
    Args:
        log_file: Path to the log file
        csv_file: Path to the output CSV file (default: same as log_file with .csv extension)
    """
    if csv_file is None:
        csv_file = log_file.replace('.logs', '.csv')
    
    print(f"Converting {log_file} to {csv_file}...")
    
    # Read the log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the log lines to extract structured data
    data = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Extract data using regex
        # Example log format: "Epoch 1/10 | train loss: 0.1234 | val loss: 0.5678, val auc: 0.9876, val eer: 0.1234 |"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract epoch information
        epoch_match = re.search(r'Epoch (\d+)/\d+', line)
        epoch = epoch_match.group(1) if epoch_match else ""
        
        # Extract train loss
        train_loss_match = re.search(r'train loss: ([\d\.]+)', line)
        train_loss = train_loss_match.group(1) if train_loss_match else ""
        
        # Extract validation metrics
        val_loss_match = re.search(r'val loss: ([\d\.]+)', line)
        val_loss = val_loss_match.group(1) if val_loss_match else ""
        
        val_auc_match = re.search(r'val auc: ([\d\.]+)', line)
        val_auc = val_auc_match.group(1) if val_auc_match else ""
        
        val_eer_match = re.search(r'val eer: ([\d\.]+)', line)
        val_eer = val_eer_match.group(1) if val_eer_match else ""
        
        # Additional info (everything after the validation metrics)
        additional_info = ""
        if val_eer_match:
            pos = val_eer_match.end()
            if pos < len(line):
                additional_info = line[pos:].strip()
        
        # Add to data list if we have at least some information
        if epoch or train_loss or val_loss or val_auc or val_eer:
            data.append([timestamp, epoch, train_loss, val_loss, val_auc, val_eer, additional_info])
    
    # Write to CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write headers
        writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_loss', 'val_auc', 'val_eer', 'additional_info'])
        # Write data
        for row in data:
            writer.writerow(row)
    
    print(f"Converted {log_file} to {csv_file}")
    print(f"Processed {len(data)} log entries")

def find_log_files(directory, recursive=False):
    """
    Find all losses.log files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
    
    Returns:
        List of paths to losses.log files
    """
    log_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "losses.logs":
                    log_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file == "losses.logs":
                log_files.append(os.path.join(directory, file))
    
    return log_files

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert losses.log to CSV format')
    parser.add_argument('log_file', nargs='?', help='Path to the log file (optional if --directory is specified)')
    parser.add_argument('--csv-file', help='Path to the output CSV file')
    parser.add_argument('--directory', '-d', help='Directory to search for losses.log files')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search directory recursively')
    args = parser.parse_args()
    
    if args.directory:
        # Find all losses.log files in the directory
        log_files = find_log_files(args.directory, args.recursive)
        
        if not log_files:
            print(f"No losses.logs files found in {args.directory}")
            return
        
        print(f"Found {len(log_files)} losses.logs files")
        
        # Convert each log file
        for log_file in log_files:
            convert_log_to_csv(log_file)
    elif args.log_file:
        # Convert a single log file
        convert_log_to_csv(args.log_file, args.csv_file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
