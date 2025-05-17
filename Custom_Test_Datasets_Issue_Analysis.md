# Custom Test Datasets Issue Analysis

## Issue Description

The system was displaying the following warning message:

```
Warning: No custom test datasets available. Using default values for results_original_dataset.
```

This warning indicates that the system couldn't find the custom test datasets, which are required for the `testset_best` saving strategy to work properly. Without these test datasets, the system falls back to using default values for evaluation metrics, which prevents the proper functioning of the `testset_best` saving strategy.

## Root Cause Analysis

After thorough investigation, the root cause was identified in the `run_experiments_direct.py` script. The issue was in how dataset paths were being passed to the training and evaluation scripts:

1. In `automation_config.py`, the dataset paths were correctly defined using an absolute path:
   ```python
   datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
   DATASET_PATHS = {
       "LMA_path": datasets_dir,
       "LMA_UBO_path": datasets_dir,
       "MIPGAN_I_path": datasets_dir,
       "MIPGAN_II_path": datasets_dir,
       "MorDiff_path": datasets_dir,
       "StyleGAN_path": datasets_dir
   }
   ```

2. However, in `run_experiments_direct.py`, when constructing the command-line arguments for the evaluation script, it wasn't correctly using these paths:
   ```python
   # Construct dataset path arguments
   dataset_path_args = ""
   if hasattr(args, "dataset_paths"):
       for path_name, path_value in args.dataset_paths.items():
           dataset_path_args += f" -{path_name} {path_value}"
   else:
       # Default dataset paths using os.path.join for cross-platform compatibility
       datasets_dir = os.path.join("..", "datasets")
       dataset_path_args = f" -LMA_path {datasets_dir} -LMA_UBO_path {datasets_dir} -MIPGAN_I_path {datasets_dir} -MIPGAN_II_path {datasets_dir} -MorDiff_path {datasets_dir} -StyleGAN_path {datasets_dir}"
   ```

3. The issue was that the code wasn't correctly checking if `args.dataset_paths` was a dictionary before trying to iterate over it with `.items()`. Additionally, it was falling back to using a relative path (`"../datasets"`) instead of an absolute path.

4. When the evaluation script ran, it couldn't find the test datasets because it was looking in the wrong location (relative to the SelfMAD-main or SelfMAD-siam directory, not the root directory).

## Solution

The solution was to modify the `run_experiments_direct.py` script to correctly use the dataset paths from `automation_config.py`:

```python
# Construct dataset path arguments
dataset_path_args = ""
if hasattr(args, "dataset_paths") and isinstance(args.dataset_paths, dict):
    # Use dataset paths from automation_config.py
    for path_name, path_value in args.dataset_paths.items():
        dataset_path_args += f" -{path_name} {path_value}"
    print(f"Using dataset paths from automation_config.py: {args.dataset_paths}")
else:
    # Default dataset paths using os.path.join for cross-platform compatibility
    # Try to use absolute path to datasets directory
    try:
        # First try to get the absolute path to the datasets directory
        datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
        if not os.path.exists(datasets_dir):
            # Fall back to relative path if absolute path doesn't exist
            datasets_dir = os.path.join("..", "datasets")
    except Exception as e:
        print(f"Warning: Could not determine absolute path to datasets directory: {e}")
        datasets_dir = os.path.join("..", "datasets")
    
    print(f"Using datasets directory: {datasets_dir}")
    dataset_path_args = f" -LMA_path {datasets_dir} -LMA_UBO_path {datasets_dir} -MIPGAN_I_path {datasets_dir} -MIPGAN_II_path {datasets_dir} -MorDiff_path {datasets_dir} -StyleGAN_path {datasets_dir}"
```

Key improvements in the solution:

1. Added a check to ensure `args.dataset_paths` is a dictionary before trying to iterate over it
2. Added debug output to show which dataset paths are being used
3. Added a fallback mechanism to use an absolute path to the datasets directory if possible
4. Added error handling to gracefully handle cases where the absolute path can't be determined

## Implementation Details

The changes were made to the `process_model` function in `run_experiments_direct.py`, specifically in the section that constructs the dataset path arguments for the evaluation script.

The modified code now:
1. Properly checks if `args.dataset_paths` is a dictionary
2. Uses the dataset paths from `automation_config.py` if available
3. Falls back to using an absolute path to the datasets directory if the paths from `automation_config.py` aren't available
4. Provides better error handling and debug output

## Expected Outcome

With these changes, the system should now be able to find the custom test datasets, and the warning message should no longer appear. The `testset_best` saving strategy should now work correctly, using the actual test dataset results instead of default values.

## Verification Steps

To verify that the issue has been fixed:

1. Run the `run_experiments_direct.py` script
2. Check if the warning message "No custom test datasets available" still appears
3. Verify that the system is using the correct dataset paths by checking the debug output
4. Confirm that the `testset_best` saving strategy is working correctly by examining the saved model weights

## Additional Recommendations

1. Consider adding more robust error handling and logging throughout the codebase to make it easier to diagnose similar issues in the future
2. Add validation checks to ensure that the dataset paths are valid and that the required directory structure exists
3. Consider adding unit tests to verify that the dataset loading code works correctly with different path configurations
