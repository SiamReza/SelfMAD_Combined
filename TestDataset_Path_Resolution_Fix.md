# Test Dataset Path Resolution Fix

## Issue Description

The system was displaying the following warning message:

```
Warning: No custom test datasets available. Using default values for results_original_dataset.
```

This warning indicates that the system couldn't find the custom test datasets, which are required for the `testset_best` saving strategy to work properly. Without these test datasets, the system falls back to using default values for evaluation metrics, which prevents the proper functioning of the `testset_best` saving strategy.

## Root Cause Analysis

After thorough investigation, the root cause was identified in the `SelfMAD-main/train__.py` script. The issue was in how the test datasets were being located:

1. In `SelfMAD-main/train__.py`, the code was checking for the existence of test data using paths derived from command-line arguments:

   ```python
   for dataset_name in custom_morph_datasets:
       dataset_path = args.get(f"{dataset_name}_path")
       if dataset_path:
           # Check if test data exists for this dataset
           bonafide_test_path = os.path.join(dataset_path, "bonafide", dataset_name, "test")
           morph_test_path = os.path.join(dataset_path, "morph", dataset_name, "test")

           if os.path.exists(bonafide_test_path) and os.path.exists(morph_test_path):
               # Create test dataset
               test_dataset = TestMorphDataset(dataset_name=dataset_name, image_size=image_size)

               if len(test_dataset) > 0:
                   custom_test_datasets[dataset_name] = {"test": test_dataset}
   ```

2. However, the `TestMorphDataset` class has its own path determination logic that checks multiple possible locations for the datasets:

   ```python
   # Try multiple possible locations for datasets
   possible_base_dirs = [
       os.environ.get("DATASETS_DIR"),  # First check environment variable
       os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"),  # Check relative to script
       os.path.join(".", "datasets"),  # Check in current directory
       os.path.join("..", "datasets"),  # Check one level up
       "/cluster/home/aminurrs/SelfMAD_Combined/datasets",  # Check specific cloud path
       os.path.abspath("datasets")  # Check absolute path
   ]

   # Use the first directory that exists, or default to the second option
   project_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                   os.path.join(project_root_guess, "datasets"))
   ```

3. The issue was that the path check in `SelfMAD-main/train__.py` was failing because it was using a different path than the one being used by `TestMorphDataset`. The `TestMorphDataset` class was finding the datasets correctly, but the check in `SelfMAD-main/train__.py` was failing.

4. This discrepancy caused the system to think that no custom test datasets were available, even though they were actually present and could be loaded by the `TestMorphDataset` class.

## Solution

The solution was to modify `SelfMAD-main/train__.py` to use the same path determination logic as the `TestMorphDataset` class:

```python
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
```

Key improvements in the solution:

1. Removed the explicit path check and relied on `TestMorphDataset`'s built-in path determination logic
2. Added more informative debug output to show which datasets were successfully loaded and how many samples they contain
3. Added a specific warning message for datasets that don't have any samples

## Implementation Details

The changes were made to the `SelfMAD-main/train__.py` file, specifically in the section that loads the custom test datasets (around lines 308-320).

The modified code now:
1. Creates the `TestMorphDataset` directly without checking for the existence of test data first
2. Relies on `TestMorphDataset`'s built-in path determination logic to find the datasets
3. Checks if the dataset has any samples and adds it to the `custom_test_datasets` dictionary if it does
4. Provides better debug output to help diagnose any remaining issues

## Expected Outcome

With these changes, the system should now be able to find the custom test datasets, and the warning message should no longer appear. The `testset_best` saving strategy should now work correctly, using the actual test dataset results instead of default values.

## Verification Steps

To verify that the issue has been fixed:

1. Run the training script with the `testset_best` saving strategy
2. Check if the warning message "No custom test datasets available" still appears
3. Verify that the system is using the correct dataset paths by checking the debug output
4. Confirm that the `testset_best` saving strategy is working correctly by examining the saved model weights

## Additional Recommendations

1. Consider adding more robust error handling and logging throughout the codebase to make it easier to diagnose similar issues in the future
2. Add validation checks to ensure that the dataset paths are valid and that the required directory structure exists
3. Consider adding unit tests to verify that the dataset loading code works correctly with different path configurations
4. Standardize the path determination logic across the codebase to avoid similar issues in the future
