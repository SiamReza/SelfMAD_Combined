# DataParallel Fixes for SelfMAD

This document describes the fixes implemented to resolve issues with DataParallel in the SelfMAD project.

## Issues Fixed

### 1. CUDA Out of Memory Error

**Problem**: When using DataParallel with specific GPU IDs, PyTorch first loads the entire model onto the first GPU in the device_ids list before distributing it. This was causing an out of memory error when trying to load the model onto GPU 0, which was already being used by SelfMAD-siam.

**Solution**: Changed the GPU assignment in `automation_config.py` to prioritize GPU 1 for SelfMAD-main:

```python
# Before:
MAIN_GPU_IDS = [0, 1] # List of GPU IDs to use for SelfMAD-main with DataParallel

# After:
MAIN_GPU_IDS = [1, 0] # List of GPU IDs to use for SelfMAD-main with DataParallel (using both GPUs, starting with GPU 1)
```

This change ensures that the SelfMAD-main model is first loaded onto GPU 1 before being distributed, preventing the out of memory error since GPU 1 won't be sharing resources with SelfMAD-siam (which is using GPU 0).

### 2. DataParallel Optimizer Access Error

**Problem**: When a model is wrapped with DataParallel, the original model becomes accessible through the `.module` attribute, but the DataParallel wrapper itself doesn't have the attributes of the original model (like 'optimizer'). This was causing an error when trying to access `model.optimizer` directly.

**Solution**: Modified all instances in `SelfMAD-main/train__.py` where `model.optimizer` was accessed directly to use the following pattern:

```python
# Access optimizer through original_model if using DataParallel
optimizer = original_model.optimizer if isinstance(model, nn.DataParallel) else model.optimizer
```

Specific changes:

1. Fixed training metrics calculation:
```python
# Before:
'lr': model.optimizer.param_groups[0]['lr'],

# After:
# Access optimizer through original_model if using DataParallel
if isinstance(model, nn.DataParallel):
    current_lr = original_model.optimizer.param_groups[0]['lr']
else:
    current_lr = model.optimizer.param_groups[0]['lr']
'lr': current_lr,
```

2. Fixed model saving in multiple places:
```python
# Before:
"optimizer": model.optimizer.state_dict(),

# After:
# Access optimizer through original_model if using DataParallel
optimizer = original_model.optimizer if isinstance(model, nn.DataParallel) else model.optimizer
"optimizer": optimizer.state_dict(),
```

## Benefits

1. **Memory Efficiency**: By changing the GPU assignment, we ensure that each GPU is used optimally without overloading any single GPU.

2. **Compatibility with DataParallel**: The fixes ensure that the code works correctly with DataParallel, allowing for multi-GPU training.

3. **Consistent Access Pattern**: Established a consistent pattern for accessing the optimizer when using DataParallel, making the code more maintainable.

## Usage

No changes are required to the way you run the code. The fixes are transparent to the user and will automatically handle the correct GPU assignment and optimizer access when DataParallel is enabled.

To enable DataParallel, make sure the following settings are in your `automation_config.py`:

```python
USE_MULTI_GPU = True  # Set to True to use DataParallel for multi-GPU training
RUN_PARALLEL = True   # Set to True to run SelfMAD-siam and SelfMAD-main in parallel
SIAM_GPU_IDS = [0]    # List of GPU IDs to use for SelfMAD-siam with DataParallel
MAIN_GPU_IDS = [1, 0] # List of GPU IDs to use for SelfMAD-main with DataParallel (using both GPUs, starting with GPU 1)
```
