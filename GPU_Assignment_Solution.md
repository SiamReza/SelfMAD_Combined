# GPU Assignment Solution for SelfMAD

This document describes the solution to the GPU assignment issues in the SelfMAD project.

## Issues Identified

We identified two main issues with the GPU assignment in the SelfMAD project:

1. **CUDA Out of Memory Error**: When using DataParallel with specific GPU IDs, PyTorch first loads the entire model onto the first GPU in the device_ids list before distributing it. This was causing an out of memory error when trying to load the model onto GPU 0, which was already being used by SelfMAD-siam.

2. **DataParallel Optimizer Access Error**: When a model is wrapped with DataParallel, the original model becomes accessible through the `.module` attribute, but the DataParallel wrapper itself doesn't have the attributes of the original model (like 'optimizer'). This was causing an error when trying to access `model.optimizer` directly.

3. **GPU Assignment Issue**: Despite fixing the above issues, only GPU 0 was being used. This might be due to issues with how DataParallel is handling the device_ids parameter or how the GPU IDs are being passed from run_experiments_direct.py to the training script.

## Solution

The simplest and most reliable solution is to disable DataParallel and use CUDA_VISIBLE_DEVICES to control which GPUs are used by each process:

1. Set `USE_MULTI_GPU = False` in automation_config.py
2. Keep `RUN_PARALLEL = True`
3. Set `SIAM_GPU_ID = 0` and `MAIN_GPU_ID = 1`

This configuration ensures that:
- SelfMAD-siam uses GPU 0
- SelfMAD-main uses GPU 1
- Each model has exclusive access to its assigned GPU
- No DataParallel is used, avoiding the issues with device_ids and optimizer access

### Changes Made

```python
# In automation_config.py
USE_MULTI_GPU = False  # Set to False to use CUDA_VISIBLE_DEVICES instead of DataParallel
RUN_PARALLEL = True    # Set to True to run SelfMAD-siam and SelfMAD-main in parallel
SIAM_GPU_ID = 0        # GPU ID to use for SelfMAD-siam
MAIN_GPU_ID = 1        # GPU ID to use for SelfMAD-main
```

## How It Works

When `RUN_PARALLEL = True` and `USE_MULTI_GPU = False`, the run_experiments_direct.py script sets the CUDA_VISIBLE_DEVICES environment variable before launching each process:

```python
# Traditional approach: set CUDA_VISIBLE_DEVICES to restrict visible GPUs
if model_type == "siam":
    gpu_id = getattr(args, "siam_gpu_id", 0)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for SelfMAD-siam")
else:  # main
    gpu_id = getattr(args, "main_gpu_id", 1)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for SelfMAD-main")
```

This ensures that each process only sees its assigned GPU, and when it calls `torch.device('cuda')`, it will use that GPU.

## Benefits

1. **Simplicity**: This approach is simpler and more reliable than using DataParallel with specific GPU IDs.
2. **Resource Isolation**: Each model has exclusive access to its assigned GPU, preventing resource contention.
3. **Compatibility**: This approach works with any PyTorch model without requiring changes to the model code.
4. **Efficiency**: Each GPU is fully utilized by a single model, maximizing performance.

## When to Use DataParallel

DataParallel is useful when you want to train a single model on multiple GPUs to handle larger batch sizes or more complex models. However, in this case, we're running two separate models (SelfMAD-siam and SelfMAD-main) in parallel, so it's more efficient to assign each model to a separate GPU using CUDA_VISIBLE_DEVICES.

If you do want to use DataParallel in the future, make sure to:
1. Set `USE_MULTI_GPU = True` in automation_config.py
2. Ensure that the GPU IDs are correctly passed to the training script
3. Use the pattern `optimizer = original_model.optimizer if isinstance(model, nn.DataParallel) else model.optimizer` to access the optimizer
