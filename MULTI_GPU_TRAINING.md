# Multi-GPU Training with DistributedDataParallel

This document explains how to use both NVIDIA RTX 5000 GPUs for training your segmentation model using the integrated distributed training implementation.

## Overview

The implementation uses PyTorch's Distributed Data Parallel (DDP) to enable training across multiple GPUs while ensuring `persistent_workers` is set to `False` as required.

## Key Features

- Uses all available GPUs efficiently through PyTorch's DDP
- Maintains same dataset behavior with `persistent_workers=False`
- Works with the existing codebase without requiring separate scripts
- Automatically detects and adapts to the number of available GPUs
- Handles all synchronization between processes
- Falls back to single-GPU training when only one GPU is available

## How to Use

Simply set `distributed: True` in your YAML configuration file and run the standard training script:

```bash
python train.py --config tumor.yaml
```

The script will:
1. Detect that distributed training is enabled
2. Count the available GPUs (both NVIDIA RTX 5000s)
3. Spawn processes for each GPU
4. Distribute the training across both GPUs

## Implementation Details

The distributed implementation:

1. Uses the standard `torch.distributed` package for multi-GPU training
2. Wraps models in `DistributedDataParallel` for gradient synchronization
3. Uses distributed samplers to split data correctly across GPUs
4. Ensures only the main process (rank 0) handles logging, reporting, and checkpointing
5. Properly synchronizes all processes at critical points with barriers

## Configuration

The existing configuration files can be used by simply adding:

```yaml
distributed: True
```

For optimal performance, you might want to:

1. Reduce the batch size per GPU (the effective batch size will be batch_size × num_gpus)
2. Set number of workers per GPU:

```yaml
data:
  batch_size: 2  # Per GPU (effective batch size = 4 with 2 GPUs)
  num_workers: 8  # Per GPU
```

## Performance Notes

- Training should be approximately 1.8-1.9× faster with two GPUs
- Memory usage is split across the GPUs
- Synchronization overhead is minimal with just two GPUs

## Troubleshooting

If you encounter any issues:

1. Make sure both GPUs are detected with `nvidia-smi`
2. Verify both GPUs have enough free memory
3. Set `distributed: False` to fall back to single-GPU training

## Implementation Changes

The changes were made to the following files:

1. `prostate158/train_anson.py`: Added distributed training support to the `SegmentationTrainer` class
2. `prostate158/data.py`: Added distributed sampling for data loading
3. `train.py`: Modified to support both distributed and single-GPU modes

These changes maintain backward compatibility, so existing single-GPU training still works. 