import monai
import argparse
import torch
import torch.multiprocessing as mp

from prostate158.utils import load_config
from prostate158.train_anson import (
    SegmentationTrainer,
    setup_distributed,
    cleanup_distributed,
)
from prostate158.report import ReportGenerator


def train_single_gpu(config):
    """
    Training on a single GPU
    """
    monai.utils.set_determinism(seed=config.seed)

    print(
        f"""
        Running supervised segmentation training on single GPU
        Run ID:     {config.run_id}
        Debug:      {config.debug}
        Out dir:    {config.out_dir}
        model dir:  {config.model_dir}
        log dir:    {config.log_dir}
        images:     {config.data.image_cols}
        labels:     {config.data.label_cols}
        data_dir    {config.data.data_dir}
        """
    )

    # create supervised trainer for segmentation task
    trainer = SegmentationTrainer(
        progress_bar=True,
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=True,
        config=config,
    )

    ## add lr scheduler to trainer
    trainer.fit_one_cycle()

    ## let's train
    trainer.run()

    ## finish script with final evaluation of the best model
    trainer.evaluate()

    ## generate a markdown document with segmentation results
    report_generator = ReportGenerator(config.run_id, config.out_dir, config.log_dir)
    report_generator.generate_report()


def train_ddp(rank, world_size, config_path):
    """
    Training function for distributed training
    """
    # Setup the distributed environment
    setup_distributed(rank, world_size)

    # Load the configuration
    config = load_config(config_path)
    monai.utils.set_determinism(seed=config.seed)

    # Only print on main process
    if rank == 0:
        print(
            f"""
            Running supervised segmentation training on {world_size} GPUs
            Run ID:     {config.run_id}
            Debug:      {config.debug}
            Out dir:    {config.out_dir}
            model dir:  {config.model_dir}
            log dir:    {config.log_dir}
            images:     {config.data.image_cols}
            labels:     {config.data.label_cols}
            data_dir    {config.data.data_dir}
            """
        )

    # Create trainer with distributed support
    trainer = SegmentationTrainer(
        progress_bar=(rank == 0),  # Only show progress bar on rank 0
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=(rank == 0),  # Only save metrics on rank 0
        config=config,
        rank=rank,
        world_size=world_size,
    )

    # Add LR scheduler
    trainer.fit_one_cycle()

    # Train the model
    trainer.run()

    # Evaluate and generate report only on rank 0
    if rank == 0:
        trainer.evaluate()
        report_generator = ReportGenerator(
            config.run_id, config.out_dir, config.log_dir
        )
        report_generator.generate_report()

    # Clean up distributed environment
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--config", type=str, required=True, help="path to the config file"
    )
    args = parser.parse_args()
    config_fn = args.config

    # Load the configuration
    config = load_config(config_fn)

    # Check if distributed training is enabled
    is_distributed = hasattr(config, "distributed") and config.distributed

    # Check available GPUs
    gpu_count = torch.cuda.device_count()

    if is_distributed and gpu_count > 1:
        print(f"Starting distributed training with {gpu_count} GPUs")
        # Spawn a process for each GPU
        mp.spawn(train_ddp, args=(gpu_count, config_fn), nprocs=gpu_count, join=True)
    else:
        if is_distributed and gpu_count <= 1:
            print(
                "Distributed training was enabled but only one GPU is available. Falling back to single GPU training."
            )
            config.distributed = False

        # Standard single-GPU training
        train_single_gpu(config)


if __name__ == "__main__":
    main()
