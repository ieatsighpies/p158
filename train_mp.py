# # train_mp.py - Dedicated multi-GPU training script
# import os
# import argparse
# import logging
# import sys
# import time
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from pathlib import Path

# from prostate158.utils import load_config
# from prostate158.model import get_model
# from prostate158.optimizer import get_optimizer
# from prostate158.loss import get_loss
# from prostate158.transforms import get_val_post_transforms
# from prostate158.data import segmentation_dataloaders
# from prostate158.report import ReportGenerator

# import monai
# from monai.engines import SupervisedTrainer, SupervisedEvaluator
# from monai.handlers import (
#     CheckpointSaver,
#     StatsHandler,
#     TensorBoardStatsHandler,
#     ValidationHandler,
#     MeanDice,
#     EarlyStopHandler,
#     LrScheduleHandler,
#     from_engine,
# )
# from monai.inferers import SimpleInferer, SlidingWindowInferer
# from monai.data.meta_tensor import MetaTensor
# from torch.nn.parallel import DistributedDataParallel as DDP

# # Allow MetaTensor to be unpickled safely
# torch.serialization.add_safe_globals([MetaTensor])


# # For distributed training
# def setup_distributed(rank, world_size, backend="gloo"):
#     """
#     Initialize the distributed environment with platform-appropriate backend.
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     os.environ["LOCAL_RANK"] = str(rank)

#     # On Windows, use gloo backend
#     if os.name == "nt" and backend == "nccl":
#         logging.info(f"Windows detected, using gloo backend instead of nccl")
#         backend = "gloo"

#     logging.info(f"Rank {rank}: Initializing process group with {backend} backend")
#     try:
#         dist.init_process_group(backend, rank=rank, world_size=world_size)
#         logging.info(f"Rank {rank}: Process group initialized successfully")
#     except Exception as e:
#         logging.error(f"Rank {rank}: Error initializing process group: {e}")
#         raise


# def cleanup_distributed():
#     """Clean up the distributed environment."""
#     if dist.is_initialized():
#         dist.destroy_process_group()


# # def train_process(rank, world_size, config_path, args):
# #     """
# #     Training function for distributed training on a single GPU.
# #     """
# #     # Setup logging
# #     log_level = logging.INFO if rank == 0 else logging.WARNING
# #     logging.basicConfig(
# #         level=log_level,
# #         format="[%(asctime)s][%(levelname)5s](Rank {}): %(message)s".format(rank),
# #         datefmt="%Y-%m-%d %H:%M:%S",
# #     )

# #     # Initialize distributed environment
# #     setup_distributed(rank, world_size, backend=args.backend)

# #     # Load config
# #     config = load_config(config_path)
# #     monai.utils.set_determinism(seed=config.seed)

# #     # Set device for this process
# #     device = torch.device(f"cuda:{rank}")
# #     torch.cuda.set_device(device)

# #     # Update config with runtime args
# #     config.device = str(device)
# #     config.distributed = True
# #     if args.debug:
# #         config.debug = True

# #     # Create model directories
# #     if rank == 0:
# #         os.makedirs(config.run_id, exist_ok=True)
# #         os.makedirs(config.model_dir, exist_ok=True)
# #         os.makedirs(config.log_dir, exist_ok=True)
# #         os.makedirs(config.out_dir, exist_ok=True)

# #     # Create dataloaders with distributed support
# #     train_loader, val_loader = segmentation_dataloaders(
# #         config=config,
# #         train=True,
# #         valid=True,
# #         test=False,
# #         rank=rank,
# #         world_size=world_size,
# #     )

# #     # Initialize the model
# #     network = get_model(config=config).to(device)
# #     network = DDP(network, device_ids=[rank], output_device=rank)

# #     # Get optimizer and loss function
# #     optimizer = get_optimizer(network, config=config)
# #     loss_fn = get_loss(config=config)
# #     val_post_transforms = get_val_post_transforms(config=config)

# #     # Create evaluator
# #     evaluator = create_evaluator(
# #         config=config,
# #         device=device,
# #         network=network,
# #         val_loader=val_loader,
# #         val_post_transforms=val_post_transforms,
# #         rank=rank,
# #     )

# #     # Create trainer
# #     trainer = create_trainer(
# #         config=config,
# #         device=device,
# #         network=network,
# #         train_loader=train_loader,
# #         evaluator=evaluator,
# #         optimizer=optimizer,
# #         loss_fn=loss_fn,
# #         rank=rank,
# #     )

# #     # Add LR scheduler if specified in config
# #     if hasattr(config, "lr_scheduler"):
# #         if config.lr_scheduler.get("OneCycleLR", False):
# #             scheduler = torch.optim.lr_scheduler.OneCycleLR(
# #                 optimizer=optimizer,
# #                 max_lr=config.lr_scheduler.OneCycleLR.max_lr,
# #                 steps_per_epoch=len(train_loader),
# #                 epochs=config.training.max_epochs,
# #             )
# #             lr_handler = LrScheduleHandler(
# #                 lr_scheduler=scheduler, print_lr=True if rank == 0 else False
# #             )
# #             trainer.add_event_handler(
# #                 monai.engines.Events.ITERATION_COMPLETED, lr_handler
# #             )

# #     # Start training
# #     start_time = time.time()
# #     logging.info(
# #         f"Rank {rank}: Starting training for {config.training.max_epochs} epochs"
# #     )
# #     trainer.run()
# #     end_time = time.time()

# #     # Log training time
# #     if rank == 0:
# #         logging.info(f"Total training time: {end_time - start_time:.2f} seconds")

# #         # Generate report if enabled
# #         if args.generate_report:
# #             report_generator = ReportGenerator(
# #                 config.run_id, config.out_dir, config.log_dir
# #             )
# #             report_generator.generate_report()

# #     # Clean up distributed environment
# #     cleanup_distributed()
# #     logging.info(f"Rank {rank}: Training completed successfully")

# # Modified train_process function in train_mp.py


# def train_process(rank, world_size, config_path, args):
#     """
#     Training function for distributed training on a single GPU.
#     """
#     # Setup logging
#     log_level = logging.INFO if rank == 0 else logging.WARNING
#     logging.basicConfig(
#         level=log_level,
#         format="[%(asctime)s][%(levelname)5s](Rank {}): %(message)s".format(rank),
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     try:
#         # Initialize distributed environment
#         setup_distributed(rank, world_size, backend=args.backend)

#         # Load config
#         config = load_config(config_path)
#         monai.utils.set_determinism(seed=config.seed)

#         # Set device for this process
#         device = torch.device(f"cuda:{rank}")
#         torch.cuda.set_device(device)

#         # Update config with runtime args
#         config.device = str(device)
#         config.distributed = True
#         if args.debug:
#             config.debug = True

#         # Create model directories - only on rank 0
#         if rank == 0:
#             os.makedirs(config.run_id, exist_ok=True)
#             os.makedirs(config.model_dir, exist_ok=True)
#             os.makedirs(config.log_dir, exist_ok=True)
#             os.makedirs(config.out_dir, exist_ok=True)

#             # Log parameters
#             logging.info(
#                 f"""
#                 Running supervised segmentation training on {world_size} GPUs
#                 Run ID:     {config.run_id}
#                 Debug:      {config.debug}
#                 Out dir:    {config.out_dir}
#                 Model dir:  {config.model_dir}
#                 Log dir:    {config.log_dir}
#                 Images:     {config.data.image_cols}
#                 Labels:     {config.data.label_cols}
#                 Data dir:   {config.data.data_dir}
#             """
#             )

#         # Wait for a short time to ensure directories are created
#         if rank > 0:
#             time.sleep(2)

#         # Create dataloaders with distributed support
#         train_loader, val_loader = segmentation_dataloaders(
#             config=config,
#             train=True,
#             valid=True,
#             test=False,
#             rank=rank,
#             world_size=world_size,
#         )

#         # Initialize the model
#         logging.info(f"Rank {rank}: Creating model")
#         network = get_model(config=config).to(device)
#         network = DDP(network, device_ids=[rank], output_device=rank)

#         # Get optimizer and loss function
#         optimizer = get_optimizer(network, config=config)
#         loss_fn = get_loss(config=config)
#         val_post_transforms = get_val_post_transforms(config=config)

#         # # Create evaluator
#         # logging.info(f"Rank {rank}: Creating evaluator")
#         # evaluator = create_evaluator(
#         #     config=config,
#         #     device=device,
#         #     network=network,
#         #     val_loader=val_loader,
#         #     val_post_transforms=val_post_transforms,
#         #     rank=rank,
#         # )

#         # Create trainer
#         logging.info(f"Rank {rank}: Creating trainer")
#         trainer = create_trainer(
#             config=config,
#             device=device,
#             network=network,
#             train_loader=train_loader,
#             evaluator=evaluator,
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             rank=rank,
#         )

#         # Add LR scheduler if specified in config
#         if hasattr(config, "lr_scheduler"):
#             if config.lr_scheduler.get("OneCycleLR", False):
#                 scheduler = torch.optim.lr_scheduler.OneCycleLR(
#                     optimizer=optimizer,
#                     max_lr=config.lr_scheduler.OneCycleLR.max_lr,
#                     steps_per_epoch=len(train_loader),
#                     epochs=config.training.max_epochs,
#                 )
#                 lr_handler = LrScheduleHandler(
#                     lr_scheduler=scheduler, print_lr=True if rank == 0 else False
#                 )
#                 from ignite.engine import Events

#                 trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_handler)

#         # Start training
#         start_time = time.time()
#         logging.info(
#             f"Rank {rank}: Starting training for {config.training.max_epochs} epochs"
#         )
#         trainer.run()
#         end_time = time.time()

#         # Log training time
#         if rank == 0:
#             logging.info(f"Total training time: {end_time - start_time:.2f} seconds")

#             # Generate report if enabled
#             if args.generate_report:
#                 report_generator = ReportGenerator(
#                     config.run_id, config.out_dir, config.log_dir
#                 )
#                 report_generator.generate_report()

#     except Exception as e:
#         logging.error(f"Rank {rank}: Error during training: {e}")
#         import traceback

#         traceback.print_exc()
#     finally:
#         # Clean up distributed environment
#         if dist.is_initialized():
#             logging.info(f"Rank {rank}: Cleaning up distributed environment")
#             dist.destroy_process_group()


# def create_evaluator(config, device, network, val_loader, val_post_transforms, rank):
#     """Create evaluator with appropriate handlers based on rank"""
#     # Only add full validation handlers on rank 0
#     val_handlers = []

#     if rank == 0:
#         # Add TensorBoard and checkpoint handlers only on rank 0
#         val_handlers = [
#             StatsHandler(
#                 output_transform=lambda x: None,
#             ),
#             TensorBoardStatsHandler(
#                 log_dir=config.log_dir,
#                 output_transform=lambda x: None,
#             ),
#             CheckpointSaver(
#                 save_dir=config.model_dir,
#                 save_dict={"net": network},
#                 save_key_metric=True,  # Will use the default key_metric_name from evaluator
#             ),
#         ]

#     # Create evaluator
#     evaluator = SupervisedEvaluator(
#         device=device,
#         val_data_loader=val_loader,
#         network=network,
#         inferer=SlidingWindowInferer(
#             roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5
#         ),
#         postprocessing=val_post_transforms,
#         key_val_metric={
#             "val_mean_dice": MeanDice(
#                 include_background=False,
#                 output_transform=from_engine(["pred", "label"]),
#             )
#         },
#         val_handlers=val_handlers,
#         amp=config.get("amp", False),
#     )

#     # Add metrics for different structures if needed
#     for m in ["MeanDice", "HausdorffDistance", "SurfaceDistance"]:
#         if hasattr(monai.handlers, m):
#             metric = getattr(monai.handlers, m)(
#                 include_background=False,
#                 reduction="mean",
#                 output_transform=from_engine(["pred", "label"]),
#             )
#             metric.attach(evaluator, m)

#     # Add config to evaluator for handler access
#     evaluator.config = config
#     return evaluator


# def create_trainer(
#     config, device, network, train_loader, evaluator, optimizer, loss_fn, rank
# ):
#     """Create trainer with appropriate handlers based on rank"""
#     # Base handlers for all ranks
#     train_handlers = [
#         ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
#     ]

#     # Additional handlers only for rank 0
#     if rank == 0:
#         train_handlers.extend(
#             [
#                 StatsHandler(
#                     tag_name="train_loss",
#                     output_transform=from_engine(["loss"], first=True),
#                 ),
#                 TensorBoardStatsHandler(
#                     log_dir=config.log_dir,
#                     tag_name="train_loss",
#                     output_transform=from_engine(["loss"], first=True),
#                 ),
#                 # Optional early stopping
#                 EarlyStopHandler(
#                     patience=config.training.early_stopping_patience,
#                     min_delta=1e-4,
#                     score_function=lambda engine: engine.state.metrics["val_mean_dice"],
#                 ),
#             ]
#         )

#     # Create trainer
#     trainer = SupervisedTrainer(
#         device=device,
#         max_epochs=config.training.max_epochs,
#         train_data_loader=train_loader,
#         network=network,
#         optimizer=optimizer,
#         loss_function=loss_fn,
#         inferer=SimpleInferer(),
#         train_handlers=train_handlers,
#         amp=config.get("amp", False),
#     )

#     # Add config to trainer for handler access
#     trainer.config = config
#     return trainer


# def main():
#     parser = argparse.ArgumentParser(
#         description="Multi-GPU training for segmentation models"
#     )
#     parser.add_argument("--config", type=str, required=True, help="Path to config file")
#     parser.add_argument(
#         "--backend",
#         type=str,
#         default="nccl",
#         choices=["nccl", "gloo"],
#         help="Distributed backend to use (nccl recommended for Linux, gloo for Windows)",
#     )
#     parser.add_argument("--debug", action="store_true", help="Enable debug mode")
#     parser.add_argument(
#         "--force_single_gpu",
#         action="store_true",
#         help="Force training on a single GPU even if multiple are available",
#     )
#     parser.add_argument(
#         "--generate_report",
#         action="store_true",
#         help="Generate performance report after training",
#     )
#     args = parser.parse_args()

#     # Check if config file exists
#     if not os.path.exists(args.config):
#         print(f"Config file {args.config} not found!")
#         return

#     # Check CUDA availability
#     if not torch.cuda.is_available():
#         print("CUDA is not available. Cannot perform GPU training.")
#         return

#     # Get number of available GPUs
#     world_size = torch.cuda.device_count()

#     # Force single GPU if requested
#     if args.force_single_gpu:
#         print("Forcing single GPU training")
#         single_gpu_train(args.config, args)
#         return

#     if world_size < 2:
#         print("Only one GPU detected, running in single GPU mode")
#         single_gpu_train(args.config, args)
#         return

#     # Setup logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format="[%(asctime)s][%(levelname)5s]: %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     # Print GPU information
#     logging.info(f"Starting distributed training with {world_size} GPUs")
#     for i in range(world_size):
#         logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

#     # Set the multiprocessing start method
#     try:
#         mp.set_start_method("spawn", force=True)
#     except RuntimeError:
#         pass

#     # Start a process for each GPU
#     try:
#         mp.spawn(
#             train_process,
#             args=(world_size, args.config, args),
#             nprocs=world_size,
#             join=True,
#         )
#     except Exception as e:
#         logging.error(f"Error during distributed training: {e}")
#         raise


# def single_gpu_train(config_path, args):
#     """Run training on a single GPU"""
#     # Load config
#     config = load_config(config_path)

#     # Set deterministic training
#     monai.utils.set_determinism(seed=config.seed)

#     # Update config
#     config.device = "cuda:0"
#     config.distributed = False
#     if args.debug:
#         config.debug = True

#     # Create output directories
#     os.makedirs(config.run_id, exist_ok=True)
#     os.makedirs(config.model_dir, exist_ok=True)
#     os.makedirs(config.log_dir, exist_ok=True)
#     os.makedirs(config.out_dir, exist_ok=True)

#     logging.info(
#         f"""
#         Running supervised segmentation training on single GPU
#         Run ID:     {config.run_id}
#         Debug:      {config.debug}
#         Out dir:    {config.out_dir}
#         Model dir:  {config.model_dir}
#         Log dir:    {config.log_dir}
#         Images:     {config.data.image_cols}
#         Labels:     {config.data.label_cols}
#         Data dir:   {config.data.data_dir}
#         """
#     )

#     # Create dataloaders
#     train_loader, val_loader = segmentation_dataloaders(
#         config=config, train=True, valid=True, test=False
#     )

#     # Create network, optimizer, loss function
#     network = get_model(config=config).to(config.device)
#     optimizer = get_optimizer(network, config=config)
#     loss_fn = get_loss(config=config)
#     val_post_transforms = get_val_post_transforms(config=config)

#     # Create evaluator with validation handlers
#     val_handlers = [
#         StatsHandler(output_transform=lambda x: None),
#         TensorBoardStatsHandler(
#             log_dir=config.log_dir,
#             output_transform=lambda x: None,
#         ),
#         CheckpointSaver(
#             save_dir=config.model_dir,
#             save_dict={"net": network},
#             save_key_metric=True,
#             key_metric_name="val_mean_dice",
#             key_metric_mode="max",
#         ),
#     ]

#     evaluator = SupervisedEvaluator(
#         device=torch.device(config.device),
#         val_data_loader=val_loader,
#         network=network,
#         inferer=SlidingWindowInferer(
#             roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5
#         ),
#         postprocessing=val_post_transforms,
#         key_val_metric={
#             "val_mean_dice": MeanDice(
#                 include_background=False,
#                 output_transform=from_engine(["pred", "label"]),
#             )
#         },
#         val_handlers=val_handlers,
#         amp=config.get("amp", False),
#     )

#     # Add metrics
#     for m in ["MeanDice", "HausdorffDistance", "SurfaceDistance"]:
#         if hasattr(monai.handlers, m):
#             metric = getattr(monai.handlers, m)(
#                 include_background=False,
#                 reduction="mean",
#                 output_transform=from_engine(["pred", "label"]),
#             )
#             metric.attach(evaluator, m)

#     # Add config to evaluator
#     evaluator.config = config

#     # Create trainer with train handlers
#     train_handlers = [
#         ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
#         StatsHandler(
#             tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
#         ),
#         TensorBoardStatsHandler(
#             log_dir=config.log_dir,
#             tag_name="train_loss",
#             output_transform=from_engine(["loss"], first=True),
#         ),
#         EarlyStopHandler(
#             patience=config.training.early_stopping_patience,
#             min_delta=1e-4,
#             score_function=lambda engine: engine.state.metrics["val_mean_dice"],
#         ),
#     ]

#     trainer = SupervisedTrainer(
#         device=torch.device(config.device),
#         max_epochs=config.training.max_epochs,
#         train_data_loader=train_loader,
#         network=network,
#         optimizer=optimizer,
#         loss_function=loss_fn,
#         inferer=SimpleInferer(),
#         train_handlers=train_handlers,
#         amp=config.get("amp", False),
#     )

#     # Add config to trainer
#     trainer.config = config

#     # Add LR scheduler if specified in config
#     if hasattr(config, "lr_scheduler"):
#         if config.lr_scheduler.get("OneCycleLR", False):
#             scheduler = torch.optim.lr_scheduler.OneCycleLR(
#                 optimizer=optimizer,
#                 max_lr=config.lr_scheduler.OneCycleLR.max_lr,
#                 steps_per_epoch=len(train_loader),
#                 epochs=config.training.max_epochs,
#             )
#             lr_handler = LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)
#             from ignite.engine import Events

#             trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_handler)

#     # Start training
#     start_time = time.time()
#     trainer.run()
#     end_time = time.time()

#     # Log training time
#     logging.info(f"Total training time: {end_time - start_time:.2f} seconds")

#     # Evaluate final model
#     evaluator.run()

#     # Generate report if enabled
#     if args.generate_report:
#         report_generator = ReportGenerator(
#             config.run_id, config.out_dir, config.log_dir
#         )
#         report_generator.generate_report()


# if __name__ == "__main__":
#     main()


# train_mp.py - Multi-GPU training script without evaluator
import os
import argparse
import logging
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from prostate158.utils import load_config
from prostate158.train_anson import SegmentationTrainer, cleanup_distributed
from prostate158.report import ReportGenerator

import monai
from monai.data.meta_tensor import MetaTensor
from torch.nn.parallel import DistributedDataParallel as DDP

# Allow MetaTensor to be unpickled safely
torch.serialization.add_safe_globals([MetaTensor])


# For distributed training
def setup_distributed(rank, world_size, backend="gloo"):
    """
    Initialize the distributed environment with platform-appropriate backend.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)

    # On Windows, use gloo backend
    if os.name == "nt" and backend == "nccl":
        logging.info(f"Windows detected, using gloo backend instead of nccl")
        backend = "gloo"

    logging.info(f"Rank {rank}: Initializing process group with {backend} backend")
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        logging.info(f"Rank {rank}: Process group initialized successfully")
    except Exception as e:
        logging.error(f"Rank {rank}: Error initializing process group: {e}")
        raise


def train_process(rank, world_size, config_path, args):
    """
    Training function for distributed training on a single GPU using SegmentationTrainer.
    """
    # Setup logging
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(levelname)5s](Rank {}): %(message)s".format(rank),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        # Initialize distributed environment
        setup_distributed(rank, world_size, backend=args.backend)

        # Load config
        config = load_config(config_path)
        monai.utils.set_determinism(seed=config.seed)

        # Set device for this process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # Update config with runtime args
        config.device = str(device)
        config.distributed = True
        if args.debug:
            config.debug = True

        # Create model directories - only on rank 0
        if rank == 0:
            os.makedirs(config.run_id, exist_ok=True)
            os.makedirs(config.model_dir, exist_ok=True)
            os.makedirs(config.log_dir, exist_ok=True)
            os.makedirs(config.out_dir, exist_ok=True)

            # Log parameters
            logging.info(
                f"""
                Running supervised segmentation training on {world_size} GPUs
                Run ID:     {config.run_id}
                Debug:      {config.debug}
                Out dir:    {config.out_dir}
                Model dir:  {config.model_dir}
                Log dir:    {config.log_dir}
                Images:     {config.data.image_cols}
                Labels:     {config.data.label_cols}
                Data dir:   {config.data.data_dir}
            """
            )

        # Wait for a short time to ensure directories are created
        if rank > 0:
            time.sleep(2)

        # Create trainer using SegmentationTrainer
        logging.info(f"Rank {rank}: Creating SegmentationTrainer")
        trainer = SegmentationTrainer(
            config=config,
            progress_bar=(rank == 0),  # Only show progress bar on rank 0
            early_stopping=True,
            metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
            save_latest_metrics=(rank == 0),  # Only save metrics on rank 0
            rank=rank,
            world_size=world_size,
        )

        # Add LR scheduler - using your existing fit_one_cycle
        logging.info(f"Rank {rank}: Setting up LR scheduler")
        trainer.fit_one_cycle()

        # Start training
        start_time = time.time()
        logging.info(
            f"Rank {rank}: Starting training for {config.training.max_epochs} epochs"
        )
        trainer.run()
        end_time = time.time()

        # Evaluate and generate report only on rank 0
        if rank == 0:
            logging.info(f"Total training time: {end_time - start_time:.2f} seconds")

            # Evaluate
            trainer.evaluate()

            # Generate report if enabled
            if args.generate_report:
                report_generator = ReportGenerator(
                    config.run_id, config.out_dir, config.log_dir
                )
                report_generator.generate_report()

    except Exception as e:
        logging.error(f"Rank {rank}: Error during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up distributed environment
        if dist.is_initialized():
            logging.info(f"Rank {rank}: Cleaning up distributed environment")
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU training for segmentation models"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend to use (nccl recommended for Linux, gloo for Windows)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--force_single_gpu",
        action="store_true",
        help="Force training on a single GPU even if multiple are available",
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate performance report after training",
    )
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found!")
        return

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform GPU training.")
        return

    # Get number of available GPUs
    world_size = torch.cuda.device_count()

    # Force single GPU if requested
    if args.force_single_gpu:
        print("Forcing single GPU training")
        single_gpu_train(args.config, args)
        return

    if world_size < 2:
        print("Only one GPU detected, running in single GPU mode")
        single_gpu_train(args.config, args)
        return

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)5s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Print GPU information
    logging.info(f"Starting distributed training with {world_size} GPUs")
    for i in range(world_size):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Start a process for each GPU
    try:
        mp.spawn(
            train_process,
            args=(world_size, args.config, args),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        logging.error(f"Error during distributed training: {e}")
        raise


def single_gpu_train(config_path, args):
    """Run training on a single GPU using SegmentationTrainer"""
    # Load config
    config = load_config(config_path)

    # Set deterministic training
    monai.utils.set_determinism(seed=config.seed)

    # Update config
    config.device = "cuda:0"
    config.distributed = False
    if args.debug:
        config.debug = True

    # Create output directories
    os.makedirs(config.run_id, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.out_dir, exist_ok=True)

    logging.info(
        f"""
        Running supervised segmentation training on single GPU
        Run ID:     {config.run_id}
        Debug:      {config.debug}
        Out dir:    {config.out_dir}
        Model dir:  {config.model_dir}
        Log dir:    {config.log_dir}
        Images:     {config.data.image_cols}
        Labels:     {config.data.label_cols}
        Data dir:   {config.data.data_dir}
        """
    )

    # Create trainer
    trainer = SegmentationTrainer(
        progress_bar=True,
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=True,
        config=config,
    )

    # Add LR scheduler
    trainer.fit_one_cycle()

    # Run training
    start_time = time.time()
    trainer.run()
    end_time = time.time()

    # Log training time
    logging.info(f"Total training time: {end_time - start_time:.2f} seconds")

    # Evaluate
    trainer.evaluate()

    # Generate report if enabled
    if args.generate_report:
        report_generator = ReportGenerator(
            config.run_id, config.out_dir, config.log_dir
        )
        report_generator.generate_report()


if __name__ == "__main__":
    main()
