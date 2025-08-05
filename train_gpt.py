import os
import sys
import pandas as pd
from prostate158.train import SegmentationTrainer
from prostate158.utils import load_config
import logging
import time
import monai
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from prostate158.data import segmentation_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Distributed utilities
# -----------------------------
# def setup_distributed():
#     """Initialize torch.distributed."""
#     dist.init_process_group(
#         backend="nccl" if torch.cuda.is_available() else "gloo",
#         init_method="env://"
#     )
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     return local_rank

# def cleanup_distributed():
#     dist.destroy_process_group()

# def is_main_process():
#     return int(os.environ.get("RANK", 0)) == 0

# -----------------------------
# Argparse
# -----------------------------
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
#     parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
#     parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for torch.distributed")
#     parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for logs/checkpoints")
#     parser.add_argument("--pretrained_model", type=str, default="/opt/ml/input/data/models/tumor.pt", help="Path to pretrained weights")
#     return parser.parse_args()

# -----------------------------
# Main training function
# -----------------------------
def main():
    # Load config
    training_config_dir = os.environ.get("SM_CHANNEL_TRAINING_CONFIG", "/opt/ml/input/data/training_config") #follow input channel
    config_path = os.path.join(training_config_dir, "tumor.yaml")
    config = load_config(config_path)
    monai.utils.set_determinism(seed=config.seed)


    # Prepare directories from SageMaker env vars or fallback
    model_dir = config.get("model_dir", "/opt/ml/input/data/models") #following input model channel
    output_dir = os.environ.get("SM_MODEL_DIR", config.get("out_dir", "/opt/ml/output"))
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", config.get("data_dir", "/opt/ml/input/data/training")) #follow input training channel
    config.model_dir = model_dir
    config.out_dir = output_dir
    config.data.data_dir = data_dir

    # Determine if we are running distributed
    # distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    # if distributed:
    #     local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", 0))
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ["WORLD_SIZE"])
    #     device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    #     setup_distributed()
    # else:
    #     rank = 0
    #     world_size = 1
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("torch version:", torch.__version__)
    print("torch file: ", torch.__file__)
    print("CUDA available?", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

    logging.info(
        f"""
        Running supervised segmentation training
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
     # Build loaders
    loaders = segmentation_dataloaders(
        config=config,
        train=True,
        valid=True,
        # rank=rank,
        # world_size=world_size
    )

    # Trainer
    # trainer = SegmentationTrainer(
    #     config=config,
    #     progress_bar=is_main_process() if distributed else True,
    #     early_stopping=is_main_process() if distributed else True,
    #     save_latest_metrics=is_main_process() if distributed else True,
    #     metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
    #     train_sampler=loaders.train.sampler if hasattr(loaders, "train") else None,
    #     train_loader=loaders.train if hasattr(loaders, "train") else loaders
    # )
    trainer = SegmentationTrainer(
        progress_bar=True,
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=True,
        config=config,
    )

    # Load pretrained weights
    # if os.path.exists(args.pretrained_model):
    #     state_dict = torch.load(args.pretrained_model, map_location=device)
    #     if isinstance(state_dict, dict) and "state_dict" in state_dict:
    #         state_dict = state_dict["state_dict"]
    #     trainer.network.load_state_dict(state_dict, strict=False)
    # else:
    #     logger.warning(f"No pretrained model found at {args.pretrained_model}")
    model_path = "/opt/ml/input/data/models/tumor.pt"
    #load pretrained weights from SageMaker model channel
    state_dict = torch.load(
        # see model channel in aws_train.ipynb
        model_path,
        map_location=trainer.config.device
    )
    # Handle both cases: direct state dict or wrapped in 'state_dict' key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    trainer.network.load_state_dict(state_dict, strict=False)

    # Wrap with DDP if needed
    # if distributed:
    #     trainer.network = DDP(trainer.network, device_ids=[local_rank])
    # else:
    #     trainer.network = trainer.network.to(device)

    # One-cycle LR scheduler
    trainer.fit_one_cycle()

    # Run training loop
    start = time.time()
    trainer.run()
    end = time.time()
    logger.info(f"Training completed in {(end - start)/60:.2f} minutes")

    # Save final metrics and losses CSVs to output_dir
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(trainer.metrics).to_csv(os.path.join(output_dir, "metric_logs.csv"))
    pd.DataFrame(trainer.loss).to_csv(os.path.join(output_dir, "loss_logs.csv"))

    # if is_main_process() or not distributed:
    #     logger.info(f"Training completed in {(end - start) / 60:.2f} minutes")

    #     os.makedirs(args.output_dir, exist_ok=True)
    #     pd.DataFrame(trainer.metrics).to_csv(os.path.join(args.output_dir, "metric_logs.csv"))
    #     pd.DataFrame(trainer.loss).to_csv(os.path.join(args.output_dir, "loss_logs.csv"))

    # Cleanup DDP
    # if distributed:
    #     cleanup_distributed()

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # args = parse_args()
    try:
        # main(args)
        main()
    except Exception as e:
        logger.exception("Training failed with error:")
        sys.exit(1)