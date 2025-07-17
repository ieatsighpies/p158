import os
import sys
import yaml
import pandas as pd
from prostate158.train import SegmentationTrainer
from prostate158.utils import load_config
import logging
import time
from pathlib import Path
from munch import Munch
import monai
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_munch(config):
    return config if isinstance(config, Munch) else Munch.fromDict(config)

def validate_config(config: Munch):
    required_keys = [
        "model_dir",
        "out_dir",
        "data",
        "data.train_csv",
        "data.valid_csv",
        "data.image_cols",
        "data.label_cols",
    ]

    for key in required_keys:
        parts = key.split(".")
        ref = config
        for part in parts:
            if not hasattr(ref, part):
                raise ValueError(f"Missing required config key: {key}")
            ref = getattr(ref, part)

def main():
    # Load config
    config_path = os.environ.get("SM_CHANNEL_TRAINING_CONFIG", "tumor.yaml")
    config = load_config(config_path)
    monai.utils.set_determinism(seed=config.seed)


    # Prepare directories from SageMaker env vars or fallback
    model_dir = os.environ.get("SM_MODEL_DIR", config.get("model_dir", "./opt/ml/input/data/model/")) #following input model channel
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", config.get("out_dir", "./opt/ml/output"))
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", config.get("data_dir", "./opt/ml/data/"))
    config.model_dir = model_dir
    config.out_dir = output_dir
    config.data.data_dir = data_dir

    print("torch version:", torch.__version__)
    print("torch file: ", torch.__file__)
    print("CUDA available?", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

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
    # Create trainer instance
    trainer = SegmentationTrainer(
        progress_bar=True,
        early_stopping=True,
        metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
        save_latest_metrics=True,
        config=config,
    )

    #load pretrained weights from SageMaker model channel
    state_dict = torch.load(
        # see model channel in aws_train.ipynb
        os.environ.get("SM_CHANNEL_MODEL"),
        map_location=trainer.config.device
    )
    # Handle both cases: direct state dict or wrapped in 'state_dict' key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    trainer.network.load_state_dict(state_dict, strict=False)

    # Attach OneCycle scheduler
    trainer.fit_one_cycle()

    # Run training loop (default 5 epochs or from config)
    start = time.time()
    trainer.run()
    end = time.time()
    logger.info(f"Training completed in {(end - start)/60:.2f} minutes")

    # Save final metrics and losses CSVs to output_dir
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(trainer.metrics).to_csv(os.path.join(output_dir, "metric_logs.csv"))
    pd.DataFrame(trainer.loss).to_csv(os.path.join(output_dir, "loss_logs.csv"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with error:")
        sys.exit(1)
