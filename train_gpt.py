import os
import sys
import yaml
import pandas as pd
from prostate158.train import SegmentationTrainer
import logging
import time
from munch import Munch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    # Load config yaml
    config_path = os.environ.get("SM_CHANNEL_TRAINING_CONFIG", "tumor.yaml")
    logger.info("Loading config from: %s", config_path)

    with open(config_path, "r") as f:
        config = Munch.fromDict(yaml.safe_load(f))


    # Prepare directories from SageMaker env vars or fallback
    model_dir = os.environ.get("SM_MODEL_DIR", config.get("model_dir", "./models"))
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", config.get("out_dir", "./outputs"))
    # data_dir = os.environ.get("SM_CHANNEL_TRAIN", config.get("data_dir", "./data/prostate158_train/train"))
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", config.get("data_dir", "./opt/ml/data/"))
    config["model_dir"] = model_dir
    config["out_dir"] = output_dir
    config["data_dir"] = "/opt/ml/data/"

    # Create trainer instance
    trainer = SegmentationTrainer(config)

    # Attach OneCycle scheduler
    trainer.fit_one_cycle()

    # Run training loop (default 5 epochs or from config)
    max_epochs = config.get("max_epochs", 5)
    trainer.run(try_resume_from_checkpoint=True, max_epochs=max_epochs)
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
