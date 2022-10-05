import importlib
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import create_wandb_config, load_checkpoint, get_logger
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.datasets.utils import get_test_loaders


def train():
    logger = get_logger("TrainingSetup")

    # Load and log experiment configuration
    config = load_config(config_type="train")
    os.environ["WANDB_DIR"] = config["trainer"]["checkpoint_dir"]
    Path(config["trainer"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    wandb_project, wandb_name, wandb_notes, wandb_config = create_wandb_config(config)
    wandb.init(
        project=wandb_project, name=wandb_name, notes=wandb_notes, config=wandb_config
    )
    logger.info((wandb_project, wandb_name))
    logger.info(config)

    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        logger.info(f"Seed the RNG for all devices with {manual_seed}")
        logger.warning(
            "Using CuDNN deterministic setting. This may slow down the training!"
        )
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # create trainer
    trainer = create_trainer(config)
    # Start training
    trainer.fit()
    return config["trainer"]["checkpoint_dir"]


def _get_predictor(model, output_dir, config):
    predictor_config = config.get("predictor", {})
    class_name = predictor_config.get("name", "StandardPredictor")

    m = importlib.import_module("pytorch3dunet.unet3d.predictor")
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)


def predict(model_path_dir):
    model_path_dir = Path(model_path_dir).absolute()
    logger = get_logger("UNet3DPredict")

    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config["model"])

    # Load model state
    # model_path = config['model_path']
    model_path = model_path_dir / 'best_checkpoint.pytorch'
    logger.info(f"Loading model from {model_path}...")
    load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config["device"]
    if torch.cuda.device_count() > 1 and not device.type == "cpu":
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for prediction")

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    # output_dir = config["loaders"].get("output_dir", None)
    output_dir = model_path_dir.parent / (model_path_dir.stem + "_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving predictions to: {output_dir}")

    # create predictor instance
    predictor = _get_predictor(model, output_dir, config)

    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)


def main():
    model_path_dir = train()
    predict(model_path_dir)


if __name__ == "__main__":
    main()
