import os
import random
from pathlib import Path

import torch
import wandb

from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import create_wandb_config
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config = load_config(config_type='train')
    os.environ['WANDB_DIR'] = config['trainer']['checkpoint_dir']
    Path(config['trainer']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    wandb_project, wandb_name, wandb_notes, wandb_config = create_wandb_config(config)
    wandb.init(
        project=wandb_project, 
        name=wandb_name, 
        notes=wandb_notes,
        config=wandb_config
    )
    logger.info((wandb_project, wandb_name))
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # create trainer
    trainer = create_trainer(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
