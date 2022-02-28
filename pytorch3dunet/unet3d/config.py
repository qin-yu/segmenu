import argparse
from datetime import datetime

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')
TIMESTAMP = '{TIMESTAMP}'


def load_config(config_type=None):
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device

    # Time-stamp the checkpoint dir
    if config_type == 'train':
        checkpoint_dir = config['trainer']['checkpoint_dir']
        if TIMESTAMP in checkpoint_dir:
            config['trainer']['checkpoint_dir'] = config['trainer']['checkpoint_dir'].replace(TIMESTAMP, datetime.now().strftime("%Y%m%d%H%M%S"))

    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
