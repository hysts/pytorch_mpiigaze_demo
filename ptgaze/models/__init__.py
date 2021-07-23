import importlib

import timm
import torch
from omegaconf import DictConfig


def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    if mode in ['MPIIGaze', 'MPIIFaceGaze']:
        module = importlib.import_module(
            f'ptgaze.models.{mode.lower()}.{config.model.name}')
        model = module.Model(config)
    elif mode == 'ETH-XGaze':
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        raise ValueError
    device = torch.device(config.device)
    model.to(device)
    return model
