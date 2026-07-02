import importlib

import timm
import torch
from omegaconf import DictConfig

from ptgaze.mode import GazeEstimationMode


def create_model(config: DictConfig) -> torch.nn.Module:
    mode = GazeEstimationMode(config.mode)
    if mode is GazeEstimationMode.ETH_XGAZE:
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        module = importlib.import_module(f"ptgaze.models.{mode.value.lower()}.{config.model.name}")
        model = module.Model(config)
    device = torch.device(config.device)
    model.to(device)
    return model
