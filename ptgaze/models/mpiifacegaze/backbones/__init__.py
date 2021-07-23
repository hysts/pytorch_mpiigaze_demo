import importlib

import torch.nn as nn
from omegaconf import DictConfig


def create_backbone(config: DictConfig) -> nn.Module:
    backbone_name = config.model.backbone.name
    module = importlib.import_module(
        f'ptgaze.models.mpiifacegaze.backbones.{backbone_name}')
    return module.Model(config)
