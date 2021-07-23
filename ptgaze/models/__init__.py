import importlib

import timm
import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    dataset_name = config.mode.lower()
    if dataset_name in ['mpiigaze', 'mpiifacegaze']:
        module = importlib.import_module(
            f'ptgaze.models.{dataset_name}.{config.model.name}')
        model = module.Model(config)
    elif dataset_name == 'ethxgaze':
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        raise ValueError
    device = torch.device(config.device)
    model.to(device)
    return model
