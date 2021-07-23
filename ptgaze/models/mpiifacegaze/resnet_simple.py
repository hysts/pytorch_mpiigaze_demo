from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .backbones import create_backbone


class Model(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.feature_extractor = create_backbone(config)
        n_channels = self.feature_extractor.n_features

        self.conv = nn.Conv2d(n_channels,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        # This model assumes the input image size is 224x224.
        self.fc = nn.Linear(n_channels * 14**2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
