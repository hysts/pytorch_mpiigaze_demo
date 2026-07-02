from collections.abc import Callable

import cv2
import numpy as np
import torch
import torchvision.transforms as T  # noqa: N812
from omegaconf import DictConfig

from .mode import GazeEstimationMode


def create_transform(config: DictConfig) -> Callable[[np.ndarray], torch.Tensor]:
    mode = GazeEstimationMode(config.mode)
    if mode is GazeEstimationMode.MPIIGAZE:
        return T.ToTensor()
    if mode is GazeEstimationMode.MPIIFACEGAZE:
        return _create_mpiifacegaze_transform(config)
    return _create_ethxgaze_transform(config)


def _create_mpiifacegaze_transform(config: DictConfig) -> Callable:
    size = tuple(config.gaze_estimator.image_size)
    return T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),  # BGR
        ]
    )


def _create_ethxgaze_transform(config: DictConfig) -> Callable:
    size = tuple(config.gaze_estimator.image_size)
    return T.Compose(
        [
            T.Lambda(lambda x: cv2.resize(x, size)),
            T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
        ]
    )
