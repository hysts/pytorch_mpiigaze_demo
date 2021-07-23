from typing import Any

import cv2
import torchvision.transforms as T
from omegaconf import DictConfig


def create_transform(config: DictConfig) -> Any:
    if config.mode == 'MPIIGaze':
        return T.ToTensor()
    elif config.mode == 'MPIIFaceGaze':
        return _create_mpiifacegaze_transform(config)
    elif config.mode == 'ETH-XGaze':
        return _create_ethxgaze_transform(config)
    else:
        raise ValueError


def _create_mpiifacegaze_transform(config: DictConfig) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224,
                                                     0.229]),  # BGR
    ])
    return transform


def _create_ethxgaze_transform(config: DictConfig) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, size)),
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                     0.225]),  # RGB
    ])
    return transform
