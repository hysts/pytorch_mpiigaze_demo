from typing import Any

import cv2
import torchvision.transforms as T
import yacs.config

from .types import GazeEstimationMethod


def create_transform(config: yacs.config.CfgNode) -> Any:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        return T.ToTensor()
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        return _create_mpiifacegaze_transform(config)
    else:
        raise ValueError


def _create_mpiifacegaze_transform(config: yacs.config.CfgNode) -> Any:
    size = config.transform.mpiifacegaze_face_size
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, (size, size))),
        T.ToTensor(),
        T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
    ])
    return transform
