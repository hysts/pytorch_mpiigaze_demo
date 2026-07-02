import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class HeadPose:
    """Head pose in the camera coordinate system."""

    rot: Rotation
    position: np.ndarray
    model3d: np.ndarray


@dataclasses.dataclass(frozen=True)
class FaceCenters:
    """3D centers of the face and eyes in the camera coordinate system."""

    face: np.ndarray
    reye: np.ndarray
    leye: np.ndarray


@dataclasses.dataclass(frozen=True)
class NormalizedPatch:
    """Result of the head pose normalization of a face or eye patch."""

    image: np.ndarray
    normalizing_rot: Rotation
    head_rot2d: np.ndarray
