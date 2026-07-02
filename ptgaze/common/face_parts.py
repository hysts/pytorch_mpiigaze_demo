import enum
from typing import TYPE_CHECKING

import numpy as np

from . import gaze

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


class FacePartsName(enum.Enum):
    FACE = enum.auto()
    REYE = enum.auto()
    LEYE = enum.auto()


class FaceParts:
    def __init__(self, name: FacePartsName) -> None:
        self.name = name
        self.center: np.ndarray | None = None
        self.head_pose_rot: Rotation | None = None
        self.normalizing_rot: Rotation | None = None
        self.normalized_head_rot2d: np.ndarray | None = None
        self.normalized_image: np.ndarray | None = None

        self.normalized_gaze_angles: np.ndarray | None = None
        self.normalized_gaze_vector: np.ndarray | None = None
        self.gaze_vector: np.ndarray | None = None

    @property
    def distance(self) -> float:
        if self.center is None:
            raise ValueError("center is not set")
        return float(np.linalg.norm(self.center))

    def angle_to_vector(self) -> None:
        if self.normalized_gaze_angles is None:
            raise ValueError("normalized_gaze_angles is not set")
        self.normalized_gaze_vector = gaze.gaze_angles_to_vector(self.normalized_gaze_angles)

    def denormalize_gaze_vector(self) -> None:
        if self.normalizing_rot is None:
            raise ValueError("normalizing_rot is not set")
        if self.normalized_gaze_vector is None:
            raise ValueError("normalized_gaze_vector is not set")
        self.gaze_vector = gaze.denormalize_gaze_vector(self.normalized_gaze_vector, self.normalizing_rot)

    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        return gaze.gaze_vector_to_angles(vector)
