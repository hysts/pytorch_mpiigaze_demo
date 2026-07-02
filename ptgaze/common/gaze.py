import numpy as np
from scipy.spatial.transform import Rotation

DIM_3D = 3


def gaze_angles_to_vector(angles: np.ndarray) -> np.ndarray:
    pitch, yaw = angles
    return -np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def gaze_vector_to_angles(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (DIM_3D,):
        error_msg = f"vector must have {DIM_3D} columns, but got {vector.shape[0]}"
        raise ValueError(error_msg)
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])


def denormalize_gaze_vector(gaze_vector: np.ndarray, normalizing_rot: Rotation) -> np.ndarray:
    # Here the gaze vector is a row vector, and rotation matrices are
    # orthogonal, so multiplying the rotation matrix from the right is
    # the same as multiplying the inverse of the rotation matrix to the
    # column gaze vector from the left.
    return gaze_vector @ normalizing_rot.as_matrix()
