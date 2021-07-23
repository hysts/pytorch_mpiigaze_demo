import dataclasses

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face


@dataclasses.dataclass(frozen=True)
class FaceModel:
    LANDMARKS: np.ndarray
    REYE_INDICES: np.ndarray
    LEYE_INDICES: np.ndarray
    MOUTH_INDICES: np.ndarray
    NOSE_INDICES: np.ndarray
    CHIN_INDEX: int
    NOSE_INDEX: int

    def estimate_head_pose(self, face: Face, camera: Camera) -> None:
        """Estimate the head pose by fitting 3D template model."""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        rvec = np.zeros(3, dtype=np.float)
        tvec = np.array([0, 0, 1], dtype=np.float)
        _, rvec, tvec = cv2.solvePnP(self.LANDMARKS,
                                     face.landmarks,
                                     camera.camera_matrix,
                                     camera.dist_coefficients,
                                     rvec,
                                     tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        rot = Rotation.from_rotvec(rvec)
        face.head_pose_rot = rot
        face.head_position = tvec
        face.reye.head_pose_rot = rot
        face.leye.head_pose_rot = rot

    def compute_3d_pose(self, face: Face) -> None:
        """Compute the transformed model."""
        rot = face.head_pose_rot.as_matrix()
        face.model3d = self.LANDMARKS @ rot.T + face.head_position

    def compute_face_eye_centers(self, face: Face, mode: str) -> None:
        """Compute the centers of the face and eyes.

        In the case of MPIIFaceGaze, the face center is defined as the
        average coordinates of the six points at the corners of both
        eyes and the mouth. In the case of ETH-XGaze, it's defined as
        the average coordinates of the six points at the corners of both
        eyes and the nose. The eye centers are defined as the average
        coordinates of the corners of each eye.
        """
        if mode == 'ETH-XGaze':
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.NOSE_INDICES])].mean(axis=0)
        else:
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.MOUTH_INDICES])].mean(axis=0)
        face.reye.center = face.model3d[self.REYE_INDICES].mean(axis=0)
        face.leye.center = face.model3d[self.LEYE_INDICES].mean(axis=0)
