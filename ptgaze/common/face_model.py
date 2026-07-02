import dataclasses

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ptgaze.mode import GazeEstimationMode

from .camera import Camera
from .results import FaceCenters, HeadPose


@dataclasses.dataclass(frozen=True)
class FaceModel:
    LANDMARKS: np.ndarray
    REYE_INDICES: np.ndarray
    LEYE_INDICES: np.ndarray
    MOUTH_INDICES: np.ndarray
    NOSE_INDICES: np.ndarray
    CHIN_INDEX: int
    NOSE_INDEX: int

    def estimate_head_pose(self, landmarks2d: np.ndarray, camera: Camera) -> HeadPose:
        """Estimate the head pose by fitting 3D template model."""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        rvec = np.zeros(3, dtype=float)
        tvec = np.array([0, 0, 1], dtype=float)
        # The landmarks are detected on the undistorted image, so the
        # distortion coefficients must not be applied here again.
        _, rvec, tvec = cv2.solvePnP(
            self.LANDMARKS,
            landmarks2d,
            camera.camera_matrix,
            np.zeros((1, 5)),
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        rot = Rotation.from_rotvec(rvec)
        model3d = self.LANDMARKS @ rot.as_matrix().T + tvec
        return HeadPose(rot=rot, position=tvec, model3d=model3d)

    def compute_centers(self, model3d: np.ndarray, mode: GazeEstimationMode) -> FaceCenters:
        """Compute the centers of the face and eyes.

        In the case of MPIIFaceGaze, the face center is defined as the
        average coordinates of the six points at the corners of both
        eyes and the mouth. In the case of ETH-XGaze, it's defined as
        the average coordinates of the six points at the corners of both
        eyes and the nose. The eye centers are defined as the average
        coordinates of the corners of each eye.
        """
        if mode is GazeEstimationMode.ETH_XGAZE:
            face_center = model3d[np.concatenate([self.REYE_INDICES, self.LEYE_INDICES, self.NOSE_INDICES])].mean(
                axis=0
            )
        else:
            face_center = model3d[np.concatenate([self.REYE_INDICES, self.LEYE_INDICES, self.MOUTH_INDICES])].mean(
                axis=0
            )
        return FaceCenters(
            face=face_center,
            reye=model3d[self.REYE_INDICES].mean(axis=0),
            leye=model3d[self.LEYE_INDICES].mean(axis=0),
        )
