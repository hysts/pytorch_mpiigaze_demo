import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ptgaze.common import Camera, NormalizedPatch


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera, normalized_distance: float) -> None:
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance

    def normalize(
        self, image: np.ndarray, head_pose_rot: Rotation, center: np.ndarray, *, grayscale: bool = False
    ) -> NormalizedPatch:
        normalizing_rot = self._compute_normalizing_rotation(center, head_pose_rot)
        normalized_image = self._normalize_image(
            image, normalizing_rot, float(np.linalg.norm(center)), grayscale=grayscale
        )
        head_rot2d = self._normalize_head_pose(head_pose_rot, normalizing_rot)
        return NormalizedPatch(image=normalized_image, normalizing_rot=normalizing_rot, head_rot2d=head_rot2d)

    def _normalize_image(
        self, image: np.ndarray, normalizing_rot: Rotation, distance: float, *, grayscale: bool
    ) -> np.ndarray:
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)
        normalized_camera_matrix = self.normalized_camera.camera_matrix

        scale = self._get_scale_matrix(distance)
        conversion_matrix = scale @ normalizing_rot.as_matrix()

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        normalized_image = cv2.warpPerspective(
            image, projection_matrix, (self.normalized_camera.width, self.normalized_camera.height)
        )

        if grayscale:
            normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)
        return normalized_image

    @staticmethod
    def _normalize_head_pose(head_pose_rot: Rotation, normalizing_rot: Rotation) -> np.ndarray:
        # The MPIIGaze dataset annotation defines the normalized head pose as
        # the pitch and yaw of the z-axis of the normalized head rotation
        # matrix R @ hR, i.e. (arcsin(z_y), arctan2(z_x, z_z)).
        normalized_head_rot = normalizing_rot * head_pose_rot
        z_axis = normalized_head_rot.as_matrix()[:, 2]
        return np.array([np.arcsin(z_axis[1]), np.arctan2(z_axis[0], z_axis[2])])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray, head_rot: Rotation) -> Rotation:
        # See section 4.2 and Figure 9 of https://arxiv.org/abs/1711.09017
        z_axis = _normalize_vector(center.ravel())
        head_rot_mat = head_rot.as_matrix()
        head_x_axis = head_rot_mat[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, self.normalized_distance / distance],
            ],
            dtype=float,
        )
