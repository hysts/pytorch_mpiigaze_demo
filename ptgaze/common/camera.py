import dataclasses
import pathlib

import cv2
import numpy as np
import yaml

DIM_3D = 3


@dataclasses.dataclass(frozen=True)
class Camera:
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coefficients: np.ndarray

    @classmethod
    def from_file(cls, camera_params_path: str) -> "Camera":
        with pathlib.Path(camera_params_path).open() as f:
            data = yaml.safe_load(f)
        return cls(
            width=data["image_width"],
            height=data["image_height"],
            camera_matrix=np.array(data["camera_matrix"]["data"]).reshape(3, 3),
            dist_coefficients=np.array(data["distortion_coefficients"]["data"]).reshape(-1, 1),
        )

    @classmethod
    def create_dummy(cls, width: int, height: int) -> "Camera":
        """Create a camera with a focal length equal to the image width and no distortion."""
        camera_matrix = np.array(
            [
                [width, 0, width // 2],
                [0, width, height // 2],
                [0, 0, 1],
            ],
            dtype=float,
        )
        return cls(width=width, height=height, camera_matrix=camera_matrix, dist_coefficients=np.zeros((5, 1)))

    def project_points(
        self, points3d: np.ndarray, rvec: np.ndarray | None = None, tvec: np.ndarray | None = None
    ) -> np.ndarray:
        if points3d.shape[1] != DIM_3D:
            error_msg = f"points3d must have {DIM_3D} columns, but got {points3d.shape[1]}"
            raise ValueError(error_msg)
        if rvec is None:
            rvec = np.zeros(DIM_3D, dtype=float)
        if tvec is None:
            tvec = np.zeros(DIM_3D, dtype=float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec, self.camera_matrix, self.dist_coefficients)
        return points2d.reshape(-1, 2)
