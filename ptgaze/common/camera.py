import dataclasses
import pathlib

import cv2
import numpy as np
import yaml

DIM_3D = 3


@dataclasses.dataclass()
class Camera:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)
    dist_coefficients: np.ndarray = dataclasses.field(init=False)

    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path: str) -> None:
        with pathlib.Path(camera_params_path).open() as f:
            data = yaml.safe_load(f)
        self.width = data["image_width"]
        self.height = data["image_height"]
        self.camera_matrix = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        self.dist_coefficients = np.array(data["distortion_coefficients"]["data"]).reshape(-1, 1)

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
