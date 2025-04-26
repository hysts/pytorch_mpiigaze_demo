import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face

AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

DIM_2D = 2
DIM_3D = 3


class Visualizer:
    def __init__(self, camera: Camera, center_point_index: int) -> None:
        self._camera = camera
        self._center_point_index = center_point_index
        self.image: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def draw_bbox(self, bbox: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), lw: int = 1) -> None:
        if self.image is None:
            raise ValueError("image is not set")
        if bbox.shape != (2, 2):
            error_msg = f"bbox must have shape (2, 2), but got {bbox.shape}"
            raise ValueError(error_msg)
        bbox = np.round(bbox).astype(int).tolist()
        cv2.rectangle(self.image, tuple(bbox[0]), tuple(bbox[1]), color, lw)

    @staticmethod
    def _convert_pt(point: np.ndarray) -> tuple[int, int]:
        return tuple(np.round(point).astype(int).tolist())

    def draw_points(self, points: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), size: int = 3) -> None:
        if self.image is None:
            raise ValueError("image is not set")
        if points.shape[1] != DIM_2D:
            error_msg = f"points must have {DIM_2D} columns, but got {points.shape[1]}"
            raise ValueError(error_msg)
        for pt in points:
            draw_pt = self._convert_pt(pt)
            cv2.circle(self.image, draw_pt, size, color, cv2.FILLED)

    def draw_3d_points(self, points3d: np.ndarray, color: tuple[int, int, int] = (255, 0, 255), size: int = 3) -> None:
        if self.image is None:
            raise ValueError("image is not set")
        if points3d.shape[1] != DIM_3D:
            error_msg = f"points3d must have {DIM_3D} columns, but got {points3d.shape[1]}"
            raise ValueError(error_msg)
        points2d = self._camera.project_points(points3d)
        self.draw_points(points2d, color=color, size=size)

    def draw_3d_line(
        self, point0: np.ndarray, point1: np.ndarray, color: tuple[int, int, int] = (255, 255, 0), lw: int = 1
    ) -> None:
        if self.image is None:
            raise ValueError("image is not set")
        if point0.shape != point1.shape == (DIM_3D,):
            error_msg = (
                f"point0 and point1 must have {DIM_3D} columns, but got {point0.shape[0]} and {point1.shape[0]}"
            )
            raise ValueError(error_msg)
        points3d = np.vstack([point0, point1])
        points2d = self._camera.project_points(points3d)
        pt0 = self._convert_pt(points2d[0])
        pt1 = self._convert_pt(points2d[1])
        cv2.line(self.image, pt0, pt1, color, lw, cv2.LINE_AA)

    def draw_model_axes(self, face: Face, length: float, lw: int = 2) -> None:
        if self.image is None:
            raise ValueError("image is not set")
        if face.head_pose_rot is None:
            raise ValueError("face.head_pose_rot is not set")
        if face.head_position is None:
            raise ValueError("face.head_position is not set")
        # Get the axes of the model coordinate system
        axes3d = np.eye(DIM_3D, dtype=float) @ Rotation.from_euler("XYZ", [0, np.pi, 0]).as_matrix()
        axes3d = axes3d * length
        axes2d = self._camera.project_points(axes3d, face.head_pose_rot.as_rotvec(), face.head_position)
        center = face.landmarks[self._center_point_index]
        center = self._convert_pt(center)
        for pt, color in zip(axes2d, AXIS_COLORS, strict=False):
            draw_pt = self._convert_pt(pt)
            cv2.line(self.image, center, draw_pt, color, lw, cv2.LINE_AA)
