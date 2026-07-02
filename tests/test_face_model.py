import numpy as np
from scipy.spatial.transform import Rotation

from ptgaze.common.face_model_68 import FaceModel68
from ptgaze.mode import GazeEstimationMode


def _posed_model3d(face_model: FaceModel68) -> np.ndarray:
    rot = Rotation.from_euler("XYZ", [5, -10, 3], degrees=True)
    position = np.array([0.02, 0.01, 0.6])
    return face_model.LANDMARKS @ rot.as_matrix().T + position


def test_eye_centers_are_corner_midpoints() -> None:
    face_model = FaceModel68()
    model3d = _posed_model3d(face_model)
    centers = face_model.compute_centers(model3d, GazeEstimationMode.MPIIFACEGAZE)
    np.testing.assert_allclose(centers.reye, model3d[[36, 39]].mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(centers.leye, model3d[[42, 45]].mean(axis=0), atol=1e-12)


def test_face_center_definition_depends_on_mode() -> None:
    # MPIIFaceGaze: mean of the four eye corners and two mouth corners.
    # ETH-XGaze: mean of the four eye corners and two nose corners.
    face_model = FaceModel68()
    model3d = _posed_model3d(face_model)

    centers = face_model.compute_centers(model3d, GazeEstimationMode.MPIIFACEGAZE)
    expected_mouth = model3d[[36, 39, 42, 45, 48, 54]].mean(axis=0)
    np.testing.assert_allclose(centers.face, expected_mouth, atol=1e-12)

    centers = face_model.compute_centers(model3d, GazeEstimationMode.ETH_XGAZE)
    expected_nose = model3d[[36, 39, 42, 45, 31, 35]].mean(axis=0)
    np.testing.assert_allclose(centers.face, expected_nose, atol=1e-12)
