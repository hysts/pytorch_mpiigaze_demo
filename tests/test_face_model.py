import numpy as np
from scipy.spatial.transform import Rotation

from ptgaze.common.face import Face
from ptgaze.common.face_model_68 import FaceModel68
from ptgaze.mode import GazeEstimationMode


def _make_posed_face(face_model: FaceModel68) -> Face:
    face = Face(bbox=np.zeros((2, 2)), landmarks=np.zeros((68, 2)))
    face.head_pose_rot = Rotation.from_euler("XYZ", [5, -10, 3], degrees=True)
    face.head_position = np.array([0.02, 0.01, 0.6])
    face_model.compute_3d_pose(face)
    return face


def test_eye_centers_are_corner_midpoints() -> None:
    face_model = FaceModel68()
    face = _make_posed_face(face_model)
    face_model.compute_face_eye_centers(face, GazeEstimationMode.MPIIFACEGAZE)
    assert face.model3d is not None
    assert face.reye.center is not None
    assert face.leye.center is not None
    np.testing.assert_allclose(face.reye.center, face.model3d[[36, 39]].mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(face.leye.center, face.model3d[[42, 45]].mean(axis=0), atol=1e-12)


def test_face_center_definition_depends_on_mode() -> None:
    # MPIIFaceGaze: mean of the four eye corners and two mouth corners.
    # ETH-XGaze: mean of the four eye corners and two nose corners.
    face_model = FaceModel68()
    face = _make_posed_face(face_model)
    assert face.model3d is not None

    face_model.compute_face_eye_centers(face, GazeEstimationMode.MPIIFACEGAZE)
    assert face.center is not None
    expected_mouth = face.model3d[[36, 39, 42, 45, 48, 54]].mean(axis=0)
    np.testing.assert_allclose(face.center, expected_mouth, atol=1e-12)

    face_model.compute_face_eye_centers(face, GazeEstimationMode.ETH_XGAZE)
    assert face.center is not None
    expected_nose = face.model3d[[36, 39, 42, 45, 31, 35]].mean(axis=0)
    np.testing.assert_allclose(face.center, expected_nose, atol=1e-12)
