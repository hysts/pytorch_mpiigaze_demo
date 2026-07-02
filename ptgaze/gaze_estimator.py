import logging

import numpy as np
import torch
from omegaconf import DictConfig

from .common import Camera, Face, FaceCenters, FaceParts, FacePartsName, HeadPose, NormalizedPatch
from .common.gaze import denormalize_gaze_vector, gaze_angles_to_vector
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from .mode import GazeEstimationMode
from .models import create_model
from .transforms import create_transform
from .utils import get_3d_face_model

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = (FacePartsName.REYE, FacePartsName.LEYE)

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._mode = GazeEstimationMode(config.mode)

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera, self._config.gaze_estimator.normalized_camera_distance
        )
        self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> list[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        head_pose = self._face_model3d.estimate_head_pose(face.landmarks, self.camera)
        centers = self._face_model3d.compute_centers(head_pose.model3d, self._mode)
        self._store_head_pose(face, head_pose, centers)

        if self._mode.input_is_face:
            patch = self._head_pose_normalizer.normalize(image, head_pose.rot, centers.face)
            gaze_angles = self._run_face_model(patch.image)
            self._store_gaze(face, patch, gaze_angles)
        else:
            reye_patch = self._head_pose_normalizer.normalize(image, head_pose.rot, centers.reye, grayscale=True)
            leye_patch = self._head_pose_normalizer.normalize(image, head_pose.rot, centers.leye, grayscale=True)
            reye_angles, leye_angles = self._run_mpiigaze_model(reye_patch, leye_patch)
            self._store_gaze(face.reye, reye_patch, reye_angles)
            self._store_gaze(face.leye, leye_patch, leye_angles)

    @staticmethod
    def _store_head_pose(face: Face, head_pose: HeadPose, centers: FaceCenters) -> None:
        face.head_pose_rot = head_pose.rot
        face.head_position = head_pose.position
        face.model3d = head_pose.model3d
        face.center = centers.face
        face.reye.center = centers.reye
        face.leye.center = centers.leye
        face.reye.head_pose_rot = head_pose.rot
        face.leye.head_pose_rot = head_pose.rot

    @staticmethod
    def _store_gaze(part: FaceParts, patch: NormalizedPatch, gaze_angles: np.ndarray) -> None:
        part.normalizing_rot = patch.normalizing_rot
        part.normalized_head_rot2d = patch.head_rot2d
        part.normalized_image = patch.image
        part.normalized_gaze_angles = gaze_angles
        part.normalized_gaze_vector = gaze_angles_to_vector(gaze_angles)
        part.gaze_vector = denormalize_gaze_vector(part.normalized_gaze_vector, patch.normalizing_rot)

    @torch.no_grad()
    def _run_mpiigaze_model(
        self, reye_patch: NormalizedPatch, leye_patch: NormalizedPatch
    ) -> tuple[np.ndarray, np.ndarray]:
        # The model works in the left-eye convention, so the right eye is
        # mirrored on input and the predicted yaw is flipped back.
        images = []
        head_poses = []
        for patch, mirror in [(reye_patch, True), (leye_patch, False)]:
            image = patch.image
            head_pose = patch.head_rot2d
            if mirror:
                image = image[:, ::-1].copy()
                head_pose = head_pose * np.array([1, -1])
            images.append(self._transform(image))
            head_poses.append(head_pose)

        device = torch.device(self._config.device)
        images_tensor = torch.stack(images).to(device)
        head_poses_tensor = torch.from_numpy(np.array(head_poses).astype(np.float32)).to(device)
        predictions = self._gaze_estimation_model(images_tensor, head_poses_tensor)
        predictions = predictions.cpu().numpy()
        return predictions[0] * np.array([1, -1]), predictions[1]

    @torch.no_grad()
    def _run_face_model(self, normalized_image: np.ndarray) -> np.ndarray:
        image = self._transform(normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()
        return prediction[0]
