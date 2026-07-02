import numpy as np
from omegaconf import DictConfig

from ptgaze.common import Face


class LandmarkEstimator:
    def __init__(self, config: DictConfig) -> None:
        # The detector backends are imported lazily so that only the selected
        # one is loaded. They are heavy imports, and dlib in particular may
        # not be installed.
        self.mode = config.face_detector.mode
        if self.mode == "dlib":
            import dlib  # noqa: PLC0415

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(config.face_detector.dlib_model_path)
        elif self.mode == "face_alignment_dlib":
            import dlib  # noqa: PLC0415
            import face_alignment  # noqa: PLC0415

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, face_detector="dlib", flip_input=False, device=config.device
            )
        elif self.mode == "face_alignment_sfd":
            import face_alignment  # noqa: PLC0415
            import face_alignment.detection.sfd  # noqa: PLC0415

            self.detector = face_alignment.detection.sfd.sfd_detector.SFDDetector(device=config.device)
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device=config.device
            )
        elif self.mode == "mediapipe":
            import mediapipe  # noqa: PLC0415

            self.detector = mediapipe.solutions.face_mesh.FaceMesh(
                max_num_faces=config.face_detector.mediapipe_max_num_faces,
                static_image_mode=config.face_detector.mediapipe_static_image_mode,
            )
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> list[Face]:
        if self.mode == "dlib":
            return self._detect_faces_dlib(image)
        if self.mode == "face_alignment_dlib":
            return self._detect_faces_face_alignment_dlib(image)
        if self.mode == "face_alignment_sfd":
            return self._detect_faces_face_alignment_sfd(image)
        if self.mode == "mediapipe":
            return self._detect_faces_mediapipe(image)
        raise ValueError

    def _detect_faces_dlib(self, image: np.ndarray) -> list[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()], dtype=float)
            bbox_np = np.array([[bbox.left(), bbox.top()], [bbox.right(), bbox.bottom()]], dtype=float)
            detected.append(Face(bbox_np, landmarks))
        return detected

    def _detect_faces_face_alignment_dlib(self, image: np.ndarray) -> list[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        bboxes = [[bbox.left(), bbox.top(), bbox.right(), bbox.bottom()] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1], detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions, strict=False):
            bbox_np = np.array(bbox, dtype=float).reshape(2, 2)
            detected.append(Face(bbox_np, landmarks))
        return detected

    def _detect_faces_face_alignment_sfd(self, image: np.ndarray) -> list[Face]:
        bboxes = self.detector.detect_from_image(image[:, :, ::-1].copy())
        bboxes = [bbox[:4] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1], detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions, strict=False):
            bbox_np = np.array(bbox, dtype=float).reshape(2, 2)
            detected.append(Face(bbox_np, landmarks))
        return detected

    def _detect_faces_mediapipe(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h) for pt in prediction.landmark], dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected
