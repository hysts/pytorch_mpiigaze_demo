from typing import List
import mediapipe as mp
import cv2
from face_detection import RetinaFace
import numpy as np
from omegaconf import DictConfig
from ptgaze.common.face import Face

class LandmarkEstimator:
    def __init__(self, config: DictConfig):
        self.mode = config.face_detector.mode
        self.detector = RetinaFace()
        self.predictor = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                             min_detection_confidence=0.5)

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._detect_faces_retina_face(image)

    def _detect_faces_retina_face(self, image: np.ndarray) -> List[Face]:
        # Detect faces using RetinaFace
        faces = self.detector(image[:, :, ::-1])
        detected = []

        for face in faces:
            box, _, score = face

            if score > 0.95:
                x, y, w, h = [int(val) for val in box]
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                # Adjust the width and height to be more balanced
                w = h
                # Extract the face region
                face_img = image[y:y + h, x:x + w]

                landmarks = self._media_pipe_land_marker(face_img, x, y, h, w)

                if landmarks is not None:
                    detected.append(Face(box, landmarks))
                else:
                    if len(detected) > 0:
                        print("use pre land marks")
                        detected.append(Face(box,detected[-1].landmarks))


        return detected

    def _media_pipe_land_marker(self, face_img,x,y,h,w) :

        # Convert the cropped face image to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Process the face image
        results = self.predictor.process(face_rgb)

        # Get the face landmarks
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            # Convert landmarks to pixel coordinates in the original image
            landmarks_np = np.array([[int(lm.x * w + x), int(lm.y * h + y)] for lm in face_landmarks],float)
            return landmarks_np