"""Structured schema for the config files.

Loaded configs are merged with this schema so that unknown keys and
type mismatches are rejected at startup instead of failing deep inside
the pipeline.
"""

import dataclasses

from omegaconf import MISSING


@dataclasses.dataclass
class ModelBackboneConfig:
    name: str = "resnet_simple"
    pretrained: str = "resnet18"
    resnet_block: str = "basic"
    resnet_layers: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 2])


@dataclasses.dataclass
class ModelConfig:
    name: str = MISSING
    backbone: ModelBackboneConfig = dataclasses.field(default_factory=ModelBackboneConfig)


@dataclasses.dataclass
class FaceDetectorConfig:
    mode: str = "mediapipe"
    dlib_model_path: str = "~/.ptgaze/dlib/shape_predictor_68_face_landmarks.dat"
    mediapipe_model_path: str = "~/.ptgaze/mediapipe/face_landmarker.task"
    mediapipe_max_num_faces: int = 3
    mediapipe_static_image_mode: bool = False


@dataclasses.dataclass
class GazeEstimatorConfig:
    checkpoint: str = MISSING
    camera_params: str = MISSING
    use_dummy_camera_params: bool = False
    # Frame size (width, height) used to build the dummy camera when
    # use_dummy_camera_params is set. Filled in at startup.
    dummy_camera_size: list[int] | None = None
    normalized_camera_params: str = MISSING
    normalized_camera_distance: float = MISSING
    image_size: list[int] | None = None


@dataclasses.dataclass
class DemoConfig:
    use_camera: bool = True
    display_on_screen: bool = True
    wait_time: int = 1
    image_path: str | None = None
    video_path: str | None = None
    output_dir: str | None = None
    output_file_extension: str = "avi"
    head_pose_axis_length: float = 0.05
    gaze_visualization_length: float = 0.05
    show_bbox: bool = True
    show_head_pose: bool = False
    show_landmarks: bool = False
    show_normalized_image: bool = False
    show_template_model: bool = False


@dataclasses.dataclass
class AppConfig:
    mode: str = MISSING
    device: str = "cpu"
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    face_detector: FaceDetectorConfig = dataclasses.field(default_factory=FaceDetectorConfig)
    gaze_estimator: GazeEstimatorConfig = dataclasses.field(default_factory=GazeEstimatorConfig)
    demo: DemoConfig = dataclasses.field(default_factory=DemoConfig)
    PACKAGE_ROOT: str = MISSING
