import bz2
import logging
import operator
import pathlib

import cv2
import torch.hub
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from .common.face_model import FaceModel
from .common.face_model_68 import FaceModel68
from .common.face_model_mediapipe import FaceModelMediaPipe

logger = logging.getLogger(__name__)


def get_3d_face_model(config: DictConfig) -> FaceModel:
    if config.face_detector.mode == "mediapipe":
        return FaceModelMediaPipe()
    return FaceModel68()


def download_dlib_pretrained_model() -> None:
    logger.debug("Called download_dlib_pretrained_model()")

    dlib_model_dir = pathlib.Path("~/.ptgaze/dlib/").expanduser()
    dlib_model_dir.mkdir(exist_ok=True, parents=True)
    dlib_model_path = dlib_model_dir / "shape_predictor_68_face_landmarks.dat"
    logger.debug("Update config.face_detector.dlib_model_path to %s", dlib_model_path.as_posix())

    if dlib_model_path.exists():
        logger.debug("dlib pretrained model %s already exists.", dlib_model_path.as_posix())
        return

    logger.debug("Download the dlib pretrained model")
    bz2_path = dlib_model_path.as_posix() + ".bz2"
    torch.hub.download_url_to_file("https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", bz2_path)
    with bz2.BZ2File(bz2_path, "rb") as f_in, pathlib.Path(dlib_model_path).open("wb") as f_out:
        data = f_in.read()
        f_out.write(data)


def download_mediapipe_face_landmarker() -> None:
    logger.debug("Called download_mediapipe_face_landmarker()")

    model_dir = pathlib.Path("~/.ptgaze/mediapipe/").expanduser()
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "face_landmarker.task"

    if model_path.exists():
        logger.debug("MediaPipe face landmarker model %s already exists.", model_path.as_posix())
        return

    logger.debug("Download the MediaPipe face landmarker model")
    torch.hub.download_url_to_file(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path.as_posix(),
    )


def download_mpiigaze_model() -> pathlib.Path:
    logger.debug("Called download_mpiigaze_model()")
    return pathlib.Path(hf_hub_download("hysts/ptgaze-mpiigaze-resnet-preact", "model.safetensors"))


def download_mpiifacegaze_model() -> pathlib.Path:
    logger.debug("Called download_mpiifacegaze_model()")
    return pathlib.Path(hf_hub_download("hysts/ptgaze-mpiifacegaze-resnet-simple", "model.safetensors"))


def download_ethxgaze_model() -> pathlib.Path:
    logger.debug("Called download_ethxgaze_model()")
    return pathlib.Path(hf_hub_download("hysts/ptgaze-eth-xgaze-resnet18", "model.safetensors"))


def set_dummy_camera_params(config: DictConfig) -> None:
    logger.debug("Called set_dummy_camera_params()")
    if config.demo.image_path:
        path = pathlib.Path(config.demo.image_path).expanduser()
        image = cv2.imread(path.as_posix())
        if image is None:
            error_msg = f"{path.as_posix()} could not be read as an image."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        h, w = image.shape[:2]
    elif config.demo.video_path:
        logger.debug("Open video %s", config.demo.video_path)
        video_path = pathlib.Path(config.demo.video_path).expanduser().as_posix()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"{config.demo.video_path} is not opened."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    else:
        raise ValueError
    logger.debug("Frame size is (%d, %d)", w, h)
    config.gaze_estimator.dummy_camera_size = [w, h]


def _expanduser(path: str) -> str:
    if not path:
        return path
    return pathlib.Path(path).expanduser().as_posix()


def expanduser_all(config: DictConfig) -> None:
    if hasattr(config.face_detector, "dlib_model_path"):
        config.face_detector.dlib_model_path = _expanduser(config.face_detector.dlib_model_path)
    if hasattr(config.face_detector, "mediapipe_model_path"):
        config.face_detector.mediapipe_model_path = _expanduser(config.face_detector.mediapipe_model_path)
    config.gaze_estimator.checkpoint = _expanduser(config.gaze_estimator.checkpoint)
    if not config.gaze_estimator.use_dummy_camera_params:
        config.gaze_estimator.camera_params = _expanduser(config.gaze_estimator.camera_params)
    config.gaze_estimator.normalized_camera_params = _expanduser(config.gaze_estimator.normalized_camera_params)
    if hasattr(config.demo, "image_path"):
        config.demo.image_path = _expanduser(config.demo.image_path)
    if hasattr(config.demo, "video_path"):
        config.demo.video_path = _expanduser(config.demo.video_path)
    if hasattr(config.demo, "output_dir"):
        config.demo.output_dir = _expanduser(config.demo.output_dir)


def _check_path(config: DictConfig, key: str) -> None:
    path_str = operator.attrgetter(key)(config)
    path = pathlib.Path(path_str)
    if not path.exists():
        error_msg = f"config.{key}: {path.as_posix()} not found."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    if not path.is_file():
        error_msg = f"config.{key}: {path.as_posix()} is not a file."
        logger.error(error_msg)
        raise ValueError(error_msg)


def check_path_all(config: DictConfig) -> None:
    if config.face_detector.mode == "dlib":
        _check_path(config, "face_detector.dlib_model_path")
    if config.face_detector.mode == "mediapipe":
        _check_path(config, "face_detector.mediapipe_model_path")
    _check_path(config, "gaze_estimator.checkpoint")
    if not config.gaze_estimator.use_dummy_camera_params:
        _check_path(config, "gaze_estimator.camera_params")
    _check_path(config, "gaze_estimator.normalized_camera_params")
    if config.demo.image_path:
        _check_path(config, "demo.image_path")
    if config.demo.video_path:
        _check_path(config, "demo.video_path")
