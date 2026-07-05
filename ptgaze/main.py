import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from .config_schema import AppConfig
from .demo import Demo
from .mode import GazeEstimationMode
from .utils import (
    check_path_all,
    download_dlib_pretrained_model,
    download_ethxgaze_model,
    download_mediapipe_face_landmarker,
    download_mpiifacegaze_model,
    download_mpiigaze_model,
    expanduser_all,
    set_dummy_camera_params,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Config file. When using a config file, all the other "
        "commandline arguments are ignored. "
        "See https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/configs/eth-xgaze.yaml",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mpiigaze", "mpiifacegaze", "eth-xgaze"],
        help="With 'mpiigaze', MPIIGaze model will be used. "
        "With 'mpiifacegaze', MPIIFaceGaze model will be used. "
        "With 'eth-xgaze', ETH-XGaze model will be used.",
    )
    parser.add_argument(
        "--face-detector",
        type=str,
        default="mediapipe",
        choices=["dlib", "face_alignment_dlib", "face_alignment_sfd", "mediapipe"],
        help="The method used to detect faces and find face landmarks (default: 'mediapipe')",
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device used for model inference.")
    parser.add_argument("--image", type=str, help="Path to an input image file.")
    parser.add_argument("--video", type=str, help="Path to an input video file.")
    parser.add_argument(
        "--camera",
        type=str,
        help="Camera calibration file. "
        "See https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/data/calib/sample_params.yaml",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, help="If specified, the overlaid video will be saved to this directory."
    )
    parser.add_argument("--ext", "-e", type=str, choices=["avi", "mp4"], help="Output video file extension.")
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="If specified, the video is not displayed on screen, and saved to the output directory.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_config_file(path: str | pathlib.Path) -> DictConfig:
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        error_msg = "The config file must define a mapping, not a list."
        raise TypeError(error_msg)
    # Merging with the structured schema rejects unknown keys and type
    # mismatches at startup.
    merged = OmegaConf.merge(OmegaConf.structured(AppConfig), config)
    if not isinstance(merged, DictConfig):
        raise TypeError(error_msg)
    return merged


def _apply_input_overrides(config: DictConfig, args: argparse.Namespace) -> None:
    if args.image and args.video:
        raise ValueError("Only one of --image or --video can be specified.")
    if args.image:
        config.demo.image_path = args.image
        config.demo.use_camera = False
    if args.video:
        config.demo.video_path = args.video
        config.demo.use_camera = False
    if args.camera:
        config.gaze_estimator.camera_params = args.camera
    elif args.image or args.video:
        config.gaze_estimator.use_dummy_camera_params = True


def _apply_output_overrides(config: DictConfig, args: argparse.Namespace) -> None:
    if args.output_dir:
        config.demo.output_dir = args.output_dir
    if args.ext:
        config.demo.output_file_extension = args.ext
    if args.no_screen:
        config.demo.display_on_screen = False
        if not config.demo.output_dir:
            config.demo.output_dir = "outputs"


def load_mode_config(args: argparse.Namespace) -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    if args.mode == "mpiigaze":
        path = package_root / "data/configs/mpiigaze.yaml"
    elif args.mode == "mpiifacegaze":
        path = package_root / "data/configs/mpiifacegaze.yaml"
    elif args.mode == "eth-xgaze":
        path = package_root / "data/configs/eth-xgaze.yaml"
    else:
        raise ValueError
    config = _load_config_file(path)
    config.PACKAGE_ROOT = package_root.as_posix()

    if args.face_detector:
        config.face_detector.mode = args.face_detector
    if args.device:
        config.device = args.device
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
        warnings.warn("Run on CPU because CUDA is not available.", stacklevel=2)
    _apply_input_overrides(config, args)
    _apply_output_overrides(config, args)

    return config


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger("ptgaze").setLevel(logging.DEBUG)

    if args.config:
        config = _load_config_file(args.config)
        config.PACKAGE_ROOT = pathlib.Path(__file__).parent.resolve().as_posix()
    elif args.mode:
        config = load_mode_config(args)
        # In the --mode flow the checkpoint comes from the Hugging Face Hub
        # cache, so the path is only known after the download.
        mode = GazeEstimationMode(config.mode)
        if mode is GazeEstimationMode.MPIIGAZE:
            checkpoint_path = download_mpiigaze_model()
        elif mode is GazeEstimationMode.MPIIFACEGAZE:
            checkpoint_path = download_mpiifacegaze_model()
        else:
            checkpoint_path = download_ethxgaze_model()
        config.gaze_estimator.checkpoint = checkpoint_path.as_posix()
    else:
        raise ValueError("You need to specify one of '--mode' or '--config'.")
    expanduser_all(config)
    if config.gaze_estimator.use_dummy_camera_params:
        set_dummy_camera_params(config)

    OmegaConf.set_readonly(config, True)  # noqa: FBT003
    logger.info(OmegaConf.to_yaml(config))

    if config.face_detector.mode == "dlib":
        download_dlib_pretrained_model()
    if config.face_detector.mode == "mediapipe":
        download_mediapipe_face_landmarker()

    check_path_all(config)

    demo = Demo(config)
    demo.run()
