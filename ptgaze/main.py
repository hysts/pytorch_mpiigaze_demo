import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from .demo import Demo
from .utils import (check_path_all, download_dlib_pretrained_model,
                    download_mpiifacegaze_model, download_mpiigaze_model,
                    expanduser_all, generate_dummy_camera_params)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Config file. When using a config file, all the other '
        'commandline arguments are ignored. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-xgaze.yaml'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='eth-xgaze',
        choices=['mpiigaze', 'mpiifacegaze', 'eth-xgaze'],
        help='With \'mpiigaze\', MPIIGaze model will be used. '
        'With \'mpiifacegaze\', MPIIFaceGaze model will be used. '
        'With \'eth-xgaze\', ETH-XGaze model will be used. '
        '(default: \'eth-xgaze\')')
    parser.add_argument(
        '--face-detector',
        type=str,
        default='dlib',
        choices=['dlib', 'face_alignment_dlib', 'face_alignment_sfd'],
        help='The method used to detect faces and find face landmarks '
        '(default: \'dlib\')')
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        help='Device used for model inference.')
    parser.add_argument('--image',
                        type=str,
                        help='Path to an input image file.')
    parser.add_argument('--video',
                        type=str,
                        help='Path to an input video file.')
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera calibration file. '
        'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        help='If specified, the overlaid video will be saved to this directory.'
    )
    parser.add_argument('--ext',
                        '-e',
                        type=str,
                        choices=['avi', 'mp4'],
                        help='Output video file extension.')
    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='If specified, the video is not displayed on screen, and saved '
        'to the output directory.')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_mode_config(args: argparse.Namespace) -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    if args.mode == 'mpiigaze':
        path = package_root / 'data/configs/mpiigaze.yaml'
    elif args.mode == 'mpiifacegaze':
        path = package_root / 'data/configs/mpiifacegaze.yaml'
    elif args.mode == 'eth-xgaze':
        path = package_root / 'data/configs/eth-xgaze.yaml'
    else:
        raise ValueError
    config = OmegaConf.load(path)

    if args.face_detector:
        config.face_detector.mode = args.face_detector
    if args.device:
        logger.debug(f'--device is {args.device}')
        config.device = args.device
        logger.debug(f'Update config.device to {config.device}')
    if config.device == 'cuda' and not torch.cuda.is_available():
        logger.debug('CUDA is not available.')
        config.device = 'cpu'
        logger.debug(f'Update config.device to cpu')
        warnings.warn('Run on CPU because CUDA is not available.')
    if args.image and args.video:
        raise ValueError('Only one of --image or --video can be specified.')
    if args.image:
        logger.debug(f'--image is {args.image}')
        config.demo.image_path = args.image
        config.demo.use_camera = False
        logger.debug(
            f'Update config.demo.image_path to {config.demo.image_path}')
        logger.debug('Set config.demo.use_camera False')
    if args.video:
        logger.debug(f'--video is {args.video}')
        config.demo.video_path = args.video
        config.demo.use_camera = False
        logger.debug(
            f'Update config.demo.video_path to {config.demo.video_path}')
        logger.debug('Set config.demo.use_camera False')
    if args.output_dir:
        logger.debug(f'--output-dir is {args.output_dir}')
        config.demo.output_dir = args.output_dir
        logger.debug(f'Update config.demo.output_dir')
    if args.ext:
        logger.debug(f'--ext is {args.ext}')
        config.demo.output_file_extension = args.ext
        logger.debug('Update config.demo.ouput_file_extension')
    if args.no_screen:
        logger.debug(f'--no-screen is set')
        config.demo.display_on_screen = False
        logger.debug('Set config.demo.display_on_screen False')
        if not config.demo.output_dir:
            logger.debug('config.demo.output_dir is not specified.')
            config.demo.output_dir = 'outputs'
            logger.debug('Set config.demo.output_dir \'outputs\'')

    return config


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)

    if args.config:
        config = OmegaConf.load(args.config)
    elif args.mode:
        config = load_mode_config(args)
    else:
        raise ValueError(
            'You need to specify one of \'--mode\' or \'--config\'.')
    expanduser_all(config)
    check_path_all(config)
    if config.demo.image_path or config.demo.video_path:
        generate_dummy_camera_params(config)
    OmegaConf.set_readonly(config, True)
    logger.info(OmegaConf.to_yaml(config))

    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()
    if args.mode:
        if config.mode == 'MPIIGaze':
            download_mpiigaze_model()
        elif config.mode == 'MPIIFaceGaze':
            download_mpiifacegaze_model()
        elif config.mode == 'ETH-XGaze':
            pass

    demo = Demo(config)
    demo.run()
