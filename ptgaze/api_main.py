import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf
import cv2

from ptgaze.api_gaze import APIGaze
from ptgaze.utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)

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
        '--input-path',
        type=str,
        help='path of the input image file'
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logging.getLogger('ptgaze').setLevel(logging.DEBUG)

    if args.config:
        config = OmegaConf.load(args.config)
        package_root = pathlib.Path(__file__).parent.resolve()
        config.PACKAGE_ROOT = package_root.as_posix()
        print('loaded config')
    else:
        raise ValueError(
            'You need to specify \'--config\'.')

    expanduser_all(config)
    if config.gaze_estimator.use_dummy_camera_params:
        generate_dummy_camera_params(config)

    OmegaConf.set_readonly(config, True)
    logger.info(OmegaConf.to_yaml(config))

    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()
    if config.mode == 'ETH-XGaze':
                download_ethxgaze_model()

    check_path_all(config)
    api = APIGaze(config)
    
    image = cv2.imread(args.input_path)
    api.run(image)