import argparse
import bz2
import logging
import operator
import pathlib
import warnings

import cv2
import torch.hub
import yacs.config
import yaml

logger = logging.getLogger(__name__)


def _download_dlib_pretrained_model(config: yacs.config.CfgNode) -> None:
    logger.debug('Called download_dlib_pretrained_model()')
    if config.face_detector.mode != 'dlib':
        logger.debug('config.facedetector.mode is not \'dlib\'.')
        return
    if config.face_detector.dlib.model:
        logger.debug('config.facedetector.mode is specified.')
        return

    dlib_model_dir = pathlib.Path('~/.ptgaze/dlib/').expanduser()
    dlib_model_dir.mkdir(exist_ok=True, parents=True)
    dlib_model_path = dlib_model_dir / 'shape_predictor_68_face_landmarks.dat'
    config.face_detector.dlib.model = dlib_model_path.as_posix()
    logger.debug(
        f'Update config.face_detector.dlib.model to {dlib_model_path.as_posix()}'
    )

    if dlib_model_path.exists():
        logger.debug(
            f'dlib pretrained model {dlib_model_path.as_posix()} already exists.'
        )
        return

    logger.debug('Download the dlib pretrained model')
    bz2_path = dlib_model_path.as_posix() + '.bz2'
    torch.hub.download_url_to_file(
        'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        bz2_path)
    with bz2.BZ2File(bz2_path, 'rb') as f_in, open(dlib_model_path,
                                                   'wb') as f_out:
        data = f_in.read()
        f_out.write(data)


def _download_eye_model() -> pathlib.Path:
    logger.debug('Called _download_eye_model()')
    output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'mpiigaze_resnet_preact.pth'
    if not output_path.exists():
        logger.debug('Download the pretrained model')
        torch.hub.download_url_to_file(
            'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth',
            output_path)
    else:
        logger.debug(f'The pretrained model {output_path} already exists.')
    return output_path


def _download_face_model() -> pathlib.Path:
    logger.debug('Called _download_face_model()')
    output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'mpiifacegaze_resnet_simple.pth'
    if not output_path.exists():
        logger.debug('Download the pretrained model')
        torch.hub.download_url_to_file(
            'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth',
            output_path)
    else:
        logger.debug(f'The pretrained model {output_path} already exists.')
    return output_path


def _generate_dummy_camera_params(config: yacs.config.CfgNode) -> None:
    logger.debug('Called _generate_dummy_camera_params()')
    if config.demo.image_path:
        path = pathlib.Path(config.demo.image_path).expanduser()
        image = cv2.imread(path.as_posix())
        h, w = image.shape[:2]
    elif config.demo.video_path:
        logger.debug(f'Open video {config.demo.video_path}')
        path = pathlib.Path(config.demo.video_path).expanduser().as_posix()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f'{config.demo.video_path} is not opened.')
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap.release()
    else:
        raise ValueError
    logger.debug(f'Frame size is ({w}, {h})')
    logger.debug(f'Close video {config.demo.video_path}')
    logger.debug(f'Create a dummy camera param file /tmp/camera_params.yaml')
    dic = {
        'image_width': int(w),
        'image_height': int(h),
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [w, 0., w // 2, 0., w, h // 2, 0., 0., 1.]
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': 5,
            'data': [0., 0., 0., 0., 0.]
        }
    }
    with open('/tmp/camera_params.yaml', 'w') as f:
        yaml.safe_dump(dic, f)
    config.gaze_estimator.camera_params = '/tmp/camera_params.yaml'
    logger.debug(
        'Update config.gaze_estimator.camera_params to /tmp/camera_params.yaml'
    )


def _update_camera_config(config: yacs.config.CfgNode) -> None:
    logger.debug('Called update_camera_config()')
    package_root = pathlib.Path(__file__).parent.resolve()
    if not config.gaze_estimator.camera_params:
        logger.debug('config.gaze_estimator.camera_params is not specified.')
        logger.debug(f'config.demo.use_camera is {config.demo.use_camera}')
        if config.demo.use_camera:
            logger.debug('Use sample_params.yaml')
            config.gaze_estimator.camera_params = (
                package_root / 'data/calib/sample_params.yaml').as_posix()
            warnings.warn('Use the sample parameters because no camera '
                          'calibration file is specified.')
        elif config.demo.image_path or config.demo.video_path:
            warnings.warn('Calibration file is not specified. Generate '
                          'dummy parameters from the video size.')
            _generate_dummy_camera_params(config)
        else:
            logger.debug('Bothe config.demo.image_path and '
                         'config.demo.video_path are not specified.')
            raise ValueError(
                'No input found. config.demo.use_camera is False and '
                'both config.demo.image_path and config.demo.video_path '
                'are not specified.')

    else:
        logger.debug('config.gaze_estimator.camera_params is specified.')

    path = pathlib.Path(config.gaze_estimator.camera_params).expanduser()
    config.gaze_estimator.camera_params = path.as_posix()
    logger.debug(
        f'Update config.gaze_estimator.camera_params to {path.as_posix()}')


def _set_eye_default_camera(config: yacs.config.CfgNode) -> None:
    logger.debug('Called _set_eye_default_camera()')
    package_root = pathlib.Path(__file__).resolve().parent
    filename = 'data/calib/normalized_camera_params_eye.yaml'
    default_params = package_root / filename
    config.gaze_estimator.normalized_camera_params = default_params.as_posix()
    config.gaze_estimator.normalized_camera_distance = 0.6
    logger.debug('config.gaze_estimator.normalized_camera_params is set.')
    logger.debug('config.gaze_estimator.normalized_camera_distance is set.')


def _set_face_default_camera(config: yacs.config.CfgNode) -> None:
    logger.debug('Called _set_face_default_camera()')
    package_root = pathlib.Path(__file__).resolve().parent
    filename = 'data/calib/normalized_camera_params_face.yaml'
    default_params = package_root / filename
    config.gaze_estimator.normalized_camera_params = default_params.as_posix()
    config.gaze_estimator.normalized_camera_distance = 1.0
    logger.debug('config.gaze_estimator.normalized_camera_params is set.')
    logger.debug('config.gaze_estimator.normalized_camera_distance is set.')


def _expanduser(path: str) -> str:
    if not path:
        return path
    return pathlib.Path(path).expanduser().as_posix()


def _expanduser_all(config: yacs.config.CfgNode) -> None:
    config.face_detector.dlib.model = _expanduser(
        config.face_detector.dlib.model)
    config.gaze_estimator.checkpoint = _expanduser(
        config.gaze_estimator.checkpoint)
    config.gaze_estimator.camera_params = _expanduser(
        config.gaze_estimator.camera_params)
    config.gaze_estimator.normalized_camera_params = _expanduser(
        config.gaze_estimator.normalized_camera_params)
    config.demo.image_path = _expanduser(config.demo.image_path)
    config.demo.video_path = _expanduser(config.demo.video_path)
    config.demo.output_dir = _expanduser(config.demo.output_dir)


def _check_path(config: yacs.config.CfgNode, key: str) -> None:
    path_str = operator.attrgetter(key)(config)
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'config.{key}: {path.as_posix()} not found.')
    if not path.is_file():
        raise ValueError(f'config.{key}: {path.as_posix()} is not a file.')


def _check_path_all(config: yacs.config.CfgNode) -> None:
    if config.face_detector.mode == 'dlib':
        _check_path(config, 'face_detector.dlib.model')
    _check_path(config, 'gaze_estimator.checkpoint')
    _check_path(config, 'gaze_estimator.camera_params')
    _check_path(config, 'gaze_estimator.normalized_camera_params')
    if config.demo.image_path:
        _check_path(config, 'demo.image_path')
    if config.demo.video_path:
        _check_path(config, 'demo.video_path')


def update_default_config(config: yacs.config.CfgNode,
                          args: argparse.Namespace) -> None:
    logger.debug('Called update_default_config()')
    if not args.mode:
        logger.debug('--mode is not specified.')
        raise ValueError
    else:
        logger.debug(f'--mode is {args.mode}')
        if args.mode == 'eye':
            config.mode = 'MPIIGaze'
            model_path = _download_eye_model()
            config.gaze_estimator.checkpoint = model_path.as_posix()
        elif args.mode == 'face':
            config.mode = 'MPIIFaceGaze'
            config.model.name = 'resnet_simple'
            model_path = _download_face_model()
            config.gaze_estimator.checkpoint = model_path.as_posix()
        else:
            raise ValueError
        logger.debug('Set config.gaze_estimator.checkpoint')
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


def update_config(config: yacs.config.CfgNode) -> None:
    if config.mode == 'MPIIGaze':
        _set_eye_default_camera(config)
    elif config.mode == 'MPIIFaceGaze':
        _set_face_default_camera(config)
    else:
        raise ValueError
    _update_camera_config(config)
    _download_dlib_pretrained_model(config)
    _expanduser_all(config)
    _check_path_all(config)

    config.freeze()
