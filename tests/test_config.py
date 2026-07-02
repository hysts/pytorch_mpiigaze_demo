import pathlib

import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

import ptgaze
from ptgaze.common import Camera
from ptgaze.config_schema import AppConfig


def _schema() -> object:
    return OmegaConf.structured(AppConfig)


def test_packaged_configs_conform_to_schema() -> None:
    config_dir = pathlib.Path(ptgaze.__file__).parent / "data/configs"
    config_paths = sorted(config_dir.glob("*.yaml"))
    assert len(config_paths) == 3
    for path in config_paths:
        OmegaConf.merge(_schema(), OmegaConf.load(path))


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigKeyError):
        OmegaConf.merge(_schema(), OmegaConf.create({"demo": {"unknown_key": 1}}))


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        OmegaConf.merge(_schema(), OmegaConf.create({"demo": {"wait_time": "not-an-int"}}))


def test_dummy_camera_matches_generated_params() -> None:
    # The dummy camera must reproduce what generate_dummy_camera_params()
    # historically wrote to a temporary YAML file: focal length equal to the
    # image width, principal point at the image center, no distortion.
    camera = Camera.create_dummy(640, 480)
    np.testing.assert_allclose(camera.camera_matrix, [[640, 0, 320], [0, 640, 240], [0, 0, 1]])
    np.testing.assert_allclose(camera.dist_coefficients, np.zeros((5, 1)))
    assert (camera.width, camera.height) == (640, 480)
