import pathlib

import pytest
from omegaconf import OmegaConf

import ptgaze
from ptgaze.mode import GazeEstimationMode


@pytest.mark.parametrize(
    ("value", "input_is_face"),
    [
        ("MPIIGaze", False),
        ("MPIIFaceGaze", True),
        ("ETH-XGaze", True),
    ],
)
def test_mode_parses_config_value(value: str, input_is_face: bool) -> None:
    mode = GazeEstimationMode(value)
    assert mode.value == value
    assert mode.input_is_face == input_is_face


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="is not a valid GazeEstimationMode"):
        GazeEstimationMode("UnknownMode")


def test_packaged_configs_have_valid_modes() -> None:
    config_dir = pathlib.Path(ptgaze.__file__).parent / "data/configs"
    config_paths = sorted(config_dir.glob("*.yaml"))
    assert len(config_paths) == 3
    modes = {GazeEstimationMode(OmegaConf.load(path).mode) for path in config_paths}
    assert modes == set(GazeEstimationMode)
