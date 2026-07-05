"""Tests for the .pth -> safetensors converter."""

import pathlib

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ptgaze.convert import _reference_config, convert, extract_state_dict, validate_state_dict
from ptgaze.mode import GazeEstimationMode
from ptgaze.models import create_model


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "conv.weight": torch.randn(4, 1, 3, 3, generator=generator),
        "fc.weight": torch.randn(2, 4, generator=generator),
        "fc.bias": torch.randn(2, generator=generator),
    }


def _mpiigaze_state_dict() -> dict[str, torch.Tensor]:
    return create_model(_reference_config(GazeEstimationMode.MPIIGAZE)).state_dict()


@pytest.mark.parametrize("wrap_in_model_key", [True, False])
def test_roundtrip(tmp_path: pathlib.Path, wrap_in_model_key: bool) -> None:
    state_dict = _tiny_state_dict()
    checkpoint = {"model": state_dict} if wrap_in_model_key else state_dict
    input_path = tmp_path / "checkpoint.pth"
    output_path = tmp_path / "model.safetensors"
    torch.save(checkpoint, input_path)

    convert(input_path, output_path)

    reloaded = load_file(output_path)
    assert reloaded.keys() == state_dict.keys()
    for key in state_dict:
        assert torch.equal(reloaded[key], state_dict[key])


def test_roundtrip_with_mode_validation_and_metadata(tmp_path: pathlib.Path) -> None:
    state_dict = _mpiigaze_state_dict()
    input_path = tmp_path / "checkpoint.pth"
    output_path = tmp_path / "model.safetensors"
    torch.save({"model": state_dict}, input_path)

    convert(input_path, output_path, GazeEstimationMode.MPIIGAZE)

    reloaded = load_file(output_path)
    for key in state_dict:
        assert torch.equal(reloaded[key], state_dict[key])
    with safe_open(output_path, framework="pt") as f:
        metadata = f.metadata()
    assert metadata["mode"] == "MPIIGaze"
    assert metadata["model"] == "resnet_preact"


def test_validation_rejects_wrong_architecture(tmp_path: pathlib.Path) -> None:
    input_path = tmp_path / "checkpoint.pth"
    output_path = tmp_path / "model.safetensors"
    torch.save({"model": _tiny_state_dict()}, input_path)

    with pytest.raises(ValueError, match="does not match"):
        convert(input_path, output_path, GazeEstimationMode.MPIIGAZE)


def test_validation_rejects_shape_mismatch() -> None:
    state_dict = _mpiigaze_state_dict()
    state_dict["fc.weight"] = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="Shape mismatches"):
        validate_state_dict(state_dict, GazeEstimationMode.MPIIGAZE)


def test_output_extension_is_enforced(tmp_path: pathlib.Path) -> None:
    input_path = tmp_path / "checkpoint.pth"
    torch.save(_tiny_state_dict(), input_path)

    with pytest.raises(ValueError, match="safetensors extension"):
        convert(input_path, tmp_path / "model.pth")


def test_extract_state_dict_rejects_non_state_dict() -> None:
    with pytest.raises(TypeError, match="non-tensor values"):
        extract_state_dict({"optimizer": {"lr": 0.1}, "step": torch.tensor(1)})
    with pytest.raises(TypeError, match="must be a dict"):
        extract_state_dict([1, 2, 3])
    with pytest.raises(ValueError, match="empty"):
        extract_state_dict({})
