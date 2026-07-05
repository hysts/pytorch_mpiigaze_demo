"""Convert a .pth training checkpoint to the safetensors file ptgaze loads.

The training repositories (hysts/pytorch_mpiigaze and its successors) save
pickle checkpoints, either as ``{"model": state_dict}`` or as a bare state
dict. ptgaze itself loads safetensors only, so self-trained checkpoints must
go through this one-time conversion.
"""

import argparse
import logging
import pathlib

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from .mode import GazeEstimationMode
from .models import create_model

logger = logging.getLogger(__name__)

_CLI_MODES = {
    "mpiigaze": GazeEstimationMode.MPIIGAZE,
    "mpiifacegaze": GazeEstimationMode.MPIIFACEGAZE,
    "eth-xgaze": GazeEstimationMode.ETH_XGAZE,
}


def _reference_config(mode: GazeEstimationMode) -> DictConfig:
    model: dict[str, object]
    if mode is GazeEstimationMode.MPIIGAZE:
        model = {"name": "resnet_preact"}
    elif mode is GazeEstimationMode.MPIIFACEGAZE:
        # `pretrained` is empty so that building the reference model does not
        # download ImageNet weights; the state dict keys are the same either way.
        model = {
            "name": "resnet_simple",
            "backbone": {
                "name": "resnet_simple",
                "pretrained": "",
                "resnet_block": "basic",
                "resnet_layers": [2, 2, 2],
            },
        }
    else:
        model = {"name": "resnet18"}
    return OmegaConf.create({"mode": mode.value, "device": "cpu", "model": model})


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Return the model state dict from a loaded checkpoint object.

    Accepts both the ``{"model": state_dict}`` layout written by the training
    repositories and a bare state dict.
    """
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        error_msg = f"The checkpoint must be a dict, but got {type(checkpoint)}."
        raise TypeError(error_msg)
    if not state_dict:
        raise ValueError("The checkpoint contains an empty state dict.")
    non_tensors = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        error_msg = f"The checkpoint is not a state dict: non-tensor values at {non_tensors}."
        raise TypeError(error_msg)
    return state_dict


def validate_state_dict(state_dict: dict[str, torch.Tensor], mode: GazeEstimationMode) -> None:
    """Check the state dict keys and shapes against the architecture used by the given mode."""
    reference = create_model(_reference_config(mode)).state_dict()
    missing = sorted(reference.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - reference.keys())
    if missing or unexpected:
        error_msg = (
            f"The state dict does not match the {mode.value} architecture. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
        raise ValueError(error_msg)
    mismatched = [
        f"{key}: {tuple(state_dict[key].shape)} != {tuple(reference[key].shape)}"
        for key in reference
        if state_dict[key].shape != reference[key].shape
    ]
    if mismatched:
        error_msg = f"The state dict does not match the {mode.value} architecture. Shape mismatches: {mismatched}."
        raise ValueError(error_msg)


def convert(input_path: pathlib.Path, output_path: pathlib.Path, mode: GazeEstimationMode | None = None) -> None:
    if output_path.suffix != ".safetensors":
        error_msg = f"The output file must have a .safetensors extension, but got {output_path.name}."
        raise ValueError(error_msg)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    metadata = {"converted_by": "ptgaze-convert"}
    if mode is not None:
        validate_state_dict(state_dict, mode)
        metadata["mode"] = mode.value
        metadata["model"] = str(_reference_config(mode).model.name)
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_file(state_dict, output_path, metadata=metadata)
    logger.info("Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Input .pth checkpoint.")
    parser.add_argument("output", type=pathlib.Path, help="Output .safetensors file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(_CLI_MODES),
        help="If specified, the state dict is validated against the architecture "
        "used by this mode, and the mode is recorded in the file metadata.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(args.input, args.output, _CLI_MODES[args.mode] if args.mode else None)
