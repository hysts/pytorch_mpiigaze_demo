"""Tests for the face detector backend selection."""

import sys

import pytest
from omegaconf import OmegaConf

from ptgaze.head_pose_estimation import LandmarkEstimator


@pytest.mark.parametrize("mode", ["dlib", "face_alignment_dlib"])
def test_dlib_modes_report_missing_dlib(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    # Setting the module entry to None makes `import dlib` raise ImportError
    # regardless of whether dlib is installed.
    monkeypatch.setitem(sys.modules, "dlib", None)
    config = OmegaConf.create({"face_detector": {"mode": mode}, "device": "cpu"})

    with pytest.raises(RuntimeError, match=r"pip install ptgaze\[dlib\]"):
        LandmarkEstimator(config)


def test_unknown_mode_raises() -> None:
    config = OmegaConf.create({"face_detector": {"mode": "unknown"}})

    with pytest.raises(ValueError):  # noqa: PT011
        LandmarkEstimator(config)
