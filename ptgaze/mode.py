import enum


class GazeEstimationMode(enum.Enum):
    """Gaze estimation method, named after the dataset the model was trained on.

    The enum values are the `mode` strings used in the config files.
    """

    MPIIGAZE = "MPIIGaze"
    MPIIFACEGAZE = "MPIIFaceGaze"
    ETH_XGAZE = "ETH-XGaze"

    @property
    def input_is_face(self) -> bool:
        """Whether the model takes a normalized face image (as opposed to per-eye images)."""
        return self is not GazeEstimationMode.MPIIGAZE
