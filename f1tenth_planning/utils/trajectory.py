from f1tenth_gym.envs.track.cubic_spline import CubicSpline2D
from f1tenth_gym.envs.track.raceline import Raceline
from typing import Sequence
import numpy as np
import pathlib


class Trajectory(Raceline):
    type: str
    positions: Sequence[Sequence[float]]
    poses: Sequence[float] = None
    theta: Sequence[float] = None
    v: Sequence[float] = None
    a: Sequence[float] = None
    steer: Sequence[float] = None
    steer_v: Sequence[float] = None

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray | None = None,
        ss: np.ndarray | None = None,
        psis: np.ndarray | None = None,
        kappas: np.ndarray | None = None,
        accxs: np.ndarray | None = None,
        spline: CubicSpline2D | None = None,
    ):
        super().__init__(xs, ys, velxs, ss, psis, kappas, accxs, spline)

    @staticmethod
    def from_file(
        path: str | pathlib.Path,
        delimiter: str | None = ",",
        fixed_speed: float | None = 1.0,
    ):
        if type(path) == str:
            path = pathlib.Path(path)
        assert path.exists(), f"Input file {path} does not exist."
        waypoints = np.loadtxt(path, delimiter=delimiter)

    def check_valid(self):
        pass

    def to_file(self):
        pass

    def subsample(self):
        pass

    def render(self):
        pass
