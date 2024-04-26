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
        raceline = Raceline.from_raceline_path(path, delimiter)
        if fixed_speed is not None:
            raceline.velxs = np.full_like(raceline.ss, fixed_speed)
        return Trajectory(
            ss=raceline.ss,
            xs=raceline.xs,
            ys=raceline.ys,
            psis=raceline.psis,
            kappas=raceline.kappas,
            velxs=raceline.velxs,
            accxs=raceline.accxs,
        )

    def check_valid(self):
        pass

    def to_file(
        self,
        path: str | pathlib.Path,
        delimiter: str | None = ",",
        fixed_speed: float | None = 1.0,
    ):
        """Save trajectory to file, with optional fixed speed.

        Parameters
        ----------
        path : str | pathlib.Path
            path to save trajectory to
        delimiter : str, optional
            delimiter to use, by default ","

        Raises
        ------
        ValueError
            If file extension is not supported
        FileExistsError
            If file already exists
        FileNotFoundError
            If directory does not exist
        """
        if type(path) == str:
            path = pathlib.Path(path)
        # Check if path ends with a valid extension (csv, txt, npy, etc.)
        if path.suffix not in [".csv", ".txt", ".npy"]:
            raise ValueError(f"File extension {path.suffix} not supported.")
        # Check if file exists 
        if path.exists():
            raise FileExistsError(f"File {path} already exists.")
        # Check if directory exists
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory {path.parent} does not exist.")
        
        if fixed_speed is not None:
            vels = np.full_like(self.ss, fixed_speed)
        else:
            vels = self.velxs

        header = f'{delimiter} '.join(['s_m','x_m','y_m','psi_rad','kappa_radpm','vx_mps','ax_mps2'])
        np.savetxt(
            path,
            np.array([self.ss, self.xs, self.ys, self.psis, self.kappas, vels]).T,
            header=header,
            comments="# ",
            delimiter=delimiter,
        )
        return

    def subsample(self, 
                  jump: int = 1):
        """Subsample trajectory to reduce number of points.

        Parameters
        ----------
        jump : int, optional
            Number of points to downsample by, by default 1

        Returns
        -------
        Trajectory
            Subsampled trajectory

        Raises
        ------
        ValueError
            If jump is not a positive integer or if jump is greater than trajectory length
        """
        if jump <= 0:
            raise ValueError("Jump must be a positive integer.")
        if jump > len(self.ss):
            raise ValueError("Jump must be less than trajectory length.")
        
        return Trajectory(
            ss=self.ss[::jump],
            xs=self.xs[::jump],
            ys=self.ys[::jump],
            psis=self.psis[::jump],
            kappas=self.kappas[::jump],
            velxs=self.velxs[::jump],
            accxs=self.accxs[::jump],
        )

    def render(self):
        pass
