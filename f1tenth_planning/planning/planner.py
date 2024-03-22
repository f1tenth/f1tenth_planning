from ..control.controller import Controller
from ..utils.trajectory import Trajectory

import numpy as np
from f1tenth_gym.envs.track import Track
from typing import Sequence, Union
from abc import abstractmethod


class Planner(Controller):

    @abstractmethod
    def plan(self, state: dict) -> Sequence[float] | np.ndarray | Trajectory:
        """Plan given a observation directly from env

        Parameters
        ----------
        state : dict
            observation as returned from the environment.

        Returns
        -------
        Union[Sequence[float] | np.ndarray | Trajectory]
               direct control actions of (steering_angle, speed) or (steering_vel, acceleration),
            or Trajectory dataclass object

        Raises
        ------
        NotImplementedError
            planning method not implemented
        """
        raise NotImplementedError("planning method not implemented")
