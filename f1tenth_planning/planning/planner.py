from f1tenth_planning.control.controller import Controller

import numpy as np
from f110_gym.envs.track import Track
from typing import Sequence
from abc import abstractmethod


class Planner(Controller):

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        state : dict
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        raise NotImplementedError("planning method not implemented")
    

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        state : dict
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        raise NotImplementedError("planning method not implemented")
