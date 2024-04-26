from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_gym.envs.track import Track


class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, config: dict | str = None) -> None:
        """Controller init

        Parameters
        ----------
        track : Track
            track object with raceline/centerline
        config : dict | str, optional
            dictionary or path to yaml with controller specific parameters, by default None

        Raises
        ------
        NotImplementedError
            controller init method not implemented
        """
        raise NotImplementedError("controller init method not implemented")

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        """Plan control action given a state observation from the environment.

        Parameters
        ----------
        state : dict
            observation as returned from the environment.

        Returns
        -------
        np.ndarray
            control action as (steering_angle, speed) or (steering_vel, acceleration)

        Raises
        ------
        NotImplementedError
            control method not implemented
        """
        raise NotImplementedError("control method not implemented")

    @abstractmethod
    def update(self, config: dict) -> None:
        """Updates setting of controller

        Parameters
        ----------
        config : dict
            configurations to update

        Raises
        ------
        NotImplementedError
            controller update method not implemented
        """
        raise NotImplementedError("controller update method not implemented.")

    @property
    def color(self) -> tuple[int, int, int]:
        """Color as rgb tuple used for controller-specific render. For example, we can visualize trajectories of different colors for different agents by changing this color.

        Returns
        -------
        tuple[int, int, int]
            RGB colors
        """
        return 128, 0, 0

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        """Set color as rgb tuple used for controller-specific render.

        Parameters
        ----------
        value : tuple[int, int, int]
            RGB colors
        """
        assert len(value) == 3, f"color must be a tuple of length 3, got {value}"
        self.color = value

    def render_waypoints(self, e):
        """Callback to render waypoints.

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.waypoints is not None:
            points = self.waypoints[:, :2]
            e.render_closed_lines(points, color=self.color, size=1)
