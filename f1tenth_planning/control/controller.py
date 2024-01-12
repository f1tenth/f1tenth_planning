from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f110_gym.envs.track import Track


class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dict | str = None) -> None:
        """
        Initialize controller.

        Args:
            track (Track): track object with raceline
            params (dict | str, optional): dictionary or path to yaml with controller-specific parameters
        """
        raise NotImplementedError("controller init method not implemented")

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        """
        Plan control action given a state observation from the environment.

        Args:
            state (dict): observation as returned from the environment.

        Returns:
            np.ndarray: control action as (steering_angle, speed)
        """
        raise NotImplementedError("control method not implemented")

    @property
    def color(self) -> tuple[int, int, int]:
        """
        Color as rgb tuple used for controller-specific render.

        For example, we can visualize trajectories of different colors for different agents by changing this color.
        """
        return 128, 0, 0

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        """
        Set color as rgb tuple used for controller-specific render.
        """
        assert len(value) == 3, f"color must be a tuple of length 3, got {value}"
        self.color = value

    def render_waypoints(self, e):
        """
        Callback to render waypoints.
        """
        if self.waypoints is not None:
            points = self.waypoints[:, :2]
            e.render_closed_lines(points, color=self.color, size=1)


def load_params(default_params: dict, new_params: dict | str = None) -> dict:
    """
    Update default parameters with new parameters from a dict or yaml file.

    Args:
        default_params (dict): default parameters
        new_params (dict or str, optional): new parameters dict or path to yaml file
    """
    if isinstance(new_params, str):
        import yaml

        with open(new_params, "r") as f:
            new_params = yaml.safe_load(f)

    default_params.update(new_params or {})
    return default_params
