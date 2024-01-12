from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f110_gym.envs.track import Track


class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dict | str = None) -> None:
        raise NotImplementedError("controller init method not implemented")

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        raise NotImplementedError("control method not implemented")

    @property
    def color(self) -> tuple[int, int, int]:
        """
        Color for rendering waypoints of this controller.
        """
        return 128, 0, 0

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        """
        Color for rendering waypoints of this controller.
        """
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
