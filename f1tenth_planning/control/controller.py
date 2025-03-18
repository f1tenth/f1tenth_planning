from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import dynamics_config

class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dynamics_config) -> None:
        """
        Initialize controller.

        Args:
            track (Track): track object with raceline
            params (dict | str, optional): dictionary or path to yaml with controller-specific parameters
        """
        self.track = track
        self.params = params
        self.waypoints = None
        self.waypoint_render = None

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
    def waypoints_color(self) -> tuple[int, int, int]:
        """
        Color as rgb tuple used for rendering waypoints (global plan).

        For example, we can visualize trajectories of different colors for different agents by changing this color.
        """
        return 0, 128, 0

    @waypoints_color.setter
    def waypoints_color(self, value: tuple[int, int, int]) -> None:
        """
        Set color as rgb tuple used for rendering waypoints (global plan).
        """
        assert len(value) == 3, f"color must be a tuple of length 3, got {value}"
        self.color = value

    def render_waypoints(self, e):
        """
        Callback to render waypoints.
        """
        if self.waypoints is not None:
            points = self.waypoints[:, :2]
            if self.waypoint_render is None:
                self.waypoint_render = e.render_closed_lines(
                    points, color=self.waypoints_color, size=1
                )
            else:
                self.waypoint_render.setData(points)
