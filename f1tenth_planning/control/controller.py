from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import dynamics_config
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum

class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dynamics_config, control_mode : tuple[SteerActionEnum, LongitudinalActionEnum]) -> None:
        """
        Initialize controller.

        Args:
            track (Track): track object with raceline
            params (dict | str, optional): dictionary or path to yaml with controller-specific parameters
        """
        self.track = track
        self.params = params
        self.control_mode = control_mode
        self.waypoints = None
        self.waypoint_render = None

    @abstractmethod
    def plan(self, state: dict, waypoints: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Plan control action given a state observation from the environment.

        Args:
            state (dict): observation as returned from the environment.
            waypoints (np.ndarray, optional): waypoints to track, overrides internal waypoints if provided.
            **kwargs: additional arguments for the controller. This can be used to update controller parameters.

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
                self.waypoint_render = e.get_lines_renderer(
                    points, color=self.waypoints_color, size=1
                )
            else:
                self.waypoint_render.update(points)
