from abc import abstractmethod, ABC

import numpy as np
from f110_gym.envs.track import Track


class Controller(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dict = None) -> None:
        raise NotImplementedError("controller init method not implemented")

    @abstractmethod
    def plan(self, state: dict) -> np.ndarray:
        raise NotImplementedError("control method not implemented")

    def render_waypoints(self, e):
        """
        Callback to render waypoints.
        """
        if self.waypoints is not None:
            points = self.waypoints[:, :2]
            e.render_closed_lines(points, color=(128, 0, 0), size=1)
