from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum

class Controller(ABC):
    """Base class for all controllers.

    A controller maps an observation to an actuation command. There is deliberately
    **one** Controller type (DESIGN.md §2): a controller that tracks a raceline, one
    that reacts to a laser scan, and one that learns a safe set across laps are all
    Controllers -- holding a reference, reading sensors and learning are internal
    implementation details, not separate kinds of controller.

    Rendering is **not** part of this interface (DESIGN.md §9). Controllers expose
    what they computed as plain attributes (``waypoints``, ``local_plan``,
    ``control_solution``, and for the MPC family ``ref_traj`` / ``x_pred`` /
    ``u_pred``); the caller draws whatever it wants from those. See
    ``examples/control/`` for render callbacks. Keeping the visualiser out of the
    library also keeps the library independent of any particular renderer API.
    """

    @abstractmethod
    def __init__(self, track: Track, params: DynamicsConfig, control_mode : tuple[SteerActionEnum, LongitudinalActionEnum]) -> None:
        """
        Initialize controller.

        Args:
            track (Track): track object with raceline
            params (DynamicsConfig): vehicle parameters used by this controller
            control_mode (tuple): the (steering, longitudinal) action semantics this
                controller emits. The caller must configure the environment to match.
        """
        self.track = track
        self.params = params
        self.control_mode = control_mode
        self.waypoints = None

    @abstractmethod
    def plan(self, state: dict, waypoints: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Plan control action given a state observation from the environment.

        Args:
            state (dict): observation as returned from the environment.
            waypoints (np.ndarray, optional): waypoints to track, overrides internal waypoints if provided.
            **kwargs: additional arguments for the controller. This can be used to update controller parameters.

        Returns:
            np.ndarray: control action, in the units declared by `control_mode`.
        """
        raise NotImplementedError("control method not implemented")
