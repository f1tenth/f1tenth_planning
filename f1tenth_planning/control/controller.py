from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig
from f1tenth_planning.control.spec import VariableSpec, waypoints_from_raceline
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum


class Controller(ABC):
    """Base class for all controllers.

    A controller maps an observation to an actuation command:

        u = controller.compute_control(observation)

    There is deliberately **one** Controller type (DESIGN.md §2). A controller that
    tracks a raceline, one that reacts to a laser scan, and one that learns a safe set
    across laps are all Controllers -- holding a reference, reading sensors and
    learning are internal implementation details, not separate kinds of controller.

    **Configuration is runtime-changeable** via :meth:`update`, which is the single
    entry point for new vehicle parameters, new algorithm config, or a new reference.
    Controllers do *not* own a Planner; the caller wires the two together::

        reference = planner.plan(obs)
        controller.update(reference=reference)
        u = controller.compute_control(obs)

    **Rendering is not part of this interface** (DESIGN.md §9). Controllers expose what
    they computed as plain attributes (``waypoints``, ``local_plan``,
    ``control_solution``, and for the MPC family ``ref_traj`` / ``x_pred`` /
    ``u_pred``); the caller draws whatever it wants. See ``examples/control/``.
    """

    #: Reference fields this controller needs, in the column order it stores them.
    #: Used to build the waypoint matrix from a raceline by name. Controllers that
    #: derive their layout from a dynamics model (the MPC family) override
    #: :attr:`reference_field_names` instead.
    REFERENCE_FIELDS: tuple[str, ...] = ()

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

    # ------------------------------------------------------------------ required
    @abstractmethod
    def compute_control(self, state: dict) -> np.ndarray:
        """Compute the actuation command for the current observation.

        Args:
            state (dict): observation as returned from the environment.

        Returns:
            np.ndarray: control action, in the units declared by `control_mode`.
        """
        raise NotImplementedError("compute_control not implemented")

    # ------------------------------------------------------------- configuration
    @property
    def reference_field_names(self) -> tuple[str, ...]:
        """Names of the reference fields this controller consumes, in column order."""
        return self.REFERENCE_FIELDS

    @property
    def reference(self) -> VariableSpec:
        """Named layout of this controller's waypoint columns.

        Lets a controller index its own reference by name (``self.reference.idx.v``)
        instead of by a literal column number.
        """
        spec = getattr(self, "_reference_spec", None)
        names = self.reference_field_names
        if spec is None or spec.names != tuple(names):
            if not names:
                raise NotImplementedError(
                    f"{type(self).__name__} does not declare REFERENCE_FIELDS"
                )
            spec = VariableSpec(names)
            self._reference_spec = spec
        return spec

    def update(self, *, config=None, params=None, reference=None) -> None:
        """Reconfigure the controller at runtime.

        Args:
            config: algorithm configuration (horizon, costs, bounds, ...). Applied by
                :meth:`_apply_config`, which subclasses override.
            params (DynamicsConfig): vehicle parameters.
            reference: a gym ``Raceline``, or an already-built ``(N, m)`` waypoint
                array in this controller's own column order.

        Every argument is keyword-only and optional, so callers set exactly what they
        mean and the signature does not grow as parameters are added.
        """
        if params is not None:
            self.params = params
        if reference is not None:
            self.waypoints = self._waypoints_from_reference(reference)
        if config is not None:
            self._apply_config(config)

    def _waypoints_from_reference(self, reference) -> np.ndarray:
        """Convert a reference into this controller's waypoint layout."""
        if isinstance(reference, np.ndarray):
            return reference
        raceline = getattr(reference, "raceline", reference)  # accept a Track too
        if not hasattr(raceline, "xs"):
            raise TypeError(
                "reference must be a Raceline, a Track, or an (N, m) array; got "
                f"{type(reference).__name__}"
            )
        return waypoints_from_raceline(raceline, self.reference_field_names)

    def _apply_config(self, config) -> None:
        """Apply an algorithm configuration. Subclasses that have one override this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support config updates"
        )

    def get_config(self):
        """The controller's current algorithm configuration, if it has one."""
        return getattr(self, "config", None)

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        """Clear per-episode state (cached errors, warm starts, RNG).

        Default is a no-op; controllers holding state across steps override it.
        """
        return None

    def complete_iteration(self) -> None:
        """Signal the end of a lap/episode.

        Default is a no-op. Iterative-learning controllers use this to commit the
        completed trajectory (e.g. LMPC adding it to its safe set and refitting its
        value function).
        """
        return None
