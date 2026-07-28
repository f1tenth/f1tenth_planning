from typing import List, Callable
import numpy as np
import jax.numpy as jnp

from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.config.controller_config import (
    APMPPIConfig,
    dynamic_ap_mppi_config,
)
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.solvers import APMPPISolver


def make_state_min_constraint(x_min: np.ndarray):
    """
    Create a constraint function that penalizes states below x_min.
    Only considers states where x_min is finite (not -inf).

    Args:
        x_min: Minimum state bounds of shape (nx,). Use -np.inf for unconstrained states.

    Returns:
        Constraint function with signature (x, u) -> (N,) where positive = violation.
    """
    # Find indices where x_min is finite
    finite_mask = np.isfinite(x_min)
    x_min_filtered = jnp.array(x_min[finite_mask], dtype=jnp.float32)
    finite_indices = jnp.array(np.where(finite_mask)[0], dtype=jnp.int32)

    def state_min_constraint(x, u):
        """
        Constraint: x >= x_min for finite bounds.
        Returns positive values when violated (x < x_min).

        Args:
            x: State trajectory of shape (N, nx)
            u: Control trajectory of shape (N, nu)

        Returns:
            Constraint violation of shape (N,)
        """
        # Extract states at finite-bound indices: (N, n_finite)
        x_filtered = x[:, finite_indices]
        # Violation: x_min - x (positive when x < x_min)
        violation = jnp.maximum(0.0, x_min_filtered - x_filtered)
        # Return L2 norm of violations per timestep
        return jnp.linalg.norm(violation, axis=-1)

    return state_min_constraint


def make_state_max_constraint(x_max: np.ndarray):
    """
    Create a constraint function that penalizes states above x_max.
    Only considers states where x_max is finite (not +inf).

    Args:
        x_max: Maximum state bounds of shape (nx,). Use np.inf for unconstrained states.

    Returns:
        Constraint function with signature (x, u) -> (N,) where positive = violation.
    """
    # Find indices where x_max is finite
    finite_mask = np.isfinite(x_max)
    x_max_filtered = jnp.array(x_max[finite_mask], dtype=jnp.float32)
    finite_indices = jnp.array(np.where(finite_mask)[0], dtype=jnp.int32)

    def state_max_constraint(x, u):
        """
        Constraint: x <= x_max for finite bounds.
        Returns positive values when violated (x > x_max).

        Args:
            x: State trajectory of shape (N, nx)
            u: Control trajectory of shape (N, nu)

        Returns:
            Constraint violation of shape (N,)
        """
        # Extract states at finite-bound indices: (N, n_finite)
        x_filtered = x[:, finite_indices]
        # Violation: x - x_max (positive when x > x_max)
        violation = jnp.maximum(0.0, x_filtered - x_max_filtered)
        # Return L2 norm of violations per timestep
        return jnp.linalg.norm(violation, axis=-1)

    return state_max_constraint


class DynamicAPMPPIPlanner(MPCController):
    """
    Convenience class that uses AP-MPPI solver with dynamic bicycle model.

    This planner automatically sets up state limit constraints based on vehicle parameters
    when no config is provided. To customize bounds, pass your own APMPPIConfig with
    x_min, x_max, u_min, u_max set to your desired values.

    State: x = [x, y, delta, v, yaw, yaw_rate, beta]
    Control: u = [delta_v, a]

    Args:
        track: Track object containing the reference raceline.
        params: Vehicle parameters for the dynamic model. Defaults to f1tenth_params().
        model: Dynamics model object. Defaults to DynamicBicycleModel.
        config: AP-MPPI configuration. If None, creates default with state limit constraints
            and bounds from vehicle parameters. Pass your own config to customize.
        solver: AP-MPPI solver. If None, creates from config and model.
        use_state_limits: Whether to automatically add state limit constraints (only when config is None).
        ref_velocity_bounds: (v_min, v_max) for reference trajectory clipping. If None, uses config.x_min/x_max[3].
    """

    def __init__(
        self,
        track: Track,
        params: DynamicsConfig = None,
        model: DynamicsModel = None,
        config: APMPPIConfig = None,
        solver: APMPPISolver = None,
        use_state_limits: bool = True,
        ref_velocity_bounds=None,
    ):
        '''
        Initialize Dynamic AP-MPPI Planner.

        Args:
            track (Track): Track object containing the reference raceline.
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. Defaults to f1tenth_params().
            model (DynamicsModel, optional): Dynamics model object. Defaults to DynamicBicycleModel.
            config (APMPPIConfig, optional): AP-MPPI configuration. If None, creates default with state limit
                constraints and bounds from vehicle parameters. Pass your own config to customize bounds.
            solver (APMPPISolver, optional): AP-MPPI solver. If None, creates from config and model.
            use_state_limits (bool, optional): Whether to automatically add state limit constraints
                from vehicle parameters. Only used when config is None. Defaults to True.
            ref_velocity_bounds (tuple[float, float], optional): (v_min, v_max) bounds for clipping
                reference trajectory velocities. Use this to set operational speed limits that differ
                from the physical limits in config.x_min/x_max. If None, uses config bounds.
        '''
        if not isinstance(solver, APMPPISolver) and solver is not None:
            raise ValueError("Solver must be an instance of APMPPISolver")
        if not isinstance(model, DynamicsModel) and model is not None:
            raise ValueError("Model must be an instance of DynamicsModel")

        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)

        if config is None:
            # Build constraints from vehicle parameters
            constraints = []
            # x = [x, y, delta, v, yaw, yaw_rate, beta]
            default_x_min = np.array([
                -np.inf,           # x position: unconstrained
                -np.inf,           # y position: unconstrained
                params.MIN_STEER,  # steering angle
                params.MIN_SPEED,  # velocity
                -np.inf,           # yaw: unconstrained
                -np.inf,           # yaw_rate: unconstrained
                -np.inf,           # beta: unconstrained
            ])
            default_x_max = np.array([
                np.inf,            # x position: unconstrained
                np.inf,            # y position: unconstrained
                params.MAX_STEER,  # steering angle
                params.MAX_SPEED,  # velocity
                np.inf,            # yaw: unconstrained
                np.inf,            # yaw_rate: unconstrained
                np.inf,            # beta: unconstrained
            ])

            if use_state_limits:
                # Only add constraints if there are finite bounds
                if np.any(np.isfinite(default_x_min)):
                    constraints.append(make_state_min_constraint(default_x_min))
                if np.any(np.isfinite(default_x_max)):
                    constraints.append(make_state_max_constraint(default_x_max))

            config = dynamic_ap_mppi_config(constraints=constraints)

            # Set default state/control bounds from vehicle parameters
            config.x_min = default_x_min
            config.x_max = default_x_max
            config.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
            config.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])

            # Stability clipping is intentionally left OFF (+/-inf) for the states we
            # constrain. Clipping a quantity that a constraint also polices would stop
            # any rollout from violating it, so every penalty weight would look
            # equally feasible and the adaptive penalty would do nothing. Set
            # config.x_clip_min/max explicitly, and wider than the constrained bounds,
            # only if rollouts actually diverge.

        # If user passed their own config, respect their bounds (don't overwrite)

        if solver is None:
            solver = APMPPISolver(config, model)
        super(DynamicAPMPPIPlanner, self).__init__(
            track,
            solver,
            model,
            params,
            ref_velocity_bounds,
        )
