import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.mpc_solver import MPCSolver
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
from f1tenth_planning.control.spec import RACELINE_ATTR_FOR_STATE
from f1tenth_planning.utils.utils import jnp_to_np


def reference_waypoints_for_model(track: Track, model: DynamicsModel) -> np.ndarray:
    """Build an (N_waypoints, nx) reference matrix in `model`'s state layout.

    Columns are matched to the raceline **by variable name**, so adding a model with a
    different state vector needs no change here. Names the raceline does not carry
    (steering angle, yaw rate, slip angle) are zero-filled.
    """
    raceline = track.raceline
    n = len(raceline.xs)
    columns = []
    for name in model.state.names:
        attr = RACELINE_ATTR_FOR_STATE.get(name)
        values = getattr(raceline, attr, None) if attr else None
        if values is None:
            columns.append(np.zeros(n))
        else:
            columns.append(np.asarray(values, dtype=float))
    return np.vstack(columns).T


class MPCController(Controller):
    """
    MPPI Controller, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        solver (MPCSolver): MPC solver object, contains MPC parameters
        model (DynamicsModel): dynamics model object, contains the vehicle dynamics
        params (DynamicsConfig, optional): Vehicle parameters for the model. If none,
            default f1tenth_params() will be used.
        ref_velocity_bounds (tuple[float, float], optional): (v_min, v_max) bounds for clipping reference trajectory velocities.
            If None, uses the solver's state bounds at the model's velocity index. Use
            this to set operational speed limits that differ from the physical limits
            used by the solver for rollout clipping.

    The waypoint matrix and the initial state are both built in the *model's* state
    layout (DESIGN.md §5.2), so a 5-state kinematic model and a 7-state dynamic model
    are handled by the same code with no slicing hook.
    """

    def __init__(
        self,
        track: Track,
        solver: MPCSolver,
        model: DynamicsModel,
        params: DynamicsConfig = None,
        ref_velocity_bounds=None,
    ):
        if params is None:
            params = f1tenth_params()
        super().__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl),
        )
        self.model = model
        self.solver = solver

        # Reference waypoints in the model's own state layout: each column is the
        # raceline field that matches the state variable's name, zeros where the
        # raceline carries nothing for it (e.g. steering angle, yaw rate, slip).
        self.waypoints = reference_waypoints_for_model(track, model)

        # Reference velocity bounds (for clipping reference trajectory)
        v_idx = model.state.index("v")
        if ref_velocity_bounds is None:
            self.ref_v_min = self.solver.config.x_min[v_idx]
            self.ref_v_max = self.solver.config.x_max[v_idx]
        else:
            self.ref_v_min, self.ref_v_max = ref_velocity_bounds

        self.x_pred = None
        self.ref_traj = None

        self.control_solution = None
        self.local_plan = None

    def plan(
        self,
        state: dict,
        waypoints=None,
        params: DynamicsConfig = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        """
        Compute the control input for the vehicle using a Kinematic MPC planner.

        Args:
            state (dict): Dictionary containing the vehicle's state.
            waypoints (numpy.ndarray [N x 5], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, delta, velocity, heading]. Overrides the static raceline if provided.
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none, uses default.
            Q (np.ndarray, optional): State cost matrix. If none, uses default.
            R (np.ndarray, optional): Control input cost matrix. If none, uses default.

        Returns:
            control: A tuple (steering_vel, acc) representing the computed steering velocity and acceleration.
            If no valid lookahead point is found, returns (0.0, 0.0) after issuing a warning.
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError(
                    "Waypoints need to be a (N x m) numpy array with m >= 3!"
                )
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )

        if Q is not None:
            if Q.shape != (
                self.solver.config.Q.shape[0],
                self.solver.config.Q.shape[1],
            ):
                raise ValueError(
                    f"Q must be of shape {self.solver.config.Q.shape}, got {Q.shape}"
                )

        if R is not None:
            if R.shape != (
                self.solver.config.R.shape[0],
                self.solver.config.R.shape[1],
            ):
                raise ValueError(
                    f"R must be of shape {self.solver.config.R.shape}, got {R.shape}"
                )

        # The model assembles its own state vector from the observation, so this works
        # for any state layout (5-state kinematic, 7-state dynamic, ...).
        x0 = self.model.state_from_observation(state)

        idx = self.model.state.idx
        x = state["pose_x"]
        y = state["pose_y"]
        yaw = state["pose_theta"]

        cx = self.waypoints[:, idx.x]
        cy = self.waypoints[:, idx.y]
        cv = self.waypoints[:, idx.v]

        # Clip the reference velocity to the operational speed limits
        # --> Ensures that the interpolated trajectory does not assume
        #     the vehicle will go faster than it should.
        clipped_velocity = np.clip(cv, a_min=self.ref_v_min, a_max=self.ref_v_max)

        self.ref_traj = calc_interpolated_reference_trajectory(
            x,
            y,
            yaw,
            cx,
            cy,
            clipped_velocity,
            self.solver.config.dt,
            self.solver.config.N,
            self.waypoints,
        ).T.copy()
        p = None
        if params is not None:
            p = self.model.parameters_vector_from_config(params)
            self.params = params

        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, p=p, Q=Q, R=R)
        self.x_pred = jnp_to_np(self.x_pred)
        self.u_pred = jnp_to_np(self.u_pred)
        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :])
        return np.array(self.u_pred[:, 0]).flatten()
