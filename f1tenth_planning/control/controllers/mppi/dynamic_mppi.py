"""
NMPC waypoint tracker using CasADi. On init, takes in model equation.
"""

import numpy as np
import jax.numpy as jnp
from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.controller_config import (
    dynamic_mppi_config,
    mpc_config,
)
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import Dynamic_Bicycle_Model
from f1tenth_planning.control.controllers.mppi.mppi import MPPI
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
from f1tenth_planning.utils.utils import jnp_to_np

class Dynamic_MPPI_Planner(Controller):
    """
    MPPI Controller, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        params: dynamics_config = f1tenth_params(),
        config: mpc_config = dynamic_mppi_config(),
    ):
        super(Dynamic_MPPI_Planner, self).__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl),
        )
        self.waypoints = np.vstack(
            [
                track.raceline.xs,  # x
                track.raceline.ys,  # y
                np.zeros_like(track.raceline.xs),  # steering angle reference
                track.raceline.vxs,  # v
                track.raceline.yaws,  # yaw
                np.zeros_like(track.raceline.xs),  # yaw rate reference
                np.zeros_like(track.raceline.xs),  # slip angle
            ]
        ).T

        u_min = np.array([self.params.MIN_DSTEER, self.params.MIN_ACCEL])
        u_max = np.array([self.params.MAX_DSTEER, self.params.MAX_ACCEL])
        self.x_min = jnp.array([
            -jnp.inf,  # x
            -jnp.inf,  # y
            self.params.MIN_STEER,  # delta
            self.params.MIN_SPEED,  # v
            -np.pi,  # yaw
            -np.inf,  # yaw rate
            -np.inf,  # slip angle
        ])
        self.x_max = jnp.array([
            jnp.inf,  # x
            jnp.inf,  # y
            self.params.MAX_STEER,  # delta
            self.params.MAX_SPEED,  # v
            np.pi,  # yaw
            np.inf,  # yaw rate
            np.inf,  # slip angle
        ])

        config.u_min = u_min
        config.u_max = u_max

        self.model = Dynamic_Bicycle_Model(self.params)
        self.solver = MPPI(config, self.model)
        # Override the reward function to use the MPPI cost
        self.solver._reward = self._reward
        # Override the step function to have clipped states and yaw wrapping
        self.solver._step = self._step        

        self.x_pred = None
        self.ref_traj = None

        self.control_solution = None
        self.local_plan = None

        self.mpc_solution_render = None
        self.local_plan_render = None

    def _reward(self, x, u, x_ref, Q, R):
        """
        Single-step reward calculated as the negative of the trajectory tracking error calculated using the quadratic cost on all but the yaw, where yaw error is calculated using yaw-normalized error and then scaled by the its factor in the cost matrix.
        The reward is negative because we want to minimize the cost.
        Args:
            x (jnp.ndarray): Current state of the vehicle.
            u (jnp.ndarray): Control input applied to the vehicle.
            x_ref (jnp.ndarray): Reference state to track.
        Returns:
            jnp.ndarray: The negative cost, which is the reward.
        """
        # Calculate the state error
        e = x - x_ref
        loss_p1 = e[:4]  # Position and velocity errors
        loss_p2 = e[5:]  # Yaw rate and slip angle errors
        cost = (
            loss_p1.T @ Q[:4, :4] @ loss_p1 +
            loss_p2.T @ Q[5:, 5:] @ loss_p2
        )
        # Calculate the yaw-normalized error
        yaw_error = jnp.arctan2(jnp.sin(e[4]), jnp.cos(e[4]))
        cost += (
            Q[4, 4] * yaw_error**2  # Yaw error scaled by its factor
        )
        # Add the control input cost
        cost += u.T @ R @ u
        return -cost
    
    def _step(self, x, u, p):
        """
        Single-step state prediction function.
        """
        next_state = self.solver.discretizer(self.solver.model.f_jax, x, u, p, self.solver.config.dt)
        # MODULO THE YAW ANGLE TO BE BETWEEN -PI AND PI
        # MODULO BY CREATING A NEW VECTOR TO AVOID INPLACE MODIFICATION
        next_state = jnp.array([
            next_state[0],  # x
            next_state[1],  # y
            next_state[2],  # delta
            next_state[3],  # v
            (next_state[4] + jnp.pi) % (2 * jnp.pi) - jnp.pi,  # yaw, wrapped to [-pi, pi]
            next_state[5],  # yaw rate
            next_state[6]   # slip angle
        ])
        # CLIP THE STATES TO BE BETWEEN THE MIN AND MAX VALUES
        next_state = jnp.clip(next_state, self.x_min, self.x_max)
        return next_state
    

    def render_control_solution(self, e):
        """
        Callback to render the lookahead point on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.x_pred is not None:
            if self.mpc_solution_render is None:
                self.mpc_solution_render = e.get_points_renderer(
                    self.control_solution.T, color=(128, 0, 0), size=4
                )
            else:
                self.mpc_solution_render.update(self.control_solution.T)

    def render_local_plan(self, e):
        """
        Render the local plan (series of waypoints) on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.ref_traj is not None:
            if self.local_plan_render is None:
                self.local_plan_render = e.get_lines_renderer(
                    self.local_plan, color=(0, 0, 128), size=4
                )
            else:
                self.local_plan_render.update(self.local_plan)

    def plan(self, state: dict, waypoints=None, params: dynamics_config = None, Q : np.ndarray = None, R: np.ndarray = None):
        """
        Compute the control input for the vehicle using a Kinematic MPC planner.

        Args:
            state (dict): Dictionary containing the vehicle's state.
            waypoints (numpy.ndarray [N x 5], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, delta, velocity, heading]. Overrides the static raceline if provided.
            params (dynamics_config, optional): Vehicle parameters for the dynamic model. If none, uses default.
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
            if Q.shape != (self.solver.config.Q.shape[0], self.solver.config.Q.shape[1]):
                raise ValueError(
                    f"Q must be of shape {self.solver.config.Q.shape}, got {Q.shape}"
                )

        if R is not None:   
            if R.shape != (self.solver.config.R.shape[0], self.solver.config.R.shape[1]):
                raise ValueError(
                    f"R must be of shape {self.solver.config.R.shape}, got {R.shape}"
                )

        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        x0 = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]])

        cx = self.waypoints[:, 0]
        cy = self.waypoints[:, 1]
        cv = self.waypoints[:, 3] 
        self.ref_traj = calc_interpolated_reference_trajectory(x, y, yaw, cx, cy, cv, self.solver.config.dt, self.solver.config.N, self.waypoints, yaw_idx=4).T.copy()

        p = None
        if params is not None:
            p = self.model.parameters_vector_from_config(params)
            self.params = params

        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj.T, p=p, Q=Q, R=R)
        self.x_pred = jnp_to_np(self.x_pred)
        self.u_pred = jnp_to_np(self.u_pred)

        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :])

        return np.array(self.u_pred[:, 0]).flatten(), {
            "predicted_state": self.x_pred,
            "predicted_control": self.u_pred,
            "steering_angle": self.x_pred[2, 1],
            "velocity": self.x_pred[3, 1],
        }
