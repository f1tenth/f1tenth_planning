"""
NMPC waypoint tracker using CasADi. On init, takes in model equation.
"""

import numpy as np
import casadi as ca
from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.controller_config import mpc_config, dynamic_mpc_config
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import Dynamic_Bicycle_Model
from f1tenth_planning.control.controllers.nonlinear_mpc.nonlinear_mpc import (
    Nonlinear_MPC_Solver,
)
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum


class Dynamic_NMPC_Planner(Controller):
    """
    NMPC Controller, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        params: dynamics_config = f1tenth_params(),
        config: mpc_config = dynamic_mpc_config(),
    ):
        super(Dynamic_NMPC_Planner, self).__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl),
        )
        self.config = config
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

        x_min = np.array(
            [
                -np.inf,
                -np.inf,
                self.params.MIN_STEER,
                self.params.MIN_SPEED,
                -np.inf,
                -np.inf,
                -np.inf,
            ]
        )
        x_max = np.array(
            [
                +np.inf,
                +np.inf,
                self.params.MAX_STEER,
                self.params.MAX_SPEED,
                +np.inf,
                +np.inf,
                +np.inf,
            ]
        )
        u_min = np.array([self.params.MIN_DSTEER, self.params.MIN_ACCEL])
        u_max = np.array([self.params.MAX_DSTEER, self.params.MAX_ACCEL])
        
        self.config.x_min = x_min
        self.config.x_max = x_max
        self.config.u_min = u_min
        self.config.u_max = u_max

        self.model = Dynamic_Bicycle_Model(self.params)
        ipopt_opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 200,
                "acceptable_tol": 1e-2,
                "acceptable_obj_change_tol": 1e-3,
                "warm_start_init_point": "yes",
            },
            "print_time": 0,
        }
        self.solver = Nonlinear_MPC_Solver(self.config, self.model, ipopt_opts)

        self.x_pred = None
        self.ref_traj = None

        self.control_solution = None
        self.local_plan = None

        self.mpc_solution_render = None
        self.local_plan_render = None

    def __yaw_normalized_quadratic_error(self, st, ref, Q):
        state_error = st - ref
        all_but_yaw_mask = np.ones(self.config.nx, dtype=bool)
        all_but_yaw_mask[4] = False  # yaw is the 5th state
        cost = state_error[all_but_yaw_mask].T @ Q @ state_error[all_but_yaw_mask]
        # Calculate the yaw-normalized error
        yaw_error = ca.arctan2(ca.sin(state_error[4]), ca.cos(state_error[4]))
        cost += yaw_error.T @ Q[4, 4] @ yaw_error
        return cost
        
    def _terminal_cost_fn(self, st, ref, Q):
        return self.__yaw_normalized_quadratic_error(st, ref, Q)

    def _running_cost_fn(self, st, ref, Q, R, con):
        # State tracking cost
        cost = self.__yaw_normalized_quadratic_error(st, ref, Q)

        # Control input cost
        cost += con.T @ R @ con
        return cost

    def render_control_solution(self, e):
        """
        Callback to render the lookahead point on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.x_pred is not None:
            self.control_solution = np.array(self.x_pred[:2, :]).T
            if self.mpc_solution_render is None:
                self.mpc_solution_render = e.get_points_renderer(
                    self.control_solution, color=(128, 0, 0), size=4
                )
            else:
                self.mpc_solution_render.update(self.control_solution)

    def render_local_plan(self, e):
        """
        Render the local plan (series of waypoints) on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.ref_traj is not None:
            self.local_plan = self.ref_traj[:2].T
            if self.local_plan_render is None:
                self.local_plan_render = e.get_lines_renderer(
                    self.local_plan, color=(0, 0, 128), size=4
                )
            else:
                self.local_plan_render.update(self.local_plan)

    def plan(self, state: dict, waypoints=None, params: dynamics_config = None):
        """
        Compute the control input for the vehicle using a Kinematic MPC planner.

        Args:
            state (dict): Dictionary containing the vehicle's state.
            waypoints (numpy.ndarray [N x 5], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, delta, velocity, heading]. Overrides the static raceline if provided.
            Q (np.ndarray, optional): State cost matrix. Defaults to None.
            R (np.ndarray, optional): Control input cost matrix. Defaults to None.
            Rd (np.ndarray, optional): Control input derivative cost matrix. Defaults to None.
            P (np.ndarray, optional): Terminal cost matrix. Defaults to None.

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

        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        x0 = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]])

        cx = self.waypoints[:, 0]
        cy = self.waypoints[:, 1]
        v_max_prev = np.max(self.waypoints[:, 3]) if self.waypoints is not None else v
        self.ref_traj = calc_interpolated_reference_trajectory(
            x, y, cx, cy, v_max_prev, self.config.dt, self.config.N, self.waypoints
        ).T.copy()
        # Reference is in [0, 2pi] so convert to [-pi, pi]
        self.ref_traj[4, :] = (self.ref_traj[4, :] + np.pi) % (2 * np.pi) - np.pi

        # If the reference switches signs compared to current state (i.e jumps from -np.pi + eps to np+pi - eps),
        # we need to adjust the reference yaw to match the current state yaw.
        # This is to avoid large yaw errors that can cause the MPC to fail.
        self.ref_traj[4, :] = np.where(
            np.abs(self.ref_traj[4, :] - x0[4]) > np.pi,
            self.ref_traj[4, :] + 2 * np.pi * np.sign(x0[4] - self.ref_traj[4, :]),
            self.ref_traj[4, :],
        )

        opti_params = None
        if params is not None:
            opti_params = self.model.parameters_vector_from_config(params)
            self.params = params

        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, p=opti_params)

        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :]).T

        return np.array(self.u_pred[:, 0]).flatten(), {
            "predicted_state": self.x_pred,
            "predicted_control": self.u_pred,
            "steering_angle": self.x_pred[2, 1],
            "velocity": self.x_pred[3, 1],
        }
