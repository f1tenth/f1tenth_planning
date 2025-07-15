from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.controller_config import mpc_config, kinematic_mpc_config
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from f1tenth_planning.control.dynamics_models.kinematic_model import Kinematic_Bicycle_Model
from f1tenth_planning.control.controllers.LTV_mpc.LTV_mpc import LTV_MPC_Solver
from f1tenth_planning.control.config.dynamics_config import dynamics_config, f1tenth_params
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
from f1tenth_gym.envs.track import Track

import numpy as np


class Kinematic_MPC_Planner(Controller):
    def __init__(
        self,
        track: Track,
        params: dynamics_config = f1tenth_params(), 
        config: mpc_config = kinematic_mpc_config(),
    ):
        super(Kinematic_MPC_Planner, self).__init__(track, params,
                                                        control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl))
        self.config = config
        self.waypoints = np.vstack([
            track.raceline.xs,                 # x
            track.raceline.ys,                 # y
            np.zeros_like(track.raceline.xs),  # steering angle reference
            track.raceline.vxs,                # v
            track.raceline.yaws,               # yaw
        ]).T
        
        x_min = np.array([-np.inf, -np.inf, self.params.MIN_STEER, self.params.MIN_SPEED, -np.inf])
        x_max = np.array([+np.inf, +np.inf, self.params.MAX_STEER, 5.0, +np.inf])
        u_min = np.array([self.params.MIN_DSTEER, self.params.MIN_ACCEL])
        u_max = np.array([self.params.MAX_DSTEER, self.params.MAX_ACCEL])
        self.model = Kinematic_Bicycle_Model(self.params)
        self.solver = LTV_MPC_Solver(self.config, self.model) 

        self.x_pred = None
        self.ref_traj = None

        self.control_solution = None
        self.local_plan = None

        self.mpc_solution_render = None
        self.local_plan_render = None

    def render_control_solution(self, e):
        """
        Callback to render the lookahead point on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.x_pred is not None:
            self.control_solution = self.x_pred[:2].T
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

    def plan(self, state:dict, waypoints=None, Q=None, R=None, Rd=None, P=None):
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
                raise ValueError("Waypoints need to be a (N x m) numpy array with m >= 3!")
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
        x0 = np.array([x, 
                       y, 
                       state["delta"], 
                       v, 
                       yaw
                       ])
        
        cx = self.waypoints[:, 0]
        cy = self.waypoints[:, 1]
        v_max_prev = np.mean(self.x_pred[3, :]) if self.x_pred is not None else v
        self.ref_traj = calc_interpolated_reference_trajectory(x, y, cx, cy, v_max_prev, self.config.dt, self.config.N, self.waypoints).T.copy()
        # Reference is in [0, 2pi] so convert to [-pi, pi]
        self.ref_traj[4, :] = (self.ref_traj[4, :] + np.pi) % (2 * np.pi) - np.pi

        # If the reference switches signs compared to current state (i.e jumps from -np.pi + eps to np+pi - eps),
        # we need to adjust the reference yaw to match the current state yaw.
        # This is to avoid large yaw errors that can cause the MPC to fail.
        condition = np.abs(self.ref_traj[4, :] - x0[4]) > np.pi
        self.ref_traj[4, condition] = self.ref_traj[4, condition] + 2 * np.pi * np.sign(x0[4] - self.ref_traj[4, condition])

        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj, Q, P, R, Rd)

        self.local_plan = self.ref_traj[:2].T
        self.control_solution = self.x_pred[:2].T

        return self.u_pred[:, 0]
