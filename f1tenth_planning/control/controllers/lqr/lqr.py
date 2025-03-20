# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz, Atsushi Sakai

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
LQR waypoint tracker
Implementation inspired by https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/lqr_steer_control/lqr_steer_control.py

Author: Hongrui Zheng, Johannes Betz, Atsushi Sakai
Last Modified: 5/5/22
"""

from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import dynamics_config, f1tenth_params
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.controller_config import lqr_config
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import update_matrix
from f1tenth_planning.utils.utils import solve_lqr
from f1tenth_planning.utils.utils import pi_2_pi

import numpy as np
import math


class LQRController(Controller):
    """
    Lateral Controller using LQR

    Args:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 5], optional, default=None): waypoints to track, columns are [x, y, velocity, heading, curvature]

    Attributes:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 5], optional, default=None): waypoints to track, columns are [x, y, velocity, heading, curvature]
        vehicle_control_e_cog (float): lateral error of cog to ref trajectory
        vehicle_control_theta_e (float): yaw error to ref trajectory
    """
    def __init__(self, 
                 track: Track, 
                 params: dynamics_config = f1tenth_params(), 
                 config: lqr_config = lqr_config(),
                 ):
        super(LQRController, self).__init__(track, params)
        self.waypoints = np.vstack([
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.vxs,
            track.raceline.yaws,
            track.raceline.ks
        ]).T

        self.config = config
        self.vehicle_control_e_cog = 0  # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0  # theta_e: yaw error to ref trajectory

        self.closest_point = None
        self.target_index = None

        self.closest_point_renderer = None
        self.local_plan_render = None

    def render_closest_point(self, e):
        """
        Callback to render the closest point.
        """
        if self.closest_point is not None:
            points = self.closest_point[:2][None]  # shape (1, 2)
            if self.closest_point_renderer is None:
                self.closest_point_renderer = e.render_points(
                    points, color=(128, 0, 0), size=4
                )
            else:
                self.closest_point_renderer.setData(points)


    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.target_index is not None:
            points = self.waypoints[self.target_index : self.target_index + 10, :2]
            if self.local_plan_render is None:
                self.local_plan_render = e.render_closed_lines(
                    points, color=(0, 0, 128), size=1
                )
            else:
                self.local_plan_render.setData(points)

    def calc_control_points(self, vehicle_state, waypoints):
        """
        Calculate the heading and cross-track errors and target velocity and curvature
        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track [x, y, velocity, heading, curvature]

        Returns:
            theta_e (float): heading error
            e_cog (float): lateral crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_veloctiy (float): target velocity
        """

        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.params.WHEELBASE * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.params.WHEELBASE * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        self.closest_point, nearest_dist, t, self.target_index = nearest_point(
            position_front_axle, self.waypoints[:, 0:2]
        )
        vec_dist_nearest_point = position_front_axle - self.closest_point

        # crosstrack error
        front_axle_vec_rot_90 = np.array(
            [
                [math.cos(vehicle_state[2] - math.pi / 2.0)],
                [math.sin(vehicle_state[2] - math.pi / 2.0)],
            ]
        )
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        # heading error
        # NOTE: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[self.target_index, 3]
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # target velocity
        goal_veloctiy = waypoints[self.target_index, 2]

        # reference curvature
        kappa_ref = self.waypoints[self.target_index, 4]

        # saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy

    def controller(
        self, vehicle_state, waypoints, ts, matrix_q, matrix_r, max_iteration, eps
    ):
        """
        Compute lateral control command.

        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track
            ts (float): discretization time step
            matrix_q ([float], len=4): weights on the states
            matrix_r ([float], len=1): weights on control input
            max_iteration (int): maximum iteration for solving
            eps (float): error tolerance for solving

        Returns:
            steer_angle (float): desired steering angle
            v_ref (float): desired velocity
        """

        # size of controlled states
        state_size = 4

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cg, yaw_ref, k_ref, v_ref = self.calc_control_points(
            vehicle_state, waypoints
        )

        # Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = update_matrix(
            vehicle_state, state_size, ts, self.params.WHEELBASE
        )

        matrix_state_ = np.zeros((state_size, 1))

        matrix_k_ = solve_lqr(
            matrix_ad_, matrix_bd_, matrix_q, matrix_r, eps, max_iteration
        )

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cog_old) / ts
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts

        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        # Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = k_ref * self.params.WHEELBASE

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, v_ref

    def plan(
        self, state:dict, waypoints=None, config : lqr_config = None
    ):
        """
        Compute lateral control command for vehicle tracking.

        Args:
            state (dict): Current vehicle state with keys:
            - "pose_x" (float): x-position of the vehicle.
            - "pose_y" (float): y-position of the vehicle.
            - "pose_theta" (float): heading (in radians) of the vehicle.
            - "linear_vel_x" (float): linear velocity of the vehicle.
            waypoints (numpy.ndarray [N x 5], optional): Array of waypoints to track, where each row is
            [x, y, velocity, heading, curvature]. If not none, this overrides the waypoints set during controller instantiation.
            config (lqr_config, optional): Configuration parameters for the LQR controller. If not N

        Returns:
            tuple:
            steering_angle (float): Desired steering angle in radians.
            speed (float): Target speed.
        """
        if waypoints is not None:
            if waypoints.shape[1] < 5 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 5, numpy array!")
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during controller instantiation or when calling plan()"
                )

        if config is not None:
            self.config = config

        # Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array([
            state["pose_x"],
            state["pose_y"],
            state["pose_theta"],
            state["linear_vel_x"]
        ])

        # Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(
            vehicle_state, 
            self.waypoints, 
            self.config.dt, 
            self.config.Q,
            self.config.R, 
            self.config.max_iterations, 
            self.config.eps
        )

        return steering_angle, speed
