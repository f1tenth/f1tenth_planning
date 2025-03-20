# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

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
Stanley waypoint tracker

Author: Hongrui Zheng, Johannes Betz
Last Modified: 5/1/22
"""

from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import dynamics_config, f1tenth_params
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import pi_2_pi

import numpy as np
import math


class StanleyController(Controller):
    """
    Stanley Controller implements a front wheel feedback controller for vehicle path tracking.

    This controller computes steering commands using the Stanley method, which combines
    the vehicle's heading error with the cross-track error relative to a reference path.
    It is suited for autonomous vehicles and employs a proportional gain to minimize tracking error.

    References:
      - Stanley: The Robot That Won the DARPA Grand Challenge:
          http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf
      - Autonomous Automobile Path Tracking:
          https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

    Args:
        track (Track): Racetrack instance containing raceline and waypoint information.
        params (dynamics_config, optional): Vehicle dynamics configuration parameters (default: f1tenth_params()).

    Attributes:
        waypoints (numpy.ndarray [N x 4]): Static list of waypoints to track; columns correspond to [x, y, velocity, heading].
        k_path (float): Proportional gain for cross-track error compensation.
        target_point (numpy.ndarray or None): The current target point on the track.
        target_index (int or None): Index of the current waypoint.
    """

    def __init__(self, track: Track, params: dynamics_config = f1tenth_params(), k_path=5.0):
        super(StanleyController, self).__init__(track, params)
        self.waypoints = np.vstack([
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.vxs,
            track.raceline.yaws
        ]).T
        self.k_path = k_path
        
        self.target_point_renderer = None
        self.local_plan_render = None
        self.target_point = None
        self.target_index = None

    def render_target_point(self, e):
        """
        Render the target point on the environment.

        Args:
            e: rendering engine instance used to visualize the target point.
        """
        if self.target_point is not None:
            points = self.target_point[:2][None]  # shape (1, 2)
            if self.target_point_renderer is None:
                self.target_point_renderer = e.render_points(
                    points, color=(128, 0, 0), size=4
                )
            else:
                self.target_point_renderer.setData(points)

    def render_local_plan(self, e):
        """
        Update the drawn waypoints (local plan) on the environment.

        Args:
            e: rendering engine instance used to visualize the local plan.
        """
        if self.target_index is not None:
            points = self.waypoints[self.target_index : self.target_index + 5, :2]
            if self.local_plan_render is None:
                self.local_plan_render = e.render_closed_lines(
                    points, color=(0, 0, 128), size=1
                )
            else:
                self.local_plan_render.setData(points)

    def calc_theta_and_ef(self, vehicle_state, waypoints):
        """
        Calculate the heading error and cross-track error relative to the path.

        Args:
            vehicle_state (numpy.ndarray [4,]): the state of the vehicle as [x, y, heading, velocity].
            waypoints (numpy.ndarray [N, 4]): the waypoints, where each row is [x, y, velocity, heading].

        Returns:
            theta_e (float): the heading error between the vehicle and the target waypoint.
            ef (numpy.ndarray): the cross-track error (signed distance) from the path.
            target_index (int): index of the closest waypoint.
            goal_veloctiy (float): target velocity at the closest waypoint.
        """
        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.params.WHEELBASE * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.params.WHEELBASE * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        self.target_point, nearest_dist, t, self.target_index = nearest_point(
            position_front_axle, self.waypoints[:, 0:2]
        )
        vec_dist_nearest_point = position_front_axle - self.target_point

        # compute cross-track error using the vehicle's orientation
        front_axle_vec_rot_90 = np.array(
            [
                [math.cos(vehicle_state[2] - math.pi / 2.0)],
                [math.sin(vehicle_state[2] - math.pi / 2.0)],
            ]
        )
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        # compute heading error (accounting for coordinate system differences)
        theta_raceline = waypoints[self.target_index, 3]
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # target velocity at the current waypoint
        goal_veloctiy = waypoints[self.target_index, 2]

        return theta_e, ef, self.target_index, goal_veloctiy

    def controller(self, vehicle_state, waypoints, k_path):
        """
        Compute the steering angle using the Stanley control algorithm.
        
        Args:
            vehicle_state (numpy.ndarray [4,]): the vehicle state [x, y, heading, velocity].
            waypoints (numpy.ndarray [N, 4]): the waypoints [x, y, velocity, heading].
            k_path (float): proportional gain for cross-track error compensation.

        Returns:
            delta (float): computed steering angle.
            goal_veloctiy (float): target velocity corresponding to the closest waypoint.
        """
        theta_e, ef, target_index, goal_veloctiy = self.calc_theta_and_ef(
            vehicle_state, waypoints
        )
        # compute the steering contribution from the cross-track error
        cte_front = math.atan2(k_path * ef, vehicle_state[3])
        delta = cte_front + theta_e

        return delta, goal_veloctiy

    def plan(self, state, waypoints=None, k_path=None):
        """
        Compute the control commands for trajectory tracking of the vehicle.

        Args:
            state (dict): Dictionary containing the vehicle's state with the following keys:
            - "pose_x" (float): x-coordinate of the vehicle's position.
            - "pose_y" (float): y-coordinate of the vehicle's position.
            - "pose_theta" (float): heading angle (in radians) of the vehicle.
            - "linear_vel_x" (float): current forward velocity of the vehicle.
            k_path (float, optional): Proportional gain for path tracking. If provided, it updates the controller's gain.
            waypoints (numpy.ndarray [N, 4], optional): Array of waypoints with shape (N, 4), where each waypoint
            is defined by [x, y, velocity, heading]. When provided, the controller's internal waypoint list is updated.

        Returns:
            tuple: A tuple containing:
            - steering_angle (float): The calculated steering angle command.
            - speed (float): The desired speed corresponding to the closest waypoint.

        Raises:
            ValueError: If neither internal waypoints exist nor are provided as an argument, or if the provided
                waypoints do not have at least 4 columns.
        """
        if waypoints is not None:
            if waypoints.shape[1] < 4 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints need to be a (N x m) numpy array with m >= 4!")
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please provide waypoints during controller instantiation or when calling plan()"
                )
            
        if k_path is not None:
            self.k_path = k_path

        k_path = np.float32(k_path)
        vehicle_state = np.array([
            state["pose_x"], 
            state["pose_y"], 
            state["pose_theta"], 
            state["linear_vel_x"]
        ])

        steering_angle, speed = self.controller(vehicle_state, self.waypoints, k_path)
        return steering_angle, speed
