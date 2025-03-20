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
Pure Pursuit waypoint tracker

Author: Hongrui Zheng
Last Modified: 5/4/22
"""

from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.dynamics_config import dynamics_config, f1tenth_params

import numpy as np
import warnings


class PurePursuitPlanner(Controller):
    """
    Pure pursuit tracking controller.
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    This controller uses a lookahead circle to determine the target waypoint and computes the
    required steering and speed commands based on the vehicle's pose and a set of static or dynamic waypoints.

    Args:
        track (Track): Track instance containing the raceline information.
        params (dynamics_config, optional): Vehicle dynamic parameters. Defaults to dynamics_config().
        max_reacquire (float, optional): Maximum radius (in meters) to reacquire the current waypoint in case the vehicle drifts. Defaults to 20.0.

    Attributes:
        waypoints (numpy.ndarray [N x 4]): Static list of waypoints to track; columns correspond to [x, y, velocity, heading].
        lookahead_point (numpy.ndarray or None): The current lookahead point computed on the track.
        target_index (int or None): Index of the current waypoint.
    """

    def __init__(self, track: Track, params: dynamics_config = f1tenth_params(), max_reacquire=20.0):
        super(PurePursuitPlanner, self).__init__(track, params)
        self.waypoints = np.vstack([
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.vxs,
            track.raceline.yaws
        ]).T
        
        self.max_reacquire = max_reacquire
        self.lookahead_point = None
        self.target_index = None

        self.lookahead_point_renderer = None
        self.local_plan_render = None

    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)
            if self.lookahead_point_renderer is None:
                self.lookahead_point_renderer = e.render_points(
                    points, color=(128, 0, 0), size=4
                )
            else:
                self.lookahead_point_renderer.setData(points)

    def render_local_plan(self, e):
        """
        Render the local plan (series of waypoints) on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.target_index is not None:
            points = self.waypoints[self.target_index : self.target_index + 10, :2]
            if self.local_plan_render is None:
                self.local_plan_render = e.render_closed_lines(
                    points, color=(0, 0, 128), size=1
                )
            else:
                self.local_plan_render.setData(points)

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the lookahead circle intersection.

        Args:
            lookahead_distance (float): The lookahead distance to pick the next tracking point.
            position (numpy.ndarray): Current position of the vehicle as [x, y].
            theta (float): Current heading angle of the vehicle in radians.

        Returns:
            numpy.ndarray or None: The selected waypoint as [x, y, velocity],
            or None if no waypoint is found within the constraints.
        """
        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.target_index, t2 = intersect_point(
                position,
                lookahead_distance,
                self.waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.target_index is None:
                return None
            current_waypoint = np.array(
                [
                    self.waypoints[self.target_index, 0],
                    self.waypoints[self.target_index, 1],
                    self.waypoints[i, 2],
                ]
            )
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, waypoints=None):
        """
        Computes the steering angle and speed command based on the current vehicle state
        and the next waypoint derived from the lookahead circle.

        Args:
            pose_x (float): Current x-position of the vehicle.
            pose_y (float): Current y-position of the vehicle.
            pose_theta (float): Current heading of the vehicle in radians.
            lookahead_distance (float): Distance ahead of the vehicle to determine the target waypoint.
            waypoints (numpy.ndarray [N x 4], optional): Optional dynamic waypoints to use instead of the static raceline.
                Each waypoint should have columns [x, y, velocity, heading].

        Returns:
            tuple: (steering_angle (float), speed (float)) commands for the vehicle.
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
        position = np.array([pose_x, pose_y])
        lookahead_distance = np.float32(lookahead_distance)
        self.lookahead_point = self._get_current_waypoint(
            lookahead_distance, position, pose_theta
        )

        if self.lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return 0.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            lookahead_distance,
            self.params.WHEELBASE,
        )

        return steering_angle, speed
