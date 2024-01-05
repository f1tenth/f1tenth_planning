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

from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation

import numpy as np
import warnings


class PurePursuitPlanner:
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track

    Attributes:
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
    """

    def __init__(self, wheelbase=0.33, waypoints=None):
        self.max_reacquire = 20.0
        self.wheelbase = wheelbase
        self.waypoints = waypoints
        self.drawn_waypoints = []
        self.lookahead_point = None
        self.current_index = None

    def render_waypoints(self, e):
        """
        Callback to render waypoints.
        """
        points = self.waypoints[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.current_index is not None:
            points = self.waypoints[self.current_index : self.current_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=2)

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)
            theta (float): current vehicle heading

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """

        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.current_index, t2 = intersect_point(
                position,
                lookahead_distance,
                self.waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.current_index is None:
                return None
            current_waypoint = np.array(
                [
                    self.waypoints[self.current_index, 0],
                    self.waypoints[self.current_index, 1],
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
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
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
            self.wheelbase,
        )

        return steering_angle, speed
