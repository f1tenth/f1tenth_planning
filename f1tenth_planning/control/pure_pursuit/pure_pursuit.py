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
Last Modified: 5/1/22
"""

from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation

import numpy as np

class PurePursuitPlanner():
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 3], optional): static waypoints to track

    Attributes:
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 3]): static list of waypoints
    """
    def __init__(self, waypoints=None):
        self.max_reacquire = 20.
        self.waypoints = waypoints

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
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind],
                          self.waypoints[:, self.conf.wpt_yind])).T
        nearest_p, nearest_dist, t, i = nearest_point(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(position,
                                                      lookahead_distance,
                                                      wpts,
                                                      i + t,
                                                      wrap=True)
            if i2 is None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, self.conf.wpt_vind])
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
            waypoints (numpy.ndarray [N x 3], optional): list of dynamic waypoints to track

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(lookahead_distance,
                                                     position,
                                                     pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta,
                                              lookahead_point,
                                              position,
                                              lookahead_distance,
                                              self.wheelbase)

        return speed, steering_angle