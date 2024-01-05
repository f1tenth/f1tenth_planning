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
from f110_gym.envs.track import Track

from f1tenth_planning.control.controller import Controller
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation

import numpy as np
import warnings


class PurePursuitPlanner(Controller):
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (Track): track object with raceline
        params (dict, optional): dictionary of parameters, including wheelbase, lookahead dist, ...
    """

    def __init__(self, track: Track, params: dict = None):
        self.params = {
            "wheelbase": 0.33,
            "max_reacquire": 20.0,
            "lookahead_distance": 1.0,
            "vgain": 1.0,
        }
        self.params.update(params or {})

        self.waypoints = np.stack([track.raceline.xs, track.raceline.ys, track.raceline.vxs], axis=1)
        self.lookahead_point = None
        self.current_index = None

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
        elif nearest_dist < self.params["max_reacquire"]:
            return self.waypoints[i, :]
        else:
            return None

    def plan(self, state: dict) -> np.ndarray:
        """
        Planner plan function overload for Pure Pursuit, returns actuation.

        Args:
            state (dict): current vehicle state, keys are ["pose_x", "pose_y", "pose_theta", "velocity"]

        Returns:
            actuation (numpy.ndarray (2, )): actuation command (steering angle, speed)
        """
        assert self.waypoints is not None, "waypoints are not set"

        pose_x, pose_y, pose_theta = state["pose_x"], state["pose_y"], state["pose_theta"]
        position = np.array([pose_x, pose_y])

        lookahead_distance = np.float32(self.params["lookahead_distance"])
        self.lookahead_point = self._get_current_waypoint(
            lookahead_distance, position, pose_theta
        )

        if self.lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return np.zeros(2)

        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            lookahead_distance,
            self.params["wheelbase"],
        )

        # scale speed according to the velocity gain
        speed = speed * self.params["vgain"]

        return np.array([steering_angle, speed])
