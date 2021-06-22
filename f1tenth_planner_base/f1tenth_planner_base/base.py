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
Parent class definitions for motion planners

Author: Hongrui Zheng
Last Modified: 5/27/21
"""
from argparse import Namespace

class Planner:
    """
    Parent class for all planners in f1tenth_planning

    Init Args:
        conf (dict): dictionary containing parameters for planners
    """
    def __init__(self, conf):
        self.conf = Namespace(**conf)
        pass

    def plan(self, args):
        """
        Based on given argument, return motion plan for current frame

        Args:
            args (list): list of optional arguments

        Returns:
            TODO: potentially add support for acceleration and steering velocity
            speed (float): current desired speed
            steering_angle (float): current desired steering angle
        """
        return None, None

    def reset(self):
        """
        Reset all memorized state of the planner
        """
        pass

class TrackingPlanner(Planner):
    """
    Trajectory tracking planners parent class

    Init Args:
        conf (dict): dictionary containing parameters for planners, see specific planners for required parameters
    """
    def __init__(self, conf):
        super().__init__(conf)

    def update_waypoints(self, waypoints):
        """
        Updates the current waypoint

        Args:
            waypoints (numpy.ndarray (n, m)): first dimension is number of points, second dimension depends on the specific planner
        """
        self.waypoints = waypoints