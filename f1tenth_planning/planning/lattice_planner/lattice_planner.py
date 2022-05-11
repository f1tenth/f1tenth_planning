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
Lattice Planner

Author: Hongrui Zheng
Last Modified: 5/5/22
"""

from f1tenth_planning.utils.utils import nearest_point

from pyclothoids import Clothoid
import numpy as np

class LatticePlanner():
    """

    """
    def __init__(self, wheelbase=0.33, waypoints=None):
        """

        """
        self.wheelbase = wheelbase
        self.waypoints = waypoints

    def add_cost_function(self, func):
        """
        Add cost function to list for eval
        """
        pass

    def add_sample_function(self, func):
        """
        Add a custom sample function to create goal grid
        """
        pass

    def sample(self):
        """
        Sample a goal grid based on sample function. Given the current vehicle state, return a list of [x, y, theta] tuples
        """
        pass

    def eval(self):
        """
        Evaluate a list of generated clothoids based on added cost functions
        """
        pass

    def plan(self, pose_x, pose_y, pose_theta, velocity, waypoints=None):
        """
        Plan for next step

        Args:
            pose_x (float):
            pose_y (float):
            pose_theta (float):
            velocity (float):
            waypoints (numpy.ndarray [N, 5], optional, default=None):

        Returns:
            steering_angle (float):
            speed (float):
            selected_traj (numpy.ndarray [M, ])
        """
        # sample a grid based on current states




def sample_grid():
    x = np.linspace(0.2, 4, 10)
    y = np.linspace(-2, 2, 11)
    all_x = []
    all_y = []
    for x1 in x:
        for y1 in y:
            clothoid0 = Clothoid.G1Hermite(0, 0, 0, x1, y1, 0)
            curr_x, curr_y = clothoid0.SampleXY(100)
            all_x.extend(curr_x)
            all_y.extend(curr_y)
            # plt.scatter(curr_x, curr_y)
    # plt.show()
def test():
    for i in range(100):
        sample_grid()
if __name__ == '__main__':
    cProfile.run('test()')
    # test()