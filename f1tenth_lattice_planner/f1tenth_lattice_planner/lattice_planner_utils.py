# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz, Aman Sinha, Matthew O'Kelly

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
Utility functions for the State Lattice Planner

Author: Hongrui Zheng
Last Modified: 5/27/21
"""

import numpy as np
from numba import njit

@njit(cache=True)
def create_grid(pose, width, length, res):
    """
    Creates a uniform grid given a center point for a grid, the width/length, and the resolution of the grid

    Args:
        pose (List): pose of the grid center point (x, y, theta) in the vehicle frame
        width (float): width of the grid in meters
        length (float): length of the grid in meters
        res (float): grid resolution, points/meter, uniform in both directions

    Returns:
        grid_x (numpy.ndarray (NxM)): x coordinates of the grid in vehicle frame
        grid_y (numpy.ndarray (NxM)): y coordinates of the grid in vehicle frame
        theta (float): goal theta for the grid points in vehicle frame
    """

    # TODO: create the range of xs and ys
    num_points_w = int(np.round(width / res))
    num_points_l = int(np.round(length / res))

    if not num_points_w % 2:
        num_points_w = num_points_w + 1
    if not num_points_l % 2:
        num_points_l = num_points_l + 1

    x = np.arange(-num_points_l // 2, num_points_l // 2, step=num_points_l)
    y = np.arange(-num_points_w // 2, num_points_w // 2, step=num_points_w)

    # meshgrid
    xs, ys = np.meshgrid(x, y)

    # TODO: mult res and add to center point
    pass

@njit(cache=True)
def grid_lookup():
    """
    Look up
    """
    pass