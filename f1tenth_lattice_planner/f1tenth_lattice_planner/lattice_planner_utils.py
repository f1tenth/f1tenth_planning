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

Explanation for the integration can be found in the appendix of:
Kelly, Alonzo, and Bryan Nagy. "Reactive nonholonomic trajectory generation via parametric optimal control." The International Journal of Robotics Research 22.7-8 (2003): 583-601.
https://journals.sagepub.com/doi/pdf/10.1177/02783649030227008?casa_token=8l-0eehYEPwAAAAA:eEc6q4jf5QwGjg6kvyquRSVLCkV_v-W2Bklgoe9aevN9z6wk7CE6gUSxlfklGrN-hf6EYHT2bFaL

Author: Hongrui Zheng
Last Modified: 5/27/21
"""

import numpy as np
from numba import njit

# Constants
# steps in a single spline
NUM_STEPS = 100

ORDER = 3
if ORDER == 3:
    PARAM_MAT = np.array(
        [[    1.,     0.,     0.,    0.],
         [-11./2,     9.,  -9./2,    1.],
         [    9., -45./2,    18., -9./2],
         [ -9./2,  27./2, -27./2,  9./2]])
elif ORDER == 5:
    PARAM_MAT = np.array(
        [[      1.,      0.,      0.,    0.,     0.,     0.],
         [      0.,      0.,      0.,    0.,     1.,     0.],
         [      0.,      0.,      0.,    0.,     0.,   1./2],
         [ -575./8,     81.,  -81./8,    1., -85./4, -11./4],
         [  333./2, -405./2,   81./2, -9./2,    45.,   9./2],
         [ -765./8,  243./2, -243./8,  9./2, -99./4, -9./4]])
else:
    assert(1==0)


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
        grid (numpy.ndarray (Nx2)): x, y coordinates of the grid in vehicle frame
    """

    x, y, th = pose

    # create the range of xs and ys
    num_points_w = int(np.round(width / res))
    num_points_l = int(np.round(length / res))

    if not num_points_w % 2:
        num_points_w = num_points_w + 1
    if not num_points_l % 2:
        num_points_l = num_points_l + 1

    x = np.arange(-num_points_l // 2, num_points_l // 2, step=num_points_l)
    y = np.arange(-num_points_w // 2, num_points_w // 2, step=num_points_w)

    # meshgrid with rotation
    xs, ys = np.meshgrid(x, y)
    grid_coord = np.stack((np.ravel(xs), np.ravel(ys)))
    rot_mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot_grid = np.dot(rot_mat, grid_coord)

    # mult res and add to center point
    grid = res * rot_grid + np.array([[x], [y]])
    return grid.T


@njit(cache=True)
def idx_lookup(val, lut_keys, lut_step_size):
    """
    Locates the index for a given look up value

    Args:
        val (float): value to look up
        lut_keys (numpy.ndarray (M, )): list of keys in look up table
        lut_step_size (float): resolution of the look up table
    Returns:
        idx (int): index of the value to look up in the look up table
    """
    idx = np.searchsorted(lut_keys, val, side='right')
    temp = (val - lut_keys[idx - 1]) / lut_step_size
    if temp >= 0 and temp <= 0.5:
        idx -= 1
    return min(lut_keys.shape[0] - 1, idx)


@njit(cache=True)
def lookup(x, y, theta, kappa0, lut_x, lut_y, lut_theta, lut_kappa, lut, step_sizes):
    """
    Looks up for the spline parameters in the LUT

    Args:
        x, y, theta (float): delta from start state
        kappa0 (float): starting steering angle
        lut_x, lut_y, lut_theta, lut (numpy.ndarray): loaded look up table
    Returns:
        params (numpy.ndarray): [s, k0, k1, k2, k3], the spline parameter
    """
    x_idx = idx_lookup(x, lut_x, step_sizes[0])
    y_idx = idx_lookup(y, lut_y, step_sizes[1])
    theta_idx = idx_lookup(theta, lut_theta, step_sizes[2])
    kappa_idx = idx_lookup(kappa0, lut_kappa, step_sizes[3])
    params = lut[x_idx, y_idx, theta_idx, kappa_idx]
    return params


@njit(cache=True)
def params_to_coefs(params):
    """
    Preparation for integration

    Args:
        params (numpy.ndarray (5, )): looked up params
    Returns:
        coefs (numpy.ndarray (4, 1) or (6, 1)): prepared coef matrix
    """
    s = params[-1]
    s2 = s**2
    s3 = s**3
    if ORDER == 3:
        coefs = np.dot(PARAM_MAT, params[:-1])
        coefs[1] /= s
        coefs[2] /= s2
        coefs[3] /= s3
        return coefs
    if ORDER == 5:
        temp = np.concatenate((params[:4],
                               np.array([params[4] * s,
                               params[5] * s2])))
        coefs = np.dot(PARAM_MAT, temp)
        coefs[0] = params[0]
        coefs[1] = params[4]
        coefs[2] = params[5] / 2.
        s4 = s**4
        s5 = s**5
        coefs[3] /= s3
        coefs[4] /= s4
        coefs[5] /= s5
        return coefs


@njit(cache=True)
def get_curvature_theta(coefs, s_cur):
    """
    Get the curvature and the heading at a certain arc length

    Args:
        coefs: (numpy.ndarray (4, 1) or (6, 1)): coef matrix
        s_cur: (float): arc length to calculate curvature and heading
    Returns:
        kappa, theta (float): curvature and heading
    """
    kappa = 0.
    theta = 0.
    for i in range(coefs.shape[0]):
        temp = coefs[i] * s_cur**i
        kappa += temp
        theta += temp * s_cur / (i + 1)
    return kappa, theta


@njit(cache=True)
def integrate_path(params):
    """
    Integrate to find the states on the splines

    Args:
        params (numpy.ndarray (5, )): looked up spline params
    Returns:
        states (numpy.ndarray (NUM_STEP, 4)): states on the spline
    """
    N = NUM_STEPS
    coefs = params_to_coefs(params)
    states = np.empty((N, 4))
    states[0] = np.zeros(4,)
    states[0, 3] = coefs[0]
    dx = 0
    dy = 0
    x = 0
    y = 0
    ds = params[-1] / N
    theta_old = 0
    for k in range(1, N):
        sk = k * ds
        kappa_k, theta_k = get_curvature_theta(coefs, sk)
        dx = dx * (1 - 1 / k) + (np.cos(theta_k) + np.cos(theta_old)) / 2 / k
        dy = dy * (1 - 1 / k) + (np.sin(theta_k) + np.sin(theta_old)) / 2 / k
        x = sk * dx
        y = sk * dy
        states[k] = [x, y, theta_k, kappa_k]
        theta_old = theta_k
    return states
    pass


@njit(cache=True)
def integrate_all(params_list):
    """
    Integrate to find the states on a list of splines

    Args:
        params_list (numpy.ndarray (N, 5)): looked up spline params
    Returns:
        states (numpy.ndarray (N*NUM_STEP, 4)): states on the spline
    """
    points_list = np.empty((params_list.shape[0] * NUM_STEPS, 4))
    for i in range(params_list.shape[0]):
        # this returns a trajectory as Nx4 array
        # roll it so s is at the end
        points = integrate_path(np.roll(params_list[i], -1))
        points_list[i * NUM_STEPS:i * NUM_STEPS + NUM_STEPS, :] = points
    return points_list


@njit(cache=True)
def grid_lookup(grid, theta, kappa0, lut_x, lut_y, lut_theta, lut_kappa, lut, lut_stepsizes):
    """
    Look up for the entire grid for spline parameters in the LUT

    Args:
        grid (numpy.ndarray (Nx2)): array of x, y coordinates for the grid
        theta (float): heading of grid end poses
        kappa0 (float): starting steering angle
        lut_x, lut_y, lut_that, lut_kappa (numpy.ndarray (M, )): lut keys
        lut (numpy.ndarray (N0, N1, N2, N3, 5)): stored look up table
    """
    params_list = np.empty((grid.shape[0], 5))
    for i in range(grid.shape[0]):
        param = lookup(grid[i, 0], grid[i, 1], theta, kappa0, lut_x, lut_y,
                       lut_theta, lut_kappa, lut, lut_stepsizes)

        # only accept splines with reasonable arc lengths
        # arc length because it's a good indicator of wheter the optim failed
        if (param[0] < 3*np.linalg.norm(grid[i, :2])) and (param[0] > 0):
            params_list[i, :] = param
        else:
            params_list[i, 0] = 0.

    # only keep valid splines
    idx = params_list[:, 0] >= 0.0001
    params_list = params_list[idx]
    grid_filtered = grid[idx]
    states_list_local = integrate_all(params_list)
    return states_list_local, params_list, grid_filtered


@njit(cache=True)
def trans_traj_list(traj_list, trans, rot):
    """
    Transform splines into global frame

    Args:
        traj_list (numpy.ndarray (NxNUM_STEP, 4)): list of splines to be transformed
        trans (numpy.ndarray ()): translation vector
        rot (numpy.ndarray ()): rotation matrix
    Returns:
        new_traj_list (numpy.ndarray (NxNUM_STEP, 4)): list of splines in global frame
    """
    # input traj_list is N*N_samples X 4 ndarray
    # xy_list = np.ascontiguousarray(traj_list[:, 0:2].T)
    xy_list = traj_list[:, 0:2].T
    rot = np.ascontiguousarray(rot)
    trans = np.ascontiguousarray(trans)
    # get homogeneous coords
    homo_xy = np.ascontiguousarray(np.vstack((xy_list, np.zeros((1, traj_list.shape[0])), np.ones((1, traj_list.shape[0])))))
    # apply rotation
    rotated_xy = np.dot(rot, homo_xy)
    rotated_xy = rotated_xy / rotated_xy[3, :]
    # apply translation
    translated_xy = rotated_xy[0:3, :] + trans
    new_traj_list = np.zeros(traj_list.shape)
    new_traj_list[:, 0:2] = translated_xy[0:2, :].T
    new_traj_list[:, 2:4] = traj_list[:, 2:4]
    return new_traj_list


@njit(cache=True)
def trans_traj_list_multiple(traj_list_all, trans, rot):
    """
    Transform a list of splines into global frame

    Args:
        traj_list_all (?): list of trajectory
        trans (numpy.ndarray ()): translation vector
        rot (numpy.ndarray ()): rotation matrix
    """
    out = []
    for traj_list in traj_list_all:
        # if it's one of the guys with no traj
        if traj_list.shape == (1, 1):
            out.append(traj_list)
        else:
            out.append(trans_traj_list(traj_list, trans, rot))
    return out