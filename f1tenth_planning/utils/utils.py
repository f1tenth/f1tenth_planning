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
Utility functions for motion planners

Author: Hongrui Zheng
Last Modified: 5/27/21
"""

import numpy as np
import math
from numba import njit

"""
Pure Pursuit utilities
"""
@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

"""
LQR utilities
"""

@njit(cache=True)
def solve_lqr(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K

    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration

    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                 np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K

@njit(cache=True)
def update_matrix(vehicle_state, state_size, timestep, wheelbase):
    """
    calc A and b matrices of linearized, discrete system.

    Args:
        vehicle_state:
        state_size:
        timestep:
        wheelbase:

    Returns:
        A:
        b:
    """

    #Current vehicle velocity
    v = vehicle_state[3]

    #Initialization of the time discrete A matrix
    matrix_ad_ = np.zeros((state_size, state_size))

    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    # b = [0.0, 0.0, 0.0, v / L].T
    matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_


"""
Geometry utilities
"""

@njit(cache=True)
def quat_2_rpy(x, y, z, w):
    """
    Converts a quaternion into euler angles (roll, pitch, yaw)

    Args:
        x, y, z, w (float): input quaternion

    Returns:
        r, p, y (float): roll, pitch yaw
    """
    t0 = 2. * (w * x + y * z)
    t1 = 1. - 2. * (x * x +  y * y)
    roll = math.atan2(t0, t1)

    t2 = 2. * (w * y - z * x)
    t2 = 1. if t2 > 1. else t2
    t2 = -1. if t2 < -1. else t2
    pitch = math.asin(t2)

    t3 = 2. * (w * z + x * y)
    t4 = 1. - 2. * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))

@njit(cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

# @njit(cache=True)
def sample_traj(clothoid, npts):
    traj = np.empty((npts, 4))
    for i in range(0, npts):
        s = i* (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = clothoid.Theta(s)
        traj[i, 3] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)

    return traj

def map_collision(point, map):
    """
    Returns whether a point is in collision with the map
    """
    pass