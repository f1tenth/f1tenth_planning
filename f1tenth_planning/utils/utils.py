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
import random

"""
Pure Pursuit utilities
"""

# @njit(cache=True)
def obsDict2oppoArray(obs):
    oppo_num = len(obs['poses_x'])
    res = np.empty((oppo_num, 3))
    for i in range(1, oppo_num):
        res[i] = np.array([obs['poses_x'][i], obs['poses_y'][i], obs['poses_theta'][i]])
    return res


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
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
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
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
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
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
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
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
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

    # Current vehicle velocity
    v = vehicle_state[3]

    # Initialization of the time discrete A matrix
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
    t1 = 1. - 2. * (x * x + y * y)
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


def rotate_along_point(p, origin, theta):
    """
    p: (n, 2)
    origin: (2, )
    """
    rot = get_rotation_matrix(theta)
    origin = origin.reshape(-1, 2)
    return origin.T + rot @ ((p - origin).T)


@njit(cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * math.pi:
        return angle - 2.0 * math.pi
    if angle < 0:
        return angle + 2.0 * math.pi

    return angle


# @njit(cache=True)
def sample_traj(clothoid, npts, v):
    traj = np.empty((npts, 5))
    for i in range(0, npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)

    return traj


@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix
        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)
        
        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def map_collision(points, dt, map_metainfo, eps=0.3):
    """
    Check wheter a point is in collision with the map

    Args:
        points (numpy.ndarray(N, 2)): points to check
        dt (numpy.ndarray(n, m)): the map distance transform
        map_metainfo (tuple (x, y, c, s, h, w, resol)): map metainfo
        eps (float, default=0.1): collision threshold
    Returns:
        collisions (numpy.ndarray (N, )): boolean vector of wheter input points are in collision

    """
    orig_x, orig_y, orig_c, orig_s, height, width, resolution = map_metainfo
    collisions = np.empty((points.shape[0],))
    for i in range(points.shape[0]):
        if dt[xy_2_rc(points[i, 0], points[i, 1], orig_x, orig_y, orig_c, orig_s, height, width, resolution)] <= eps:
            collisions[i] = True
        else:
            collisions[i] = False
    return np.ascontiguousarray(collisions)

@njit(cache=True)
def x2y_distances_argmin(X, Y):
    """
    X: (n, 2)
    Y: (m, 2)

    return (n, 1)
    """
    # pass
    n = len(X)
    min_idx = np.zeros(n)
    for i in range(n):
        diff = Y - X[i]  # (m, 2)
        # It is because numba does not support 'axis' keyword
        norm2 = diff * diff  # (m, 2)
        norm2 = norm2[:, 0] + norm2[:, 1]
        min_idx[i] = np.argmin(norm2)
    return min_idx

@njit(cache=True)
def get_trmtx(pose):
    """
    Get transformation matrix of vehicle frame -> global frame
    Args:
        pose (np.ndarray (3, )): current pose of the vehicle
    return:
        H (np.ndarray (4, 4)): transformation matrix
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = np.cos(th)
    sin = np.sin(th)
    H = np.array([[cos, -sin, 0., x], [sin, cos, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return H


@njit(cache=True)
def get_vertices(pose, length, width):
    """
    Utility function to return vertices of the car body given pose and size
    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width
    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    # H = get_trmtx(pose)
    # rl = H.dot(np.asarray([[-length / 2], [width / 2], [0.], [1.]])).flatten()
    # rr = H.dot(np.asarray([[-length / 2], [-width / 2], [0.], [1.]])).flatten()
    # fl = H.dot(np.asarray([[length / 2], [width / 2], [0.], [1.]])).flatten()
    # fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.], [1.]])).flatten()
    # rl = rl / rl[3]
    # rr = rr / rr[3]
    # fl = fl / fl[3]
    # fr = fr / fr[3]
    # vertices_1 = np.asarray([[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]])

    c = np.cos(pose[2])
    s = np.sin(pose[2])
    x, y = pose[0], pose[1]
    tl_x = -length/2 * c + width/2 * (-s) + x
    tl_y = -length / 2 * s + width / 2 * c + y
    tr_x = length/2 * c + width/2 * (-s) + x
    tr_y = length / 2 * s + width / 2 * c + y
    bl_x = -length/2 * c + (-width/2) * (-s) + x
    bl_y = -length / 2 * s + (-width / 2) * c + y
    br_x = length/2 * c + (-width/2) * (-s) + x
    br_y = length / 2 * s + (-width / 2) * c + y
    vertices = np.asarray([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
    # assert np.linalg.norm(vertices_1-vertices) < 1e-4
    # print(vertices_1, vertices)
    return vertices


@njit(cache=True)
def avgPoint(vertices):
    """
    Return the average point of multiple vertices
    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on
    Returns:
        avg (np.ndarray, (2,)): average point of the vertices
    """
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors
    Args:
        a, b, c (np.ndarray, (2,)): input vectors
    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector
    Args:
        pt (np.ndarray, (2,)): input vector
    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """
    Return the index of the vertex furthest away along a direction in the list of vertices
    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on
    Returns:
        idx (int): index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """
    Minkowski sum support function for GJK
    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
        d (np.ndarray, (2, )): direction to find the support along
    Returns:
        support (np.ndarray, (n, 2)): Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap
    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False
