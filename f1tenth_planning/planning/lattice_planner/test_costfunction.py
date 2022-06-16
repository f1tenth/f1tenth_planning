import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from f1tenth_planning.utils.utils import \
    intersect_point, nearest_point, get_rotation_matrix, rotate_along_point, sample_traj, zero_2_2pi
from pyclothoids import Clothoid
from numba import njit
import matplotlib
import matplotlib.cm as cm

module = os.path.dirname(os.path.abspath(__file__))

"""
waypoints: [x, y, v, heading, kappa]
grid: [x, y, heading, v], (n, 4)
"""
def read_optimalwp(wpFile_path=os.path.join(module, 'example_waypoints.csv')):
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=1)
    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]
    # waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
    return np.array(wp_x), np.array(wp_y), waypoints


def draw_pts(pts, ax, c=None, mksize=5.0, label=None):
    """
    pts: (2, n)
    """
    x = pts[0]
    y = pts[1]
    if c:
        ax.plot(x, y, 'o', c=c, markersize=mksize, label=label)
    else:
        ax.plot(x, y, markersize=mksize, label=label, linewidth=0.5)


# @njit(cache=True)
def sample_lookahead_square(pose_x,
                            pose_y,
                            pose_theta,
                            velocity,
                            waypoints,
                            lookahead_distances=(1.0, 1.5, 2.0, 2.5),
                            widths=np.linspace(-1.3, 1.3, num=7)):
    """
    Example function to sample goal points. In this example it samples a rectangular grid around a look-ahead point.

    TODO: specify waypoints idx somehow? as planner arguments?

    Args:
        pose_x ():
        pose_y ():
        pose_theta ():
        velocity ():
        waypoints ():
        lookahead_distances ():
        widths ():

    Returns:
        grid (numpy.ndarray (n, 3)): Returned grid of goal points
    """
    # get lookahead points to create grid along waypoints
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    local_span = np.vstack((np.zeros_like(widths), widths))
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(nearest_p, d, waypoints[:, 0:2], t + nearest_i, wrap=True)
        i2 = int(i2)  # for numba, explicitly set the int type
        lh_pt_theta = waypoints[i2, 3]
        lh_pt_v = waypoints[i2, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span + lh_pt.reshape(2, -1)
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v
    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, len(widths)).reshape(1, -1)
    v_grid = np.repeat(v_grid, len(widths)).reshape(1, -1)
    grid = np.vstack((xy_grid, theta_grid, v_grid)).T
    return grid


def value2color(value):
    """
    value: np.ndarray (n, )
    """

    minima = np.min(value)
    maxima = np.max(value)
    rgba = []

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    for v in value:
        rgba.append(mapper.to_rgba(v))

    return rgba

def draw_lattice_grid(truncation=50):
    wp_x, wp_y, waypoints = read_optimalwp()
    print(waypoints.shape)
    wp_x = wp_x[:truncation]
    wp_y = wp_y[:truncation]
    waypoints = waypoints[:truncation, :]

    fake_position = np.array([wp_x[truncation//2], wp_y[truncation//2]])
    fake_position += np.random.random(2)
    fake_pose_theta = np.pi
    print(f'fake_position: {fake_position}')
    nearest_p, nearest_dist, t, nearest_i = nearest_point(fake_position, waypoints[:, 1:3])
    print(f'nearest_p: {nearest_p}, ratio: {t}, idx: {nearest_i}')
    goal_grid = sample_lookahead_square(fake_position[0], fake_position[1], fake_pose_theta, 0.0, waypoints)
    xy_grid = goal_grid[:, :2]

    # clo
    all_traj = []
    all_traj_clothoid = []
    for point in goal_grid:
    # for point in goal_local_grid.T:
        # print(f'grid_point: {point}')
        clothoid = Clothoid.G1Hermite(fake_position[0], fake_position[1], fake_pose_theta, point[0], point[1], point[2])
        # clothoid = Clothoid.G1Hermite(0.0, 0.0, fake_pose_theta, point[0], point[1], point[2])
        traj = sample_traj(clothoid, 20, v=point[3])
        all_traj.append(traj)
        all_traj_clothoid.append(np.array(clothoid.Parameters))



    fig, ax = plt.subplots()
    ax.plot(wp_x, wp_y, '-bo', markersize=1.0, label='optimal raceline')
    ax.plot(fake_position[0], fake_position[1], 'ro', markersize=5.0, label='car_position')
    draw_pts(xy_grid.T, ax, c='r', mksize=2.0, label='lattice points')
    # draw_pts(xy_grid_local[:2], ax, c='g', mksize=2.0, label='lattice points in local(just for show)')
    for i in range(len(all_traj)):
        draw_pts(np.array(all_traj[i]).T[:2], ax, mksize=0.5)
    plt.axis('equal')
    plt.legend()
    plt.show()


def get_length_cost(traj, traj_clothoid, cur_pose):
    # not division by zero, grid lookup only returns s >= 0.
    return traj_clothoid[-1]



draw_lattice_grid()