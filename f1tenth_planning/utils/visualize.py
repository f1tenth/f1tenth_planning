import matplotlib.pyplot as plt
from f1tenth_planning.planning.lattice_planner.lattice_planner import *

from f1tenth_planning.utils.utils import \
    intersect_point, nearest_point, get_rotation_matrix, rotate_along_point, sample_traj, zero_2_2pi
from pyclothoids import Clothoid
import matplotlib
import matplotlib.cm as cm


def draw_pts(pts, ax, c='b', mksize=5.0, label=None, pointonly=False, linewidth=0.5):
    """
    pts: (2, n)
    """
    x = pts[0]
    y = pts[1]
    if pointonly:
        ax.plot(x, y, 'o', c=c, markersize=mksize, label=label)
    else:
        ax.plot(x, y, c=c, markersize=mksize, label=label, linewidth=linewidth)


def value2color(value):
    """
    value: np.ndarray (n, )
    """

    minima = np.min(value)
    maxima = np.max(value)
    rgba = []

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)

    for v in value:
        rgba.append(mapper.to_rgba(v))

    return rgba, mapper


def draw_lattice_grid(fake_poses, pos_idx, planner, waypoints, width=0.31, length=0.58):
    lh_pt_nums = 4

    wp_x = waypoints[:, 0]
    wp_y = waypoints[:, 1]
    wp_num = waypoints.shape[0]

    wp_x = wp_x[pos_idx - 25:pos_idx + 25]
    wp_y = wp_y[pos_idx - 25:pos_idx + 25]
    waypoints = waypoints[pos_idx - 50:pos_idx + 50, :]

    ego_pose = fake_poses[0]
    goal_grid = planner.sample(ego_pose[0], ego_pose[1], ego_pose[2], 2.0, waypoints)
    xy_grid = goal_grid[:, :2]
    all_traj = []
    all_traj_clothoid = []
    for point in goal_grid:
        clothoid = Clothoid.G1Hermite(ego_pose[0], ego_pose[1], ego_pose[2], point[0], point[1], point[2])
        traj = sample_traj(clothoid, 20, point[3])
        all_traj.append(traj)
        # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
        all_traj_clothoid.append(np.array(clothoid.Parameters))
    all_costs = planner.eval(np.array(all_traj), np.array(all_traj_clothoid), fake_poses[1:, :].reshape(-1, 3))
    draw_cost_colors, draw_cmapper = value2color(all_costs)
    # select best trajectory
    best_traj_idx = planner.select(all_costs)
    best_traj = all_traj[best_traj_idx]
    print(all_costs)
    print(best_traj_idx)

    ### processing map, get track coordinate
    map_img = planner.map_img
    map_r = planner.map_resolution
    map_ori_x = planner.map_metainfo[0]
    map_ori_y = planner.map_metainfo[1]
    track_coor = np.nonzero(map_img == 0)
    # axis transformation
    track_y = track_coor[0] * map_r + map_ori_y
    track_x = track_coor[1] * map_r + map_ori_x
    track_xy = np.vstack((track_x, track_y))  # (2, n)
    # filter track near the sampled fake car position
    track_idx = np.nonzero(np.linalg.norm(track_xy.T - ego_pose[:2], axis=1) < 8.0)[0]
    track_x = track_x[track_idx]
    track_y = track_y[track_idx]

    ### track, waypoints, car
    fig, (ax, ax1) = plt.subplots(1, 2,figsize=(20, 10))
    # fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(track_x, track_y, 'o', c='k', markersize=0.3, label='track')
    ax.plot(wp_x, wp_y, c='b', markersize=1.0, label='optimal raceline')
    ax.plot(ego_pose[0], ego_pose[1], 'ro', markersize=5.0, label='car_position')

    ### fake opponents
    for i in range(1, len(fake_poses)):
        ax.plot(fake_poses[i][0], fake_poses[i][1], 'go', markersize=5.0, label='opponent_position')
        oppo_vertices = get_vertices(fake_poses[i], length, width).T
        draw_pts(oppo_vertices, ax, c='g', mksize=5.0, label='lattice points', pointonly=True)

    draw_pts(xy_grid.T, ax, c='r', mksize=2.0, label='lattice points', pointonly=True)

    ### trajs
    n = len(all_traj)
    for i in range(n):
        draw_pts(np.array(all_traj[i]).T[:2], ax, c=draw_cost_colors[i], linewidth=1.0)
        if i % 2 == 0:
            ax.text(xy_grid[i][0], xy_grid[i][1], i)
        if i == best_traj_idx:
            ax.text(xy_grid[i][0], xy_grid[i][1], i, c='r')
    draw_pts(np.array(best_traj).T[:2], ax, c='k', linewidth=1.0)

    ### cost
    bars = ax1.bar(np.arange(0, len(all_costs), 1), all_costs, 0.5, label='cost of trajs')
    ax1.bar_label(bars, fmt='%.1f')
    ax1.set_xlim(0)

    ax.axis('equal')
    fig.colorbar(draw_cmapper, orientation='horizontal', label='Cost')
    plt.axis('equal')
    plt.legend()
    plt.show()