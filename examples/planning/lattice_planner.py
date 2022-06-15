import gym
import numpy as np
import random

from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner
from f1tenth_planning.planning.lattice_planner.lattice_planner import sample_lookahead_square, get_length_cost
from f1tenth_planning.planning.gap_follower.gap_follower import Gap_follower

from pyglet.gl import GL_POINTS


def main():
    """
    Lattice Planner example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """
    global planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target
    global draw_waypoints
    global waypoints_xytheta

    # loading waypoints
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=1)
    waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))
    planner = LatticePlanner(waypoints=waypoints)
    planner.add_sample_function(sample_lookahead_square)
    planner.add_cost_function(get_length_cost)

    # planners for opponents
    gap_follower = Gap_follower()

    # rendering
    draw_grid_pts = []
    draw_traj_pts = []
    draw_target = []
    draw_waypoints = []

    # create environment
    num_agents = 1
    env = gym.make('f110_gym:f110-v0', map='./Spielberg_map', map_ext='.png', num_agents=num_agents)
    env.add_render_callback(render_callback)
    obs, _, done, _ = env.reset(random_position(num_agents))
    env.render()

    laptime = 0.0
    while not done:
        # ego
        steer, speed, best_traj = planner.plan(obs['poses_x'][0],
                                               obs['poses_y'][0],
                                               obs['poses_theta'][0],
                                               1.7)
        action = np.array([[steer, speed/2]])
        # oppo
        for i in range(1, num_agents):
            scan = obs['scans'][i]
            steer, speed = gap_follower.plan(scan)
            action = np.vstack((action, np.array([[steer, speed]])))
        obs, timestep, done, _ = env.step(action)
        laptime += timestep
        env.render(mode='human_fast')
    print('Sim elapsed time:', laptime)


def random_position(sampled_number=1):
    global waypoints_xytheta
    ego_idx = random.sample(range(len(waypoints_xytheta)), 1)[0]
    for i in range(sampled_number):
        starting_idx = ego_idx + i*2
        x, y, theta = waypoints_xytheta[starting_idx][0],  waypoints_xytheta[starting_idx][1], waypoints_xytheta[starting_idx][2]
        if i==0:
            res = np.array([[x, y, theta]])  # (1, 3)
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res


def render_callback(e):
    """
    Custom render call back function for Lattice Planner Example

    Args:
        e: environment renderer
    """

    global planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target
    global draw_waypoints
    global waypoints_xytheta

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 400
    e.right = right + 400
    e.top = top + 400
    e.bottom = bottom - 400

    scaled_points = 50. * waypoints_xytheta[:, :2]

    # for i in range(waypoints_xytheta.shape[0]):
    #     if len(draw_waypoints) < waypoints_xytheta.shape[0]:
    #         b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
    #                         ('c3B/stream', [183, 193, 222]))
    #         draw_waypoints.append(b)
    #     else:
    #         draw_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    if planner.goal_grid is not None:
        goal_grid_pts = np.vstack([planner.goal_grid[:, 0], planner.goal_grid[:, 1]]).T
        scaled_grid_pts = 50. * goal_grid_pts
        for i in range(scaled_grid_pts.shape[0]):
            if len(draw_grid_pts) < scaled_grid_pts.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                draw_grid_pts.append(b)
            else:
                draw_grid_pts[i].vertices = [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.]

        best_traj_pts = np.vstack([planner.best_traj[:, 0], planner.best_traj[:, 1]]).T
        scaled_btraj_pts = 50. * best_traj_pts
        for i in range(scaled_btraj_pts.shape[0]):
            if len(draw_traj_pts) < scaled_btraj_pts.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                draw_traj_pts.append(b)
            else:
                draw_traj_pts[i].vertices = [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.]

if __name__ == '__main__':
    main()