import gym
import numpy as np

from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner
from f1tenth_planning.planning.lattice_planner.lattice_planner import sample_lookahead_square, get_length_cost

from pyglet.gl import GL_POINTS

def main():
    global planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target
    """
    Pure Pursuit example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # loading waypoints
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=0)
    planner = LatticePlanner(waypoints=waypoints)
    planner.add_sample_function(sample_lookahead_square)
    planner.add_cost_function(get_length_cost)

    # rendering
    draw_grid_pts = []
    draw_traj_pts = []
    draw_target = []

    # create environment
    env = gym.make('f110_gym:f110-v0', map='./Spielberg_map', map_ext='.png', num_agents=1)
    env.add_render_callback(render_callback)
    obs, _, done, _ = env.reset(np.array([[0.0, -0.14, 3.40]]))
    env.render()

    laptime = 0.0
    while not done:
        steer, speed, best_traj = planner.plan(obs['poses_x'][0],
                                               obs['poses_y'][0],
                                               obs['poses_theta'][0],
                                               1.7)
        obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
        laptime += timestep
        env.render(mode='human')
    print('Sim elapsed time:', laptime)

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