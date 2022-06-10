from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner
import yaml
from argparse import Namespace
import gym
import numpy as np
import time
from f1tenth_planning.planning.purepursuit_planner.purepursuit_planner import PurePursuitPlanner
from pyglet.gl import GL_POINTS

################# Create Environment ######################
with open('config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
print(conf)
work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
planner = LatticePlanner(conf, 0.17145 + 0.15875)
draw_grid_pts = []
draw_traj_pts = []
draw_target = []
################# Render Callback ######################
def render_callback(env_renderer):
    # custom extra drawing function
    global planner
    global draw_grid_pts
    global draw_traj_pts
    global draw_target

    e = env_renderer

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
################# Render Callback ######################

env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
env.add_render_callback(render_callback)

obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
env.render()
################# Create Environment ######################


################# Main Loop ######################
# planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)
laptime = 0.0
start = time.time()
while not done:
    steer, speed, best_traj = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])
    # speed, steer = planner.plan(obs['poses_x']nearest_point[0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
    #                             work['vgain'])
    # print(f'cur speed {speed}, cur steer {steer}')
    obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
    laptime += step_reward
    env.render(mode='human_fast')

print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
################# Main Loop ######################