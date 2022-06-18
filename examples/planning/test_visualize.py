import os
from f1tenth_planning.planning.lattice_planner.lattice_planner import *
from f1tenth_planning.planning.gap_follower.gap_follower import Gap_follower
from f1tenth_planning.utils.visualize import draw_lattice_grid
import random


module = os.path.dirname(os.path.abspath(__file__))

"""
waypoints: [x, y, v, heading, kappa]
grid: [x, y, heading, v], (n, 4)
"""
waypoints_xytheta = None
waypoints = np.array([0, 0])
width = 0.31
length = 0.58


def safe_plus(mod, start, delta):
    return (start + delta + mod) % mod


def random_position(sampled_number=1, car_gap_idx=15):
    # TODO: the 25 term just for test, make sure waypoints are not at the beginning or end of the track
    global waypoints_xytheta
    global waypoints
    ego_idx = random.sample(range(20, len(waypoints_xytheta) - 25), 1)[0]
    for i in range(sampled_number):
        starting_idx = (ego_idx + i * car_gap_idx) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1], \
                      waypoints_xytheta[starting_idx][2]
        if i == 0:
            res = np.array([[x, y, theta]])  # (1, 3)
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx



def main():
    global waypoints_xytheta
    global waypoints
    # loading waypoints
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=1)
    waypoints_xytheta = np.hstack((waypoints[:, :2], waypoints[:, 3].reshape(-1, 1)))

    # init Lattice Planner
    planner = LatticePlanner(waypoints=waypoints, map_path='./Spielberg_map', map_ext='.png')
    planner.add_sample_function(sample_lookahead_square)
    planner.add_cost_function(get_map_collision)
    planner.add_cost_function(get_length_cost)
    planner.add_cost_function(get_obstacle_collision)
    planner.add_cost_function(get_mean_curvature)

    # fake position
    num_agents = 2
    fake_pos, pos_idx = random_position(num_agents)
    draw_lattice_grid(fake_pos, pos_idx, planner, waypoints)


main()
