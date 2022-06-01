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
Lattice Planner

Author: Hongrui Zheng
Last Modified: 5/5/22
"""

from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_rotation_matrix
from f1tenth_planning.utils.utils import sample_traj
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner

from pyclothoids import Clothoid
import numpy as np
from numba import njit

class LatticePlanner():
    """

    """
    def __init__(self, wheelbase=0.33, waypoints=None):
        """

        """
        self.wheelbase = wheelbase
        self.waypoints = waypoints

        self.sample_func = None
        self.cost_funcs = []
        self.selection_func = None

        self.tracker = PurePursuitPlanner()

    def add_cost_function(self, func):
        """
        Add cost function to list for eval.

        Specification of a candidate cost function:
            Args:
                pose_x
                pose_y
                pose_theta
                velocity
                kwargs (TODO?)

            Returns:
                costs
        """
        if type(func) is list:
            self.cost_funcs.extend(func)
        else:
            self.cost_funcs.append(func)

    def add_sample_function(self, func):
        """
        Add a custom sample function to create goal grid
        
        Specification of a candidate sample function:
            Args:
                pose_x (float):
                pose_y (float):
                pose_theta (float):
                velocity (float):
                waypoints (numpy.ndarray [N, 5]):

            Returns:
                goal_grid (numpy.ndarray [N, 3]):

        Args:
            func (function): function that takes in the current observation and provide sampled goal states

        Returns:
            None
        """
        self.sample_func = func

    def add_selection_function(self, func):
        """
        Add a custom selection fucntion to select a trajectory. The selection function returns the index of the 'best' cost.

        Specification of a candidate selection function:
            Args:
                costs ():

            Returns:
                index ():
        """
        self.selection_func = func

    def sample(self, pose_x, pose_y, pose_theta, velocity, waypoints):
        """
        Sample a goal grid based on sample function. Given the current vehicle state, return a list of [x, y, theta] tuples
        
        Args:
            None

        Returns:
            goal_grid (numpy.ndarray [N, 3]): list of goal states, columns are [x, y, theta]
        """
        if self.sample_func is None:
            raise NotImplementedError('Please set a sample function before sampling.')

        goal_grid = self.sample_func(pose_x, pose_y, pose_theta, velocity, waypoints)

        return goal_grid

    def eval(self, all_traj, cost_weights):
        """
        Evaluate a list of generated clothoids based on added cost functions
        
        Args:
            pose_x
            pose_y
            pose_theta
        Returns:
            costs
        """
        if len(self.cost_funcs) == 0:
            raise NotImplementedError('Please set cost functions before evaluating.')
        if len(self.cost_funcs) != len(cost_weights):
            raise ValueError('Length of cost weights must be the same as number of cost functions.')
        if np.sum(cost_weights) != 1:
            raise ValueError('Cost weights must add up to 1.')

        all_costs = []
        # loop through all trajectories
        for traj in all_traj:
            cost = 0.
            # loop through all cost functions
            for i, func in enumerate(self.cost_funcs):
                cost += cost_weights[i] * func(traj)
            all_costs.append(cost)
        return all_costs


    def select(self, all_costs):
        """
        Select the best trajectory based on the selection function, defaults to argmin if no custom function is defined.
        
        Args:
            all_costs ():

        Returns:
            idx ():
        """
        if self.selection_func is None:
            self.selection_func = np.argmin
        best_idx = self.selection_func(all_costs)
        return best_idx

    def plan(self, pose_x, pose_y, pose_theta, velocity, waypoints=None):
        """
        Plan for next step

        Args:
            pose_x (float):
            pose_y (float):
            pose_theta (float):
            velocity (float):
            waypoints (numpy.ndarray [N, 5], optional, default=None):

        Returns:
            steering_angle (float):
            speed (float):
            selected_traj (numpy.ndarray [M, ])
        """
        # sample a grid based on current states
        goal_grid = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints)

        # generate clothoids
        all_traj = []
        for point in goal_grid:
            clothoid = Clothoid.G1Hermite(0., 0., 0., point[0], point[1], point[2])
            traj = sample_traj(clothoid, 100)
            all_traj.append(traj)

        # evaluate all trajectory on all costs
        all_costs = self.eval(np.array(all_traj))

        # select best trajectory
        best_traj_idx = self.select(all_costs)
        best_traj = all_traj[best_traj_idx]

        # track best trajectory
        steer, speed = self.tracker.plan(pose_x,
                                         pose_y,
                                         pose_theta,
                                         0.8,
                                         best_traj)

        return steer, speed, best_traj

"""

Example function for sampling a grid of goal points

"""


def sample_lookahead_square(pose_x,
                            pose_y,
                            pose_theta,
                            velocity,
                            waypoints,
                            lookahead_distances=[0.4, 0.6, 0.8, 1.0],
                            widths=np.linspace(-1.0, 1.0, num=7)):
    """
    Example function to sample goal points. In this example it samples a rectangular grid around a look-ahead point.

    Args:
        pose_x ():
        pose_y ():
        pose_theta ():
        velocity ():
        waypoints ():
        lookahead_distances ():
        widths ():
    
    Returns:
        grid (): Returned grid of goal points
    """
    # get lookahead points to create grid along waypoints
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, i = nearest_point(position, waypoints[:, 0:2])
    lh_centers = []
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(position, d, waypoints[:, 0:2], i + t, wrap=True)
        lh_centers[i] = waypoints[i2, [0, 1, 3]]
    lh_centers = np.array(lh_centers)
    grid = np.repeat(lh_centers, len(widths), axis=0)
    widths_rep = np.repeat(widths[:, None], lh_centers.shape[0], axis=0)
    # deviate points from center
    grid[:, 1] += widths_rep[:, 0]
    # rotate grid
    rot = get_rotation_matrix(pose_theta)
    rotated_grid = np.dot(rot, grid)
    return rotated_grid

"""

Example functions for different costs

"""

@njit(cache=True)
def get_length_cost(param_list):
    # not division by zero, grid lookup only returns s >= 0.
    return 1./param_list[:, 0]

@njit(cache=True)
def get_max_curvature(traj_list, num_traj):
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        out[i] = np.max(np.abs(traj_list[i*trajectory_generator.NUM_STEPS:(i+1)*trajectory_generator.NUM_STEPS, 3]))
    return out

@njit(cache=True)
def get_mean_curvature(traj_list, num_traj):
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        out[i] = np.mean(np.abs(traj_list[i*trajectory_generator.NUM_STEPS:(i+1)*trajectory_generator.NUM_STEPS, 3]))
    return out

@njit(cache=True)
def get_similarity_cost(traj_list, prev_path, num_traj):
    N = trajectory_generator.NUM_STEPS
    prev_shifted = prev_path[N_SHIFT:-N_CULL, 2]
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        traj = traj_list[i*N:(i+1)*N, 2]
        traj_shifted = traj[:- N_SHIFT - N_CULL]
        out[i] = np.sum(np.square((traj_shifted - prev_shifted)))
    return out