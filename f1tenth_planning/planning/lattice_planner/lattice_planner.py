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
Last Modified: 6/13/22
"""

from f1tenth_planning.utils.utils import *
# from f1tenth_planning.utils.utils import intersect_point
# from f1tenth_planning.utils.utils import get_rotation_matrix
# from f1tenth_planning.utils.utils import sample_traj
# from f1tenth_planning.utils.utils import zero_2_2pi
# from f1tenth_planning.utils.utils import pi_2_pi
# from f1tenth_planning.utils.utils import map_collision
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner

from pyclothoids import Clothoid
import numpy as np
from numba import njit

class LatticePlanner():
    """

    """
    def __init__(self, wheelbase=0.33, waypoints=None, **kwargs):
        """

        """
        self.wheelbase = wheelbase
        self.waypoints = waypoints

        self.sample_func = None
        self.cost_funcs = []
        self.selection_func = None

        self.best_traj = None
        self.prev_traj = np.array([False])
        self.goal_grid = None

        self.tracker = PurePursuitPlanner()

        try:
            self.map_path = kwargs['map_path']
            try:
                self.map_ext = kwargs['map_ext']
            except:
                raise ValueError('Map image extenstion must also be specified if using a map.')
            
            import os
            from PIL import Image
            import yaml
            from scipy.ndimage import distance_transform_edt as edt
            # load map image
            map_img_path = os.path.splitext(self.map_path)[0] + self.map_ext
            self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
            self.map_img[self.map_img <= 128.] = 0.
            self.map_img[self.map_img > 128.] = 255.
            self.map_height = self.map_img.shape[0]
            self.map_width = self.map_img.shape[1]
            # load map yaml
            with open(self.map_path + '.yaml', 'r') as yaml_stream:
                try:
                    map_metadata = yaml.safe_load(yaml_stream)
                    self.map_resolution = map_metadata['resolution']
                    self.origin = map_metadata['origin']
                except yaml.YAMLError as ex:
                    print(ex)

            self.orig_x = self.origin[0]
            self.orig_y = self.origin[1]
            self.orig_s = np.sin(self.origin[2])
            self.orig_c = np.cos(self.origin[2])

            self.dt = self.map_resolution * edt(self.map_img)
            self.map_metainfo = (self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution)

        except Exception as ex:
            print(ex)
            self.dt = None

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

    def eval(self, all_traj, all_traj_clothoid, opp_poses, cost_weights=None):
        """
        Evaluate a list of generated clothoids based on added cost functions
        
        Args:
            pose_x
            pose_y
            pose_theta
        Returns:
            costs
        """
        if cost_weights is None:
            cost_weights = np.array([1/len(self.cost_funcs)] * len(self.cost_funcs))
        if len(self.cost_funcs) == 0:
            raise NotImplementedError('Please set cost functions before evaluating.')
        if len(self.cost_funcs) != len(cost_weights):
            raise ValueError('Length of cost weights must be the same as number of cost functions.')
        if np.sum(cost_weights) != 1:
            raise ValueError('Cost weights must add up to 1.')

        cost = np.array([0.0] * len(all_traj_clothoid))
        # loop through all trajectories
        for i, func in enumerate(self.cost_funcs):
            cur_cost = func(all_traj, all_traj_clothoid, opp_poses, self.prev_traj, self.dt, self.map_metainfo)
            cost += cost_weights[i] * cur_cost
        return cost
        ### non-vectorized
        # for traj, traj_clothoid in zip(all_traj, all_traj_clothoid):
        #     cost = 0.
        #     # loop through all cost functions
        #     for i, func in enumerate(self.cost_funcs):
        #         if self.dt is None:
        #             cost += cost_weights[i] * func(traj, traj_clothoid, opp_poses)
        #         else:
        #             cost += cost_weights[i] * func(traj, traj_clothoid, opp_poses, self.dt, self.map_metainfo)
        #     all_costs.append(cost)
        # return all_costs


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

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None):
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
        if waypoints is None:
            waypoints = self.waypoints

        # sample a grid based on current states
        self.goal_grid = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints)

        # generate clothoids
        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, point[0], point[1], point[2])
            traj = sample_traj(clothoid, 20, point[3])
            all_traj.append(traj)
            # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
            all_traj_clothoid.append(np.array(clothoid.Parameters))

        # evaluate all trajectory on all costs
        all_costs = self.eval(np.array(all_traj), np.array(all_traj_clothoid), opp_poses)

        # select best trajectory
        best_traj_idx = self.select(all_costs)
        self.best_traj = all_traj[best_traj_idx]
        self.prev_traj = self.best_traj.copy()
        # track best trajectory
        steer, speed = self.tracker.plan(pose_x,
                                         pose_y,
                                         pose_theta,
                                         0.3,
                                         self.best_traj)

        return steer, speed, self.best_traj

"""

Example function for sampling a grid of goal points

"""

# @njit(cache=True)
def sample_lookahead_square(pose_x,
                            pose_y,
                            pose_theta,
                            velocity,
                            waypoints,
                            lookahead_distances=np.array([1.8, 2.1, 2.4, 2.7]),
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
        lh_pt, i2, t2 = intersect_point(np.ascontiguousarray(nearest_p), d, waypoints[:, 0:2], t + nearest_i, wrap=True)
        i2 = int(i2)
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

"""

Example functions for different costs

traj: np.ndarray, (n, m, 5), (x, y, v, heading, arc_length)
traj_clothoid: np.ndarray, (n, 6), (x0, y0, theta0, k0, dk, arc_length)
opp_poses: np.ndarray, (k, 3)

Return: 
cost: np.ndarray, (n, 1) 
"""
from sklearn.metrics import pairwise_distances_argmin

@njit(cache=True)
def get_length_cost(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    return traj_clothoid[:, -1]


@njit(cache=True)
def get_map_collision(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    if dt is None:
        raise ValueError('Map Distance Transform dt has to be set when using this cost function.')
    # points: (n, 2)
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)  # (nxm, 5)
    collisions = map_collision(all_traj_pts[:, 0:2], dt, map_metainfo)  # (nxm)
    collisions = collisions.reshape(len(traj), -1)  #(n, m)
    cost = []
    for traj_collision in collisions:
        if np.any(traj_collision):
            cost.append(100.)
        else:
            cost.append(0.)
    return np.array(cost)


def get_max_curvature(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1].reshape(-1, 1)  # (n, 1)
    k1 = k0 + dk * s  # (n, 1)
    cost = np.max(np.hstack((k0, k1)), axis=1)
    return cost


def get_mean_curvature(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    # print('d')
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts  # (n, m)
    cost = np.mean(traj_k, axis=1)
    return cost


def get_similarity_cost(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    scale = 10.0
    if not prev_traj.all():
        return np.zeros((len(traj)))
    prev_traj = prev_traj[:, :2]
    traj = traj[:, :, :2]
    diff = traj-prev_traj
    cost = diff * diff
    cost = np.sum(cost, axis=(1, 2)) * scale
    return cost


@njit(cache=True)
def get_obstacle_collision(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    width, length = 0.6, 0.3
    n, m, _ = traj.shape
    cost = np.zeros(n)
    traj_xyt = traj[:, :, :3]
    for i, tr in enumerate(traj_xyt):
        # print(i)
        close_p_idx = x2y_distances_argmin(np.ascontiguousarray(opp_poses[:, :2]), np.ascontiguousarray(tr[:, :2]))
            # pairwise_distances_argmin(np.ascontiguousarray(opp_poses[:, :2]), np.ascontiguousarray(tr[:, :2]))
            # (3, )
        for opp_pose, p_idx in zip(opp_poses, close_p_idx):
            opp_box = get_vertices(opp_pose, length, width)
            p_box = get_vertices(tr[int(p_idx)], length, width)
            if collision(opp_box, p_box):
                cost[i] = 100.
    return cost



