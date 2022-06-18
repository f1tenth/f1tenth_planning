from numba import njit
import numpy as np
from functools import wraps
from time import time
from f1tenth_planning.utils.utils import *
from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner
from sklearn.metrics import pairwise_distances_argmin

traj_clothoid = np.ones((28, 6))
traj = np.zeros((28, 20, 5)) #(n, m, 5)
traj[0, 2, 0] = 1
traj[0, 2, 1] = 1
prev_traj = np.ones((20, 5))
oppo_poses = np.array([
    [0, 0, 0],
    [1, 2, 0],
    [2, 2, 0]
])


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('func:%r args:[%r, %r] took: %2.8f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap

@njit(cache=True)
def get_length_cost(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    return traj_clothoid[:, -1]

# @njit(cache=True)
# @timing
def get_mean_curvature(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    # print('d')
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts # (n, m)
    cost = np.mean(traj_k, axis=1)
    return cost
    # print(cost.shape)
    # print(cost)

# @timing
# @njit(cache=True)
def get_max_curvature(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1].reshape(-1, 1)  # (n, 1)
    k1 = k0 + dk * s  # (n, 1)
    cost = np.max(np.hstack((k0, k1)), axis=1)
    return cost
    # print(cost)


# @njit(cache=True)
def get_similarity_cost(traj, traj_clothoid, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
    scale = 10.0
    if not prev_traj.all():
        return np.zeros((len(traj)))
    # prev_traj = np.repeat(prev_traj[np.newaxis, :, :2], len(traj), axis=0)  # (m, 5) to (n, m, 2)
    prev_traj = prev_traj[:, :2]
    traj = traj[:, :, :2]
    diff = traj-prev_traj
    cost = diff * diff
    cost = np.sum(cost, axis=(1, 2)) * scale
    return cost

@njit(cache=True)
def get_map_collision(traj, traj_clothoid=None, opp_poses=None, prev_traj=None, dt=None, map_metainfo=None):
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


@njit(cache=True)
def get_array():
    k = np.ones(12).reshape(6, 2)
    kk = np.ones(2)
    return np.argmin(kk)

# get_mean_curvature()
# get_max_curvature()
# k = get_array()
# print(k.shape)
# k = get_array()
# print(k)
# print(k.shape)
# print(cost)
# cost = get_obstacle_collision()
# cost2 = get_obstacle_collision()


def time_test(func, **kwargs):
    func(**kwargs)
    # func()
    start = time()
    i = 100
    for _ in range(i):
        cost = func(**kwargs)
    end = time()
    interval = end-start
    print(f'execute the function: {func.__name__} for {i} times need {interval} s, avarage{1000 * interval/i} ms')
    print(f'cost of fake data is \n{cost}')


if __name__ == '__main__':
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=1)
    planner = LatticePlanner(waypoints=waypoints, map_path='./Spielberg_map', map_ext='.png')

    time_test(get_map_collision, **{'traj': traj, 'dt': planner.dt, 'map_metainfo': planner.map_metainfo})
    time_test(get_obstacle_collision, **{'traj': traj, 'traj_clothoid': traj_clothoid, 'opp_poses': oppo_poses, 'prev_traj':prev_traj})
    time_test(get_mean_curvature, **{'traj': traj, 'traj_clothoid': traj_clothoid, 'opp_poses': oppo_poses, 'prev_traj':prev_traj})
    time_test(get_max_curvature, **{'traj': traj, 'traj_clothoid': traj_clothoid, 'opp_poses': oppo_poses, 'prev_traj':prev_traj})
    time_test(get_length_cost, **{'traj': traj, 'traj_clothoid': traj_clothoid, 'opp_poses': oppo_poses, 'prev_traj':prev_traj})
    time_test(get_similarity_cost, **{'traj': traj, 'traj_clothoid': traj_clothoid, 'opp_poses': oppo_poses, 'prev_traj': prev_traj})

# k = get_array()
# print(k)
# print(k.shape)