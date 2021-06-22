# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz, Aman Sinha, Matthew O'Kelly

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
Utility functions and look up table generation for the State Lattice Planner

Author: Hongrui Zheng, Aman Sinha
Last Modified: 5/27/21
"""

import numpy as np
import numba
from numba import prange
from numba.typed import List
import math
import scipy.optimize as spo
import argparse

# max/min curvature (rad)
# KMAX = 30000000.0
# KMIN = -30000000.0
KMAX = 30.0
KMIN = -30.0
# max/min curvature rate (rad/sec)
# DKMAX = 300000000.0
# DKMIN = -300000000.0
DKMAX = 30.0
DKMIN = -30.0
# max/min accel/decel (m/s^2)
DVMAX = 2.000
DVMIN = -6.000
# speed control logic a coeff
ASCL = 0.1681
# ASCL = 16.81
# speed control logic b coeff
BSCL = -0.0049
# speed control logic threshold (m/s)
VSCL = 4.0
# speed control logic safety factor
SF = 1.000

DT = 0.0001

NUM_STEPS = 100

ORDER = 3
if ORDER == 3:
    PARAM_MAT = np.array(
        [[    1.,     0.,     0.,    0.],
         [-11./2,     9.,  -9./2,    1.],
         [    9., -45./2,    18., -9./2],
         [ -9./2,  27./2, -27./2,  9./2]])
elif ORDER == 5:
    PARAM_MAT = np.array(
        [[      1.,      0.,      0.,    0.,     0.,     0.],
         [      0.,      0.,      0.,    0.,     1.,     0.],
         [      0.,      0.,      0.,    0.,     0.,   1./2],
         [ -575./8,     81.,  -81./8,    1., -85./4, -11./4],
         [  333./2, -405./2,   81./2, -9./2,    45.,   9./2],
         [ -765./8,  243./2, -243./8,  9./2, -99./4, -9./4]])
else:
    assert(1==0)


@numba.njit(cache=True)
def params_to_coefs(params):
    s = params[-1]
    s2 = s**2
    s3 = s**3
    if ORDER == 3:
        coefs = np.dot(PARAM_MAT, params[:-1])
        coefs[1] /= s
        coefs[2] /= s2
        coefs[3] /= s3
        return coefs
    if ORDER == 5:
        temp = np.concatenate((params[:4], np.array([params[4]*s, params[5]*s2])))
        coefs = np.dot(PARAM_MAT, temp)
        coefs[0] = params[0]
        coefs[1] = params[4]
        coefs[2] = params[5]/2.
        s4 = s**4
        s5 = s**5
        coefs[3] /= s3
        coefs[4] /= s4
        coefs[5] /= s5
        return coefs
    assert(1==0)

#state is sx, sy, theta, kappa, v
#params is p0, p1, p2, p3, s
@numba.njit(cache=True)
def init_x(start_state, goal_state):
    sx_f = goal_state[0]
    sy_f = goal_state[1]
    theta_f = goal_state[2]
    kappa_0 = start_state[3]
    kappa_f = goal_state[3]

    d = math.sqrt(sx_f**2 + sy_f**2)
    d_theta = abs(theta_f)
    s = d*(d_theta**2/5.0 + 1.0) + (2.0/5.0)*d_theta + 0.0001
    c = 0.
    a = (6.0*theta_f/(s**2)) - (2*kappa_0/s) + (4*kappa_f/s)
    b = (3.0/(s**2)) * (kappa_0+kappa_f) + (6.0*theta_f/(s**3))

    p0 = kappa_0
    p1 = p0 + a*s/3. + b*s**2/9.
    p2 = p0 + a*(2*s/3.) + b*(4.*s**2)/9.
    p3 = kappa_f
    return np.array([p1, p2, s])

#state is sx, sy, theta, kappa, v
#params is p0, p1, p2, p3, s
@numba.njit(cache=True)
def speed_control_logic(state):
    """
    Function to compute safe/feasible speed and curvature
    """
    vcmd = abs(state[4])
    kappa = state[3]

    # compute safe speed
    compare_v = (kappa-ASCL)/BSCL
    vcmd_max = max(VSCL, compare_v)
    # compute safe curvature
    compare_kappa = ASCL + (BSCL*vcmd)
    kmax_scl = min(KMAX, compare_kappa)
    # check if max curvatre for speed is exceeded
    if kappa >= kmax_scl:
        vcmd = SF * vcmd_max
    # update velocity command
    state[4] = vcmd
    return state

#state is sx, sy, theta, kappa, v
#params is p0, p1, p2, p3, s
@numba.njit(cache=True)
def response_to_control_inputs(state, state_next, dt):
    """ Function for computing the vehicles response to control inputs
    """
    # call speed control logic for safe speed
    state_next = speed_control_logic(state_next)

    kappa = state[3]
    kappa_next = state_next[3]
    v = state[4]
    v_next = state_next[4]

    kdot = (kappa_next - kappa)/dt
    kdot = min(kdot, DKMAX)
    kdot = max(kdot, DKMIN)
    kappa_next = kappa + kdot*dt
    kappa_next = min(kappa_next, KMAX)
    kappa_next = max(kappa_next, KMIN)
    state_next[3] = kappa_next

    vdot = (v_next - v)/dt
    vdot = min(vdot, DVMAX)
    vdot = max(vdot, DVMIN)
    state_next[4] = v + vdot*dt
    return state_next

@numba.njit(cache=True)
def get_curvature_command(params, s_cur):
    coefs = params_to_coefs(params)
    out = 0.
    for i in range(coefs.shape[0]):
        out += coefs[i]*s_cur**i
    return out

@numba.njit(cache=True)
def get_curvature_theta(coefs, s_cur):
    out = 0.
    out2 = 0.
    for i in range(coefs.shape[0]):
        temp = coefs[i]*s_cur**i
        out += temp
        out2 += temp*s_cur/(i+1)
    return out, out2

@numba.njit(cache=True)
def get_velocity_command(v_goal, v, dt):
    # TODO: change velocity profile to actual profile from sys id
    accel = 50.0
    decel = -80.0
    if v < v_goal:
        return v + accel*dt
    if v > v_goal:
        return v + decel*dt
    return v

#state is sx, sy, theta, kappa, v
#params is p0, p1, p2, p3, s
@numba.njit(cache=True)
def motion_model(state, goal_state, params, dt):
    # get motion model predictive horizon, assuming constant accel/decel
    horizon = 0
    if goal_state[4] == 0 and state[4] == 0:
        # triangular velocity profile, use speed limit
        horizon = (2.0*params[-1])/VSCL
    else:
        # trapezoidal velocity profile
        horizon = (2.0*params[-1])/(state[4]+goal_state[4])

    v_goal = goal_state[4]

    cur_s = 0
    cur_state = np.copy(state)
    next_state = np.empty(5,)
    t = 0
    while t < horizon:
        sx = cur_state[0]
        sy = cur_state[1]
        theta = cur_state[2]
        kappa = cur_state[3]
        v = state[4]
        next_state[0] = sx + v*np.cos(theta)*dt
        next_state[1] = sy + v*np.sin(theta)*dt
        next_state[2] = theta + v*kappa*dt
        next_state[3] = get_curvature_command(params, cur_s)
        next_state[4] = get_velocity_command(v_goal, v, dt)
        # potential changes to next kappa and next v
        next_state = response_to_control_inputs(cur_state, next_state, dt)
        cur_s += dt*v
        t += dt
        cur_state = next_state
    return cur_state

@numba.njit(cache=True)
def integrate_path(params):
    N = NUM_STEPS
    coefs = params_to_coefs(params)
    states = np.empty((N,4))
    states[0] = np.zeros(4,)
    states[0,3] = coefs[0]
    dx = 0
    dy = 0
    x = 0
    y = 0
    ds = params[-1]/N
    theta_old = 0
    for k in range(1,N):
        sk = k*ds
        kappa_k, theta_k = get_curvature_theta(coefs, sk)
        dx = dx*(1-1/k) + (np.cos(theta_k)+np.cos(theta_old))/2/k
        dy = dy*(1-1/k) + (np.sin(theta_k)+np.sin(theta_old))/2/k
        x = sk*dx
        y = sk*dy
        states[k] = [x,y,theta_k, kappa_k]
        theta_old = theta_k
    return states

@numba.njit(cache=True)
def integrate_all(curvature_list):
    points_list = np.empty((curvature_list.shape[0]*NUM_STEPS, 4))
    for i in range(curvature_list.shape[0]):
        # this returns a trajectory as Nx4 array
        # roll it so s is at the end
        points = integrate_path(np.roll(curvature_list[i], -1))
        points_list[i*NUM_STEPS:i*NUM_STEPS+NUM_STEPS, :] = points
    return points_list

@numba.njit(cache=True)
def integrate_parallel(curvature_list_all):
    out = List()
    for j in range(len(curvature_list_all)):
        curvature_list = curvature_list_all[j]
        points_list = np.empty((curvature_list.shape[0]*NUM_STEPS, 4))
        for i in range(curvature_list.shape[0]):
            # this returns a trajectory as Nx4 array
            # roll it so s is at the end
            points = integrate_path(np.roll(curvature_list[i], -1))
            points_list[i*NUM_STEPS:i*NUM_STEPS+NUM_STEPS, :] = points
        if curvature_list.shape[0] == 0:
            # if num_traj for that guy is 0, append a single -1 as 2d array
            # this should be still fine in normal plan because normal plan returns before this happens
            out.append(-1.*np.ones((1, 1)))
        else:
            out.append(points_list)
    return out


@numba.njit(cache=True)
def integrate_path_motion_model(state, goal_state, params, dt):
    # get motion model predictive horizon, assuming constant accel/decel
    horizon = 0
    if goal_state[4] == 0 and state[4] == 0:
        # triangular velocity profile, use speed limit
        horizon = (2.0*params[-1])/VSCL
    else:
        # trapezoidal velocity profile
        horizon = (2.0*params[-1])/(state[4]+goal_state[4])

    v_goal = goal_state[4]

    cur_s = 0
    cur_state = np.copy(state)
    next_state = np.empty(5,)
    t = 0
    states = np.empty((int(horizon/dt)+1,5))
    counter = 0
    states[counter] = cur_state
    while t < horizon:
        sx = cur_state[0]
        sy = cur_state[1]
        theta = cur_state[2]
        kappa = cur_state[3]
        v = state[4]
        next_state[0] = sx + v*np.cos(theta)*dt
        next_state[1] = sy + v*np.sin(theta)*dt
        next_state[2] = theta + v*kappa*dt
        next_state[3] = get_curvature_command(params, cur_s)
        next_state[4] = get_velocity_command(v_goal, v, dt)
        # potential changes to next kappa and next v
        next_state = response_to_control_inputs(cur_state, next_state, dt)
        cur_s += dt*v
        t += dt
        cur_state = next_state
        counter+=1
        states[counter] = cur_state
    return states

@numba.njit(cache=True)
def trajectory5(x, p0, p3, sv, gx, gy, gt, gv, p4, p5):
    p = np.empty(7,)
    p[0] = p0
    p[1] = x[0]
    p[2] = x[1]
    p[3] = p3
    p[4] = p4
    p[5] = p5
    p[6] = x[2]
    start_state = np.array([0., 0., 0., p0, sv])
    goal_state = np.array([gx, gy, gt, p3, gv])
    next_state = motion_model(start_state, goal_state, p, DT)
    return (next_state[0]-gx)**2 + (next_state[1]-gy)**2+2*(next_state[2]-gt)**2 + 0.00001*x[2]**2

@numba.njit(cache=True)
def trajectory3(x, p0, p3, sv, gx, gy, gt, gv):
    p = np.empty(5,)
    p[0] = p0
    p[1] = x[0]
    p[2] = x[1]
    p[3] = p3
    p[4] = x[2]
    start_state = np.array([0., 0., 0., p0, sv])
    goal_state = np.array([gx, gy, gt, p3, gv])
    next_state = motion_model(start_state, goal_state, p, DT)
    return (next_state[0]-gx)**2 + (next_state[1]-gy)**2+2*(next_state[2]-gt)**2 + 0.00001*x[2]**2

# TODO interpolate between the nearest 2? could be litttt
@numba.njit(cache=True)
def lookup(x, y, theta, kappa0, lut_x, lut_y, lut_theta, lut_kappa, lut, step_sizes):
    x_idx = idx_tweaker(x, lut_x, step_sizes[0])
    y_idx = idx_tweaker(y, lut_y, step_sizes[1])
    theta_idx = idx_tweaker(theta, lut_theta, step_sizes[2])
    kappa_idx = idx_tweaker(kappa0, lut_kappa, step_sizes[3])
    # params should be stored as [s, k0, k1, k2, k3]
    return lut[x_idx, y_idx, theta_idx, kappa_idx]

@numba.njit(cache=True)
def idx_tweaker(val, lut_keys, lut_step_size):
    idx = np.searchsorted(lut_keys, val, side='right')
    temp = (val-lut_keys[idx-1])/lut_step_size
    if temp >= 0 and temp <= 0.5:
        idx -= 1
    return min(lut_keys.shape[0]-1, idx)


def build_lut(x_bounds=[-1, 10], y_bounds=[-8,8], theta_bounds=[-np.pi/2,np.pi/2], kappa_bounds=[-1,1],
             nx=111, ny=161, nt=33, nk=11):
    delta_x = np.linspace(x_bounds[0], x_bounds[1], nx)
    delta_y = np.linspace(y_bounds[0], y_bounds[1], ny)
    delta_theta = np.linspace(theta_bounds[0], theta_bounds[1], nt)
    kappa0 = np.linspace(kappa_bounds[0], kappa_bounds[1], nk)
    from joblib import Parallel, delayed
    options = {'maxiter': 20, 'maxfev': 1000}
    def get_params(x,y,t,k):
        p0 = k
        p3 = 0.0
        sv = 5.
        gx = x
        gy = y
        gt = t
        gv = 5.
        start_state = np.array([0., 0., 0., p0, sv])
        goal_state = np.array([gx, gy, gt, p3, gv])
        x0 = init_x(start_state, goal_state)
        params = np.array([p0, x0[0], x0[1], p3, x0[2]])
        out = spo.minimize(lambda x: trajectory3(x, p0, p3, sv, gx, gy, gt, gv), x0, method='Powell', options=options)
        #print('func', out.fun, out.x)
        if (out.fun > 1.) or (out.x[2] < 0) or (out.x[2] > 4*np.linalg.norm([gx, gy])):
            return -1*np.ones(5,)
        params[1] = out.x[0]
        params[2] = out.x[1]
        params[-1] = out.x[2]
        return np.roll(params, 1)

    Dx, Dy, Dt, Dk = np.meshgrid(delta_x, delta_y, delta_theta, kappa0, indexing='ij')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(get_params)(Dx[np.unravel_index(i, Dx.shape)],
                                                                 Dy[np.unravel_index(i, Dx.shape)],
                                                                 Dt[np.unravel_index(i, Dx.shape)],
                                                                 Dk[np.unravel_index(i, Dx.shape)])
                                                                 for i in range(np.prod(Dx.shape)))
    results = np.array(results)
    results = results.reshape((*Dx.shape, results.shape[1]))
    np.savez_compressed('results.npz', result=results)


def build_lut_split(guy, num_guys=11, x_bounds=[-1, 10], y_bounds=[-8,8], theta_bounds=[-np.pi/2,np.pi/2], kappa_bounds=[-1,1],
             nx=111, ny=161, nt=33, nk=11):
    delta_x = np.linspace(x_bounds[0], x_bounds[1], nx)
    delta_y = np.linspace(y_bounds[0], y_bounds[1], ny)
    delta_theta = np.linspace(theta_bounds[0], theta_bounds[1], nt)
    kappa0 = np.linspace(kappa_bounds[0], kappa_bounds[1], nk)
    from joblib import Parallel, delayed
    options = {'maxiter': 20, 'maxfev': 1000}
    def get_params(x,y,t,k):
        p0 = k
        p3 = 0.0
        sv = 5.
        gx = x
        gy = y
        gt = t
        gv = 5.
        start_state = np.array([0., 0., 0., p0, sv])
        goal_state = np.array([gx, gy, gt, p3, gv])
        x0 = init_x(start_state, goal_state)
        params = np.array([p0, x0[0], x0[1], p3, x0[2]])
        out = spo.minimize(lambda x: trajectory3(x, p0, p3, sv, gx, gy, gt, gv), x0, method='Powell', options=options)
        #print('func', out.fun, out.x)
        if (out.fun > 1.) or (out.x[2] < 0) or (out.x[2] > 4*np.linalg.norm([gx, gy])):
            return -1*np.ones(5,)
        params[1] = out.x[0]
        params[2] = out.x[1]
        params[-1] = out.x[2]
        return np.roll(params, 1)


    Dx, Dy, Dt, Dk = np.meshgrid(delta_x, delta_y, delta_theta, kappa0, indexing='ij')
    idx_range = np.split(np.arange(np.prod(Dx.shape)), num_guys)[guy]

    results = Parallel(n_jobs=-1, verbose=10)(delayed(get_params)(Dx[np.unravel_index(i, Dx.shape)],
                                                                 Dy[np.unravel_index(i, Dx.shape)],
                                                                 Dt[np.unravel_index(i, Dx.shape)],
                                                                 Dk[np.unravel_index(i, Dx.shape)])
                                                                 for i in idx_range)
    results = np.array(results)
    np.savez_compressed('result_'+str(guy)+'.npz', result=results)

    #real = results.reshape((*Dx.shape, results.shape[1]))


    #real2 = np.empty((*Dx.shape, results.shape[1]))
    #for i in range(np.prod(Dx.shape)):
    #    real2[np.unravel_index(i, Dx.shape)] = results[i]
    #assert(np.allclose(real,real2,atol=1e-10))
    #return real, results.reshape(real.shape)
    #np.savez_compressed('lut_inuse.npz', x=delta_x, y=delta_y, theta=delta_theta, kappa=kappa0, lut=real)

def fill_in_lut_holes(lut_file):
    from joblib import Parallel, delayed
    import multiprocessing
    hi = np.load(lut_file)
    lut_x = hi['x']
    lut_y = hi['y']
    lut_theta = hi['theta']
    lut = hi['lut']
    idx_tuple = np.where(lut[:,:,:,0]<=-0.9)
    idx = np.stack(idx_tuple, axis=1)
    pos = np.stack((lut_x[idx[:,0]], lut_y[idx[:,1]], lut_theta[idx[:,2]]),axis=1)
    def get_hole_filler(x,y,t,i):
        p0 = 0.0
        p3 = 0.0
        sv = 5.
        gx = x
        gy = y
        gt = t
        gv = 5.
        start_state = np.array([0., 0., 0., p0, sv])
        goal_state = np.array([gx, gy, gt, p3, gv])
        x0 = init_x(start_state, goal_state)
        params = np.array([p0, x0[0], x0[1], p3, x0[2]])
        out = spo.minimize(lambda x: trajectory3(x, p0, p3, sv, gx, gy, gt, gv), x0, method='Powell')
        if (out.fun > 1.) or (out.x[2] < 0) or (out.x[2] > 4*np.linalg.norm([gx, gy])):
            return -1*np.ones(5,)
        params[1] = out.x[0]
        params[2] = out.x[1]
        params[-1] = out.x[2]
        print('i', i)
        return np.roll(params, 1)

    num_cores = multiprocessing.cpu_count()
    #np.random.shuffle(pos)
    results = Parallel(n_jobs=num_cores)(delayed(get_hole_filler)(pos[i,0], pos[i,1], pos[i,2], i) for i in range(pos.shape[0]))
    results = np.array(results)
    lut[idx_tuple] = results
    np.savez_compressed('lut_inuse.npz', x=lut_x, y=lut_y, theta=lut_theta, lut=lut)
    return idx, lut, results


if __name__ =='__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--guy', type=int, required=True)
    #args = parser.parse_args()
    #build_lut_split(guy=args.guy)
    #assert(1==0)
    build_lut()
    assert False

    #build_lut_split(guy=0, num_guys=3, nx=2, ny=2, nt=2, nk=3)
    #build_lut_split(guy=1, num_guys=3, nx=2, ny=2, nt=2, nk=3)
    #build_lut_split(guy=2, num_guys=3, nx=2, ny=2, nt=2, nk=3)
    #build_lut(nx=2, ny=2, nt=2, nk=3)



    # p0 = 0.0
    # p3 = 0.0
    # sv = 5.
    # gx = 2.
    # gy = 0.1
    # gt = np.pi/2*0
    # gv = 5.
    # p4 = 0.
    # p5 = 0.
    # start_state = np.array([0., 0., 0., p0, sv])
    # goal_state = np.array([gx, gy, gt, p3, gv])
    # x0 = init_x(start_state, goal_state)
    # if ORDER == 3:
    #     params = np.array([p0, x0[0], x0[1], p3, x0[2]])
    #     out = spo.minimize(lambda x: trajectory3(x, p0, p3, sv, gx, gy, gt, gv), x0, method='Powell')
    # elif ORDER == 5:
    #     params = np.array([p0, x0[0], x0[1], p3, p4, p5, x0[2]])
    #     out = spo.minimize(lambda x: trajectory5(x, p0, p3, sv, gx, gy, gt, gv, p4, p5), x0, method='Powell')
    # #next_state = motion_model(start_state, goal_state, params, DT)

    # options = {'maxiter': 20, 'maxfev': 1000}
    # import matplotlib.pyplot as plt
    # curvature_list = []
    # for pzero in np.linspace(-1,1,11):
    #     print(pzero)
    #     start_state = np.array([0., 0., 0., pzero, sv])
    #     x0 = init_x(start_state, goal_state)
    #     params = np.array([pzero, x0[0], x0[1], p3, x0[2]])
    #     out = spo.minimize(lambda x: trajectory3(x, pzero, p3, sv, gx, gy, gt, gv), x0, method='Powell', options=options)
    #     print('func val', out.fun)
    #     print('evals', out.nfev)
    #     print('iterations', out.nit)
    #     params[1] = out.x[0]
    #     params[2] = out.x[1]
    #     params[-1] = out.x[2]
    #     curvature_list.append(np.copy(np.roll(params, 1)))

    # states = integrate_all(np.array(curvature_list))
    # for i in range(int(states.shape[0]/NUM_STEPS)):
    #     idx1 = i*NUM_STEPS
    #     idx2 = (i+1)*NUM_STEPS
    #     plt.plot(states[idx1:idx2,0], states[idx1:idx2,1])

    # plt.axis('equal')
    # plt.show()


    # gy_list = []
    # pzero_test = 0.
    # gx = 5.0
    # for gy in np.linspace(-4,4,100):
    #     print(gy)
    #     pzero_test = 0.
    #     start_state = np.array([0., 0., 0., pzero_test, sv])
    #     pzero_test = 0.
    #     x0 = init_x(start_state, goal_state)
    #     params = np.array([pzero_test, x0[0], x0[1], p3, x0[2]])
    #     out = spo.minimize(lambda x: trajectory3(x, pzero_test, p3, sv, gx, gy, gt, gv), x0, method='Powell', options=options)
    #     print('func val', out.fun)
    #     print('evals', out.nfev)
    #     print('iterations', out.nit)
    #     params[1] = out.x[0]
    #     params[2] = out.x[1]
    #     params[-1] = out.x[2]
    #     gy_list.append(np.copy(np.roll(params, 1)))

    # states2 = integrate_all(np.array(gy_list))
    # for i in range(int(states2.shape[0]/NUM_STEPS)):
    #     idx1 = i*NUM_STEPS
    #     idx2 = (i+1)*NUM_STEPS
    #     plt.plot(states2[idx1:idx2,0], states2[idx1:idx2,1])

    # plt.axis('equal')
    # plt.show()


#     # curvature_list = []
#     # curvature_list.append(np.copy(np.roll(params, 1)))
#     # params[1] = out.x[0]
#     # params[2] = out.x[1]
#     # params[-1] = out.x[2]
#     # curvature_list.append(np.copy(np.roll(params, 1)))

#     # ##states = integrate_path_motion_model(start_state, goal_state, params, DT)
#     # #states = integrate_path(params)
#     # states = integrate_parallel(np.array(curvature_list))
#     # #import matplotlib.pyplot as plt
#     # #plt.plot(states[:NUM_STEPS,0], states[:NUM_STEPS,1])
#     # #plt.plot(states[NUM_STEPS:,0], states[NUM_STEPS:,1])
#     # #plt.axis('equal')
#     # #plt.show()
#     # idx, lut, results = fill_in_lut_holes('./lut_inuse.npz')