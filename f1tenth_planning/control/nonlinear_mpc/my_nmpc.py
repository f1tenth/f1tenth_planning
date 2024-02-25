from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi, tan
import matplotlib.pyplot as plt

# setting matrix_weights' variables
Q_x = 1
Q_y = 1
Q_theta = 1
Q_delta = 1
Q_vx = 0.5
Ra = 0.01
Rvd = 0.1

factor = 1
step_horizon = 0.1  # time between steps in seconds
N = 10              # number of look ahead steps
Lx = factor*0.3            # L in J Matrix (half robot x-axis length)
Ly = factor*0.3            # l in J Matrix (half robot y-axis length)
sim_time = 40      # simulation time

ax_min          = -3            # lower bound for a_x
ax_max          = 3             # upper bound for a_x

# specs
x_init = 0
y_init = 0
delta_init = 0
vx_init = 0
theta_init = 0

x_target = 25
y_target = 25
delta_target = 0
vx_target = 0
theta_target = 0

g = 9.81

x_max      = ca.inf
y_max      = ca.inf
d_max      = 0.9
vx_max     = 10
theta_max  = ca.inf

x_min      = -ca.inf
y_min      = -ca.inf
d_min      = -0.9
vx_min     = -10
theta_min  = -ca.inf

t0 = 0
state_init = ca.DM([x_init, y_init, delta_init, vx_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, delta_target, vx_target, theta_target])  # target state

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
delta = ca.SX.sym('delta')
V_x = ca.SX.sym('V_x')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    delta,
    V_x,
    theta
)
n_states = states.numel()

# control symbolic variables
A_x = ca.SX.sym('A_x')
V_d = ca.SX.sym('V_d')
controls = ca.vertcat(
    A_x,
    V_d,
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states, N+1)

# state weights matrix (Q_X, Q_Y,Q_DELTA, Q_VX Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_delta, Q_vx, Q_theta)

# controls weights matrix
R = ca.diagcat(Ra, Rvd)

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
# ---- dynamic constraints --------
fsteer = lambda delta, vdelta: vdelta # ideal, continuous time steering-speed
facc = lambda speed, along: along # ideal, continuous time acceleration
RHS = ca.vertcat(
                    V_x*cos(theta),
                    V_x*sin(theta),
                    V_d,
                    A_x,
                    (V_x/(Lx+Ly)) * tan(delta)
                ) # dx/dt = f(x,u)

# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:, 0]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[:, k+1]).T @ Q @ (st - P[:, k+1]) \
        + con.T @ R @ con    
    
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

ipopt_opts = {
    'ipopt': {
        'print_level': 0,
        'max_iter': 1000,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6,
        'warm_start_init_point': 'yes',
    },
    'print_time': 0,
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, ipopt_opts)

lbx = -ca.inf*ca.DM.ones((n_states*(N+1) + n_controls*N, 1))
ubx =  ca.inf*ca.DM.ones((n_states*(N+1) + n_controls*N, 1))


lbx[0+n_states: n_states*(N+1): n_states] = x_min     # X lower bound
lbx[1+n_states: n_states*(N+1): n_states] = y_min     # Y lower bound
lbx[2+n_states: n_states*(N+1): n_states] = d_min        # delta lower bound
lbx[3+n_states: n_states*(N+1): n_states] = vx_min        # vx lower bound
lbx[4+n_states: n_states*(N+1): n_states] = theta_min     # theta lower bound

ubx[0+n_states: n_states*(N+1): n_states] = x_max      # X upper bound
ubx[1+n_states: n_states*(N+1): n_states] = y_max      # Y upper bound
ubx[2+n_states: n_states*(N+1): n_states] = d_max         # delta upper bound
ubx[3+n_states: n_states*(N+1): n_states] = vx_max         # vx upper bound
ubx[4+n_states: n_states*(N+1): n_states] = theta_max      # theta upper bound

lbx[n_states*(N+1)::n_controls]           = ax_min            # lower bound for a_x
ubx[n_states*(N+1)::n_controls]           = ax_max            # upper bound for a_x

# lbg is all zeros
lbg = ca.vertcat(
    ca.DM.zeros((n_states*(N+1), 1)),
)
ubg = ca.vertcat(
    ca.DM.zeros((n_states*(N+1), 1)),
)

args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0)
times = np.array([[0]])

cat_ref = DM2Arr(ca.hcat((state_init, state_target)))

###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec

    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        
        args['p'] = ca.horzcat(
            state_init,    # current state
            ca.repmat(state_target, 1, N)   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)


        cat_ref = np.dstack((
            cat_ref,
            ca.hcat((state_init, state_target))
        ))

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.dstack((
            cat_controls,
            DM2Arr(u)
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u , f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(shift_timestep(step_horizon, t0, X0[:,-1], u[:,-1], f)[1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1
        
        # if(np.allclose(u[:,0], 0)):
        #     break
        # print(state_init)
        
        # break

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)