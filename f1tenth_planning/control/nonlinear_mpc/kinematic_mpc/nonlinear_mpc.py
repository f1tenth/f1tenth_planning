"""
NMPC waypoint tracker using CasADi. On init, takes in model equation. 
"""
from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_gym.gym.f110_gym.envs.track import Track
import casadi as ca

@dataclass
class mpc_config:
    NXK: int = 5  # length of kinematic state vector: z = [x, y, v, yaw, steer]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([18.5, 18.5, 3.5, 0.1])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([18.5, 18.5, 3.5, 0.1])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_DSTEER: float = -np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MIN_ACCEL: float = -3.0  # minimum acceleration [m/ss]
    WHEELBASE: float = 0.33  # wheelbase [m]

class NMPCPlanner:
    """
    NMPC Controller, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        config: mpc_config = mpc_config(),
        debug=False,
    ):
        self.waypoints = [
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.yaws,
            track.raceline.vxs,
        ]
        self.config = config
        self.oa = None
        self.odelta_v = None
        self.ox = None
        self.oy = None
        self.ref_path = None
        self.debug = debug
        self.mpc_prob_init()

        self.drawn_waypoints = []

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.ref_path is not None:
            points = self.ref_path[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)

    def render_mpc_sol(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.ox is not None and self.oy is not None:
            e.render_lines(np.array([self.ox, self.oy]).T, color=(0, 0, 128), size=2)

    def plan(self, states, waypoints=None):
        return 
    
    def calc_ref_trajectory_kinematic(self, state, cx, cy, cyaw, sp):
        return 
    
    def mpc_prob_init(self):
        
        x = ca.SX.sym("x"),
        y = ca.SX.sym("y"),
        delta = ca.SX.sym("delta"),
        v = ca.SX.sym("v"),
        yaw = ca.SX.sym("yaw"),
        states = ca.vertcat(
            x,
            y,
            delta,
            v,
            yaw
        )
        n_states = states.numel()

        # control symbolic variables
        a = ca.SX.sym('a')
        delta_v = ca.SX.sym('delta_v')
        controls = ca.vertcat(
            a,
            delta_v
        )
        n_controls = controls.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', n_states, self.config.TK + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', n_controls, self.config.TK)

        # coloumn vector for storing initial state and target state
        P = ca.SX.sym('P', n_states, self.config.TK+1)

        # state weights matrix converted from config Qk
        Q = ca.diagcat(*np.diag(self.config.Qk))

        # controls weights matrix
        R = ca.diagcat(*np.diag(self.config.Rk))

        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
        # ---- dynamic constraints --------
        fsteer = lambda delta, vdelta: vdelta # ideal, continuous time steering-speed
        facc = lambda speed, along: along # ideal, continuous time acceleration
        RHS = ca.vertcat(
                            x[3] * ca.cos(x[4]),  # dx/dt = v * cos(yaw)
                            x[3] * ca.sin(x[4]),  # dy/dt = v * sin(yaw)
                            u[1],  # d(delta)/dt = delta_v
                            u[0],  # dv/dt = a
                            (x[3]/(self.config.WHEELBASE)) * ca.tan(x[2])  # dyaw/dt = (v/(Lx+Ly)) * tan(delta)
                        ) # dx/dt = f(x,u)

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        f = ca.Function('f', [states, controls], [RHS])

        cost_fn = 0  # cost function
        g = X[:, 0] - P[:, 0]  # x(0) = x0 constraint in the equation

        # runge kutta
        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            cost_fn = cost_fn \
                + (st - P[:, k+1]).T @ Q @ (st - P[:, k+1]) \
                + con.T @ R @ con    
            
            st_next = X[:, k+1]
            k1 = f(st, con)
            k2 = f(st + self.config.DTK/2*k1, con)
            k3 = f(st + self.config.DTK/2*k2, con)
            k4 = f(st + self.config.DTK * k3, con)
            st_next_RK4 = st + (self.config.DTK / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
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
        # Solver initialization, this is the main solver for the NMPC problem which will be called at each time step
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, ipopt_opts)
        
        lbx = -ca.inf*ca.DM.ones((n_states*(self.config.TK+1) + n_controls*self.config.TK, 1))
        ubx =  ca.inf*ca.DM.ones((n_states*(self.config.TK+1) + n_controls*self.config.TK, 1))


        lbx[0+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # X lower bound
        lbx[1+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # Y lower bound
        lbx[2+n_states: n_states*(self.config.TK+1): n_states] = self.config.MIN_STEER        # delta lower bound\
        lbx[3+n_states: n_states*(self.config.TK+1): n_states] = self.config.MIN_SPEED        # vx lower bound
        lbx[4+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # theta lower bound

        ubx[0+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # X upper bound
        ubx[1+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # Y upper bound
        ubx[2+n_states: n_states*(self.config.TK+1): n_states] = self.config.MAX_STEER        # delta upper bound
        ubx[3+n_states: n_states*(self.config.TK+1): n_states] = self.config.MAX_SPEED        # vx upper bound
        ubx[4+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # theta upper bound

        lbx[n_states*(self.config.TK+1)::n_controls]           = self.config.MIN_ACCEL            # lower bound for a_x
        ubx[n_states*(self.config.TK+1)::n_controls]           = self.config.MAX_ACCEL            # upper bound for a_x
        lbx[n_states*(self.config.TK+1)+1::n_controls]         = self.config.MIN_DSTEER          # lower bound for delta_v
        ubx[n_states*(self.config.TK+1)+1::n_controls]         = self.config.MAX_DSTEER          # upper bound for delta_v

        # lbg is all zeros
        lbg = ca.vertcat(
            ca.DM.zeros((n_states*(self.config.TK+1), 1)),
        )
        ubg = ca.vertcat(
            ca.DM.zeros((n_states*(self.config.TK+1), 1)),
        )

        # store the arguments for the solver, these are updated at each time step
        self.args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }
        self.U0 = ca.DM.zeros((n_controls, self.config.TK))

        return
    
    def mpc_prob_solve(self, goal_state, x0):

        # ref_path is goal_state repeated TK times
        self.ref_path = np.array([goal_state for _ in range(self.config.TK+1)]).T

        self.args['p'] = ca.horzcat(
            x0,    # current state
            ca.repmat(goal_state, 1, self.config.TK)   # target state
        )
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(x0, self.config.NXK*(self.config.TK+1), 1),
            ca.reshape(self.U0, self.config.NU*self.config.TK, 1)
        )
        sol = self.solver(
            x0=self.args['x0'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            p=self.args['p']
        )
        
        u_sol = ca.reshape(sol['x'][self.config.NXK*(self.config.TK+1):], self.config.NU, self.config.TK)
        x_sol = ca.reshape(sol['x'][:self.config.NXK*(self.config.TK+1)], self.config.NXK, self.config.TK+1)

        self.ox = x_sol[0, :].full().flatten()
        self.oy = x_sol[1, :].full().flatten()
        self.odelta = x_sol[2, :].full().flatten()
        self.oa = u_sol[0, :].full().flatten()
        self.odelta_v = u_sol[1, :].full().flatten()

        # Return the first control action
        return self.oa[0], self.odelta_v[0]
