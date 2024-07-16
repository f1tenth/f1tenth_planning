"""
NMPC waypoint tracker using CasADi. On init, takes in model equation. 
"""
from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.utils.utils import nearest_point, intersect_point
from f1tenth_gym.envs.track import Track
import casadi as ca

@dataclass
class mpc_config:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, delta, v_x, yaw, yaw_rate, beta]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 5  # finite time horizon length kinematic
    Rk: list = field(
        default_factory=lambda: np.diag([0.5, 4.0])
    )  # input cost matrix, penalty for inputs - [steering_speed, accel]
    Rd: list = field(
        default_factory=lambda: np.diag([0.3, 4.0])
    )  # input difference cost matrix, penalty for change of inputs - [steering_speed, accel]
    Qk: list = field(
        default_factory=lambda: np.diag([18.5, 18.5, 0.0, 1.5, 0.0, 0.0, 0.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qf: list = field(
        default_factory=lambda: np.diag([18.5, 18.5, 0.0, 1.5, 0.0, 0.0, 0.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
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

    # Vehicle parameters
    MU: float = 1.0489 # friction coefficient
    C_SF: float = 4.718 # front cornering stiffness
    C_SR: float = 5.4562 # rear cornering stiffness
    LF: float = 0.15875 # distance from center of gravity to front axle
    LR: float = 0.17145 # distance from center of gravity to rear axle
    H: float = 0.074 # height of center of gravity
    M: float = 3.74 # mass of vehicle
    I: float = 0.04712 # moment of inertia

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
        # [x, y, delta, v_x, yaw, yaw_rate, beta]
        self.waypoints = [
            track.raceline.xs,
            track.raceline.ys,
            np.zeros_like(track.raceline.xs),
            track.raceline.vxs,
            track.raceline.yaws,
            np.zeros_like(track.raceline.xs),
            np.zeros_like(track.raceline.xs),
        ]
        self.config = config
        self.oa = None
        self.odelta_v = None
        self.ox = None
        self.oy = None
        self.ref_path = None
        self.ref_point = None
        self.debug = debug
        self.mpc_prob_init()
        self.max_reacquire = 20.0

        self.drawn_waypoints = []

    def _get_current_waypoint(self, lookahead_distance, position):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """

        waypoints = np.array(self.waypoints).T
        nearest_p, nearest_dist, t, i = nearest_point(position, waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.current_index, t2 = intersect_point(
                position,
                lookahead_distance,
                waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.current_index is None:
                return None
            current_waypoint = waypoints[self.current_index, :]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return waypoints[i, :]
        else:
            return None
        
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

    def render_goal_state(self, e):
        """
        update goal state being drawn by EnvRenderer
        """
        if self.ref_point is not None:
            points = self.ref_point[:2][None]
            e.render_points(points, color=(0, 128, 0), size=3)

    def render_mpc_sol(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.ox is not None and self.oy is not None:
            e.render_lines(np.array([self.ox, self.oy]).T, color=(0, 0, 128), size=2) 
    
    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state["pose_x"], state["pose_y"]]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]

        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state["linear_vel_x"]) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[3, :] = sp[ind_list]
        cyaw[cyaw - state["pose_theta"] > 4.5] = np.abs(
            cyaw[cyaw - state["pose_theta"] > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state["pose_theta"] < -4.5] = np.abs(
            cyaw[cyaw - state["pose_theta"] < -4.5] + (2 * np.pi)
        )
        ref_traj[4, :] = cyaw[ind_list]

        return ref_traj
    
    def mpc_prob_init(self):
        
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        delta = ca.SX.sym('delta')
        v = ca.SX.sym('v')
        yaw = ca.SX.sym('yaw')
        yaw_rate = ca.SX.sym('yaw_rate')
        beta = ca.SX.sym('beta')

        states = ca.vertcat(
            x,
            y,
            delta,
            v,
            yaw,
            yaw_rate,
            beta
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

        # set gravity constant
        g = 9.81  # [m/s^2]

        mu = self.config.MU
        C_Sf = self.config.C_SF
        C_Sr = self.config.C_SR
        lf = self.config.LF
        lr = self.config.LR
        h = self.config.H
        m = self.config.M
        I = self.config.I

        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
        # ---- dynamic constraints --------
        fsteer = lambda delta, vdelta: vdelta # ideal, continuous time steering-speed
        facc = lambda speed, along: along # ideal, continuous time acceleration
        
        v_s = 2.0
        v_b = 0.1
        v_min = v_s/2

        # weights for mixing both models
        w_std = 0.5 * (ca.tanh((v - v_s)/v_b) + 1)
        w_ks = 1 - w_std

        dyaw_slow = v * ca.cos(beta) * ca.tan(delta) / (lr + lf)
        d_beta_slow = (lr * delta_v) / ((lr + lf) * ca.cos(delta) ** 2 * (1 + (ca.tan(delta) ** 2 * lr / (lr + lf)) ** 2))
        dyaw_rate_slow = 1 / (lr + lf) * (a * ca.cos(beta) * ca.tan(delta) -
                            v * ca.sin(beta) * ca.tan(delta) * d_beta_slow  +
                            v * ca.cos(beta) * delta_v / (ca.cos(delta) ** 2))
        
        dyaw_fast = yaw_rate                # dyaw/dt = yaw_rate
        dyaw_rate_fast = -mu * m / (v * I * (lr + lf)) * (
                                        lf ** 2 * C_Sf * (g * lr - a * h) + lr ** 2 * C_Sr * (g * lf + a * h)) * yaw_rate \
                            + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h)) * beta \
                            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta # dyaw_rate/dt = RHS
        d_beta_fast = (mu / (v ** 2 * (lr + lf)) * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf) - 1) * yaw_rate \
                            - mu / (v * (lr + lf)) * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h)) * beta \
                            + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta    # dbeta/dt = RHS
        
        # Mixed model using weights
        # For a switched model, check the commented code below
        RHS = ca.vertcat(
                            v * ca.cos(yaw + beta),  # dx/dt = v * cos(yaw + beta)
                            v * ca.sin(yaw + beta),  # dy/dt = v * sin(yaw + beta)
                            delta_v,                 # d(delta)/dt = delta_v
                            a,                       # dv/dt = a
                            w_std * dyaw_fast + w_ks * dyaw_slow, # dyaw/dt = f(x,u)
                            w_std * dyaw_rate_fast + w_ks * dyaw_rate_slow, # dyaw_rate/dt = f(x,u)
                            w_std * d_beta_fast + w_ks * d_beta_slow # dbeta/dt = f(x,u)
                        ) # dx/dt = f(x,u)

        RHS_LOW_SPEED = ca.vertcat(
                            v * ca.cos(yaw + beta),  # dx/dt = v * cos(yaw + beta)
                            v * ca.sin(yaw + beta),  # dy/dt = v * sin(yaw + beta)
                            delta_v,                 # d(delta)/dt = delta_v
                            a,                       # dv/dt = a
                            dyaw_slow,  # dyaw/dt = yaw_rate
                            dyaw_rate_slow,                  # dyaw_rate/dt = RHS
                            d_beta_slow                   # dbeta/dt = d_beta
                        ) # dx/dt = f(x,u)
    
        RHS_DYN = ca.vertcat(
                            v * ca.cos(yaw + beta),  # dx/dt = v * cos(yaw + beta)
                            v * ca.sin(yaw + beta),  # dy/dt = v * sin(yaw + beta)
                            delta_v,                 # d(delta)/dt = delta_v
                            a,                       # dv/dt = a
                            yaw_rate,                # dyaw/dt = yaw_rate
                            -mu * m / (v * I * (lr + lf)) * (
                                        lf ** 2 * C_Sf * (g * lr - a * h) + lr ** 2 * C_Sr * (g * lf + a * h)) * yaw_rate \
                            + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h)) * beta \
                            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta, # dyaw_rate/dt = RHS
                            (mu / (v ** 2 * (lr + lf)) * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf) - 1) *
                            yaw_rate \
                            - mu / (v * (lr + lf)) * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h)) * beta \
                            + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta    # dbeta/dt = RHS
                        ) # dx/dt = f(x,u)

        RHS = ca.if_else(v > v_s, RHS, RHS_LOW_SPEED)

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        f = ca.Function('f', [states, controls], [RHS])

        cost_fn = 0  # cost function
        g = X[:, 0] - P[:, 0]  # x(0) = x0 constraint in the equation

        # runge kutta
        for k in range(self.config.TK):
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

        cost_fn = cost_fn + (X[:, -1] - P[:, -1]).T @ self.config.Qf @ (X[:, -1] - P[:, -1])
        
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
                'max_iter': 2000,
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

        # [x, y, delta, v_x, yaw, yaw_rate, beta]
        lbx[0+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # X lower bound
        lbx[1+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # Y lower bound
        lbx[2+n_states: n_states*(self.config.TK+1): n_states] = self.config.MIN_STEER        # delta lower bound\
        lbx[3+n_states: n_states*(self.config.TK+1): n_states] = self.config.MIN_SPEED        # vx lower bound
        lbx[4+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # theta lower bound
        lbx[5+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # theta_dot lower bound
        lbx[6+n_states: n_states*(self.config.TK+1): n_states] = -np.inf     # beta lower bound

        ubx[0+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # X upper bound
        ubx[1+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # Y upper bound
        ubx[2+n_states: n_states*(self.config.TK+1): n_states] = self.config.MAX_STEER        # delta upper bound
        ubx[3+n_states: n_states*(self.config.TK+1): n_states] = self.config.MAX_SPEED        # vx upper bound
        ubx[4+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # theta upper bound
        ubx[5+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # theta_dot upper bound
        ubx[6+n_states: n_states*(self.config.TK+1): n_states] = np.inf     # beta upper bound

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
        # [x, y, delta, v_x, yaw, yaw_rate, beta]
        curr_state = ca.vertcat(
            x0["pose_x"],
            x0["pose_y"],
            x0["delta"],
            x0["linear_vel_x"],
            x0["pose_theta"],
            x0["ang_vel_z"],
            x0["beta"]
        )

        self.args['p'] = ca.horzcat(
            curr_state,    # current state
            ca.repmat(goal_state, 1, self.config.TK)   # target state
        )
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(ca.repmat(curr_state, 1, self.config.TK+1), self.config.NXK*(self.config.TK+1), 1),
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

    def plan(self, current_state):
        """
        Plan a trajectory using the NMPC controller.

        Args:
            current_state (f1tenth_gym_ros.msg.State): current state of the vehicle

        Returns:
            (float, float): steering angle and acceleration
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan()"
            )

        # if current_state["linear_vel_x"] < 0.1:
        #     return self.config.MAX_ACCEL, 0.0

        # # calculate the reference trajectory
        # self.ref_path = self.calc_ref_trajectory(
        #     current_state, self.waypoints[0], self.waypoints[1], self.waypoints[2], self.waypoints[3]
        # )

        # # Goal state is the last point in the reference trajectory
        # goal_state = self.ref_path[:, -1]

        # # Goal velocity is that of the first point in the reference trajectory
        # goal_state[3] = self.ref_path[3, 0]
        
        v_lookahead = max(current_state['linear_vel_x'], 0.1)
        lookahead_distance = v_lookahead * (self.config.TK * self.config.DTK)
        goal_state = self._get_current_waypoint(lookahead_distance, np.array([current_state["pose_x"], current_state["pose_y"]]))
        
        # Ref point is the goal_state, with the velocity from the closest point
        self.ref_point = goal_state.copy()

        closest_point = self._get_current_waypoint(0.0, np.array([current_state["pose_x"], current_state["pose_y"]]))
        self.ref_point[3] = closest_point[3]
        
        # solve the NMPC problem
        oa, odelta_v = self.mpc_prob_solve(goal_state, current_state)

        return oa, odelta_v