"""
NMPC waypoint tracker using CasADi. On init, takes in model equation. 
"""

from dataclasses import dataclass, field
import numpy as np
from numba import njit
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_gym.envs.track import Track
import casadi as ca

@dataclass
class mpc_config:
    NXK: int = 7  # length of dynamic state vector: z = [s, ey, delta, vx, vy, wz, eyaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 5  # finite time horizon length
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 1.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 1.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([0.0, 65.0, 0.0, 0.5, 5.0, 0.0, 15.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [s, ey, delta, vx, vy, wz, eyaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([0.0, 65.0, 0.0, 0.5, 5.0, 0.0, 15.0])
    )  # final state error matrix, penalty  for the final state constraints: [s, ey, delta, vx, vy, wz, eyaw]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_DSTEER: float = -np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 10.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 9.51  # maximum acceleration [m/ss]
    MIN_ACCEL: float = -9.51  # minimum acceleration [m/ss]
    V_SWITCH: float = 1.0  # switching velocity from kinematic to dynamic [m/s]

    # Vehicle parameters
    MU: float = 1.0  # friction coefficient
    C_SF: float = 5.0  # front cornering stiffness
    C_SR: float = 5.0  # rear cornering stiffness
    BF = 1.0  # TODO
    BR = 1.0  # TODO
    DF = None  # friction force front, determined by mu and m post init
    DR = None  # friction force rear, determined by mu and m post init
    LF: float = 0.2735  # distance from center of gravity to front axle
    LR: float = 0.2585  # distance from center of gravity to rear axle
    H: float = 0.1875  # height of center of gravity
    M: float = 15.32  # mass of vehicle
    I: float = 0.64332  # moment of inertia

    def __post_init__(self):
        self.DF = self.MU * self.M * 9.81 / 2.0
        self.DR = self.MU * self.M * 9.81 / 2.0


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
        track: Track = None,
        config: mpc_config = mpc_config(),
        debug=False,
    ):
        self.track = track
        if track is not None:
            self.waypoints = [
                track.raceline.xs.copy(),
                track.raceline.ys.copy(),
                track.raceline.yaws.copy(),
                track.raceline.vxs.copy(),
                track.raceline.ks.copy(),
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
        self.mpc_render = None
        self.goal_state_render = None
        self.waypoint_render = None

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.array(self.waypoints).T[:, :2]
        if self.waypoint_render is None:
            self.waypoint_render = e.render_closed_lines(
                points, color=(128, 0, 0), size=1
            )
        else:
            self.waypoint_render.setData(points)

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
            points = np.array([self.ox, self.oy]).T
            if self.mpc_render is None:
                self.mpc_render = e.render_lines(points, color=(0, 0, 128), size=2)
            else:
                self.mpc_render.setData(points)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, ckap):
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
        ref_traj = np.zeros((self.config.NXK + 1, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(
            np.array([state["pose_x"], state["pose_y"]]), np.array([cx, cy]).T
        )

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]

        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        ref_traj[5, :] = ckap[ind]

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

        self.opti = ca.Opti()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.X = self.opti.variable(self.config.NXK, self.config.TK + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        self.U = self.opti.variable(self.config.NU, self.config.TK)

        # coloumn vector for storing initial state and target state, and curvature
        self.P = self.opti.parameter(self.config.NXK + 1, self.config.TK + 1)

        # state weights matrix converted from config Qk
        Q = ca.diagcat(*np.diag(self.config.Qk))

        # controls weights matrix
        R = ca.diagcat(*np.diag(self.config.Rk))

        # ---- dynamic constraints --------
        def f(x, u, p):
            # params
            cur = p[-1]
            # controls
            a = u[0]
            deltv = u[1]
            # states
            s = x[0]
            ey = x[1]
            delta = x[2]
            vx = x[3]
            vy = x[4]
            wz = x[5]
            epsi = x[6]

            # Compute tire split angle
            alpha_f = delta - ca.atan2(vy + self.config.LF * wz, vx)
            alpha_r = -ca.atan2(vy - self.config.LF * wz, vx)

            # Compute lateral force at front and rear tire
            Fyf = self.config.DF * ca.sin(
                self.config.C_SF * ca.atan(self.config.BF * alpha_f)
            )
            Fyr = self.config.DR * ca.sin(
                self.config.C_SR * ca.atan(self.config.BR * alpha_r)
            )

            # [s, ey, delta, vx, vy, wz, epsi]
            deriv_x_hs = ca.vertcat(
                ((vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1 - cur * ey)),
                (vx * ca.sin(epsi) + vy * ca.cos(epsi)),
                deltv,
                (a - 1 / self.config.M * Fyf * ca.sin(delta) + wz * vy),
                (1 / self.config.M * (Fyf * ca.cos(delta) + Fyr) - wz * vx),
                (
                    1
                    / self.config.I
                    * (self.config.LF * Fyf * ca.cos(delta) - self.config.LR * Fyr)
                ),
                (wz - (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1 - cur * ey) * cur),
            )

            # [s, ey, delta, vx, vy(0.0), wz(0.0), epsi]
            deriv_x_ls = ca.vertcat(
                (vx * ca.cos(epsi)) / (1 - ey * cur),
                (vx * ca.sin(epsi)),
                deltv,
                a,
                0.0,
                0.0,
                (vx * ca.tan(delta)) / (self.config.LR + self.config.LF)
                - cur * ((vx * ca.cos(epsi)) / (1 - cur * ey)),
            )

            deriv_x = ca.if_else(
                ca.sqrt(vx**2 + vy**2) < self.config.V_SWITCH, deriv_x_ls, deriv_x_hs
            )

            return deriv_x

        cost_fn = 0  # cost function

        # initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.P[:-1, 0])

        # runge kutta
        for k in range(self.config.TK):
            st = self.X[:, k]
            con = self.U[:, k]
            p = self.P[:, k + 1]
            cost_fn = cost_fn + (st - p[:-1]).T @ Q @ (st - p[:-1]) + con.T @ R @ con

            st_next = self.X[:, k + 1]
            k1 = f(st, con, p)
            k2 = f(st + self.config.DTK / 2 * k1, con, p)
            k3 = f(st + self.config.DTK / 2 * k2, con, p)
            k4 = f(st + self.config.DTK * k3, con, p)
            st_next_RK4 = st + (self.config.DTK / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(st_next == st_next_RK4)

        self.opti.minimize(cost_fn)

        # control constraints
        self.opti.subject_to(self.U[0, :] > self.config.MIN_ACCEL)
        self.opti.subject_to(self.U[0, :] < self.config.MAX_ACCEL)
        self.opti.subject_to(self.U[1, :] > self.config.MIN_DSTEER)
        self.opti.subject_to(self.U[1, :] < self.config.MAX_DSTEER)

        # state constraints
        self.opti.subject_to(self.X[2, :] > self.config.MIN_STEER)
        self.opti.subject_to(self.X[2, :] < self.config.MAX_STEER)
        self.opti.subject_to(self.X[3, :] > self.config.MIN_SPEED)
        self.opti.subject_to(self.X[3, :] < self.config.MAX_SPEED)

        # solver
        jit_options = {"flags": ["-O3"], "verbose": True}
        ipopt_opts = {
            "ipopt": {
                "print_level": 1,
                "max_iter": 200,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
                "warm_start_init_point": "yes",
            },
            "print_time": 0,
            "jit": True, 
            "compiler": "shell",
            "jit_options": jit_options,
        }
        self.opti.solver("ipopt", ipopt_opts)

    def mpc_prob_solve(self, goal_state, current_state):
        s, ey, epsi = self.track.cartesian_to_frenet(
            current_state["pose_x"],
            current_state["pose_y"],
            current_state["pose_theta"],
        )

        current_state_vec = ca.vertcat(
            s,
            ey,
            current_state["delta"],
            current_state["linear_vel_x"],
            current_state["linear_vel_y"],
            current_state["ang_vel_z"],
            epsi,
        )

        if self.debug:
            self.debug_states = np.array(
                [
                    s,
                    ey,
                    current_state["delta"],
                    current_state["linear_vel_x"],
                    current_state["linear_vel_y"],
                    current_state["ang_vel_z"],
                    epsi,
                    self.ref_path[5][0],
                ]
            )

        self.opti.set_initial(
            self.X, ca.repmat(current_state_vec, 1, self.config.TK + 1)
        )
        self.opti.set_value(
            self.P,
            ca.horzcat(
                ca.vertcat(current_state_vec, 0.0),
                ca.repmat(goal_state, 1, self.config.TK),
            ),
        )

        # solve
        sol = self.opti.solve()

        # extract solution
        u_sol = sol.value(self.U)
        x_sol = sol.value(self.X)

        self.oa = u_sol[0, :].flatten()
        self.odelta_v = u_sol[1, :].flatten()
        self.odelta = x_sol[2, :].flatten()

        self.ox = x_sol[0, :].flatten()
        self.oy = x_sol[1, :].flatten()
        # TODO convert back to cartesian
        for i, (s, ey) in enumerate(zip(self.ox, self.oy)):
            curr_x, curr_y, _ = self.track.frenet_to_cartesian(s, ey, 0.0)
            self.ox[i] = curr_x
            self.oy[i] = curr_y

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

        # calculate the reference trajectory
        self.ref_path = self.calc_ref_trajectory(
            current_state,
            self.waypoints[0],
            self.waypoints[1],
            self.waypoints[2],
            self.waypoints[3],
            self.waypoints[4],
        )

        # Goal state is the last point's velocity and all zeros for the other states (s, ey, delta, vx, vy, wz, epsi, curv)
        goal_state = ca.vertcat(
            0.0, 0.0, 0.0, self.ref_path[3][-1], 0.0, 0.0, 0.0, self.ref_path[5][0]
        )

        # solve the NMPC problem
        oa, odelta_v = self.mpc_prob_solve(goal_state, current_state)

        if self.debug:
            return oa, odelta_v, self.debug_states
        else:
            return oa, odelta_v