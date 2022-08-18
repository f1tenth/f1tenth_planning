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
Dynamic Single Track MPC waypoint tracker

Author: Hongrui Zheng, Johannes Betz, Ahmad Amine
Last Modified: 8/1/22
"""

import math
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import cvxpy
import numpy as np
from f1tenth_planning.utils.utils import nearest_point, pi_2_pi, quat_2_rpy
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from cvxpy.atoms.affine.wraps import psd_wrap


@dataclass
class mpc_config:
    NX: int = 7  # length of state vector: z = [x, y, delta, v, yaw, yaw rate, beta]
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    T: int = 40  # finite time horizon length
    TK: int = 8  # finite time horizon length kinematic
    R: list = field(
        default_factory=lambda: diags([0.5, 0.01])
    )  # input cost matrix, penalty for inputs - [steering_speed, accel]
    Rd: list = field(
        default_factory=lambda: diags([0.3, 0.01])
    )  # input difference cost matrix, penalty for change of inputs - [steering_speed, accel]
    Q: list = field(
        default_factory=lambda: diags([32.0, 32.0, 0.0, 1.0, 0.5, 0.0, 0.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qf: list = field(
        default_factory=lambda: diags([32.0, 32.0, 0.0, 1.0, 0.5, 0.0, 0.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    N_IND_SEARCH: int = 20  # Search index number
    DT: float = 0.025  # time step [s]
    DTK: float = 0.1  # time step [s] kinematic
    dl: float = 0.03  # dist step [m]
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    V_KS: float = 2.0  # switching velocity from kinematic to dynamic [m/s]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class STMPCPlanner:
    """
    Dynamic Single Track MPC Controller, uses the ST model from Common Road

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track
        mass, l_f, l_r, h_CoG, c_f, c_r, Iz, mu

    Attributes:
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
    """

    def __init__(
        self,
        waypoints=None,
        config=mpc_config(),
        params=np.array(
            [3.74, 0.15875, 0.17145, 0.074, 4.718, 5.4562, 0.04712, 1.0489]
        ),
        debug=False,
    ):
        self.waypoints = waypoints
        self.config = config
        self.vehicle_params = params
        self.odelta_v = None
        self.oa = None
        self.init_flag = 0
        self.mpc_prob_init_kinematic()
        self.debug = debug

    def plan(self, states, waypoints=None):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle

        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )
        vehicle_state = State(
            x=states[0],
            y=states[1],
            delta=states[2],
            v=states[3],
            yaw=states[4],
            yawrate=states[5],
            beta=states[6],
        )

        if states[3] <= self.config.V_KS:

            (
                speed,
                steering_angle,
                mpc_ref_path_x,
                mpc_ref_path_y,
                mpc_pred_x,
                mpc_pred_y,
                mpc_ox,
                mpc_oy,
            ) = self.MPC_Control_kinematic(vehicle_state, self.waypoints)
        else:

            (
                speed,
                steering_angle,
                mpc_ref_path_x,
                mpc_ref_path_y,
                mpc_pred_x,
                mpc_pred_y,
                mpc_ox,
                mpc_oy,
            ) = self.MPC_Control(vehicle_state, self.waypoints, self.vehicle_params)

        return steering_angle, speed

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
        ref_traj = np.zeros((self.config.NX, self.config.T + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[3, 0] = sp[ind]
        ref_traj[4, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DT
        dind = travel / self.config.dl
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.T)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[3, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 5] = np.abs(cyaw[cyaw - state.yaw > 5] - (2 * np.pi))
        cyaw[cyaw - state.yaw < -5] = np.abs(cyaw[cyaw - state.yaw < -5] + (2 * np.pi))
        ref_traj[4, :] = cyaw[ind_list]

        return ref_traj

    def calc_ref_trajectory_kinematic(self, state, cx, cy, cyaw, sp):
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
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 5] = np.abs(cyaw[cyaw - state.yaw > 5] - (2 * np.pi))
        cyaw[cyaw - state.yaw < -5] = np.abs(cyaw[cyaw - state.yaw < -5] + (2 * np.pi))
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od_v, xref, vehicle_params):

        # Create Vector that includes the predicted path for the next T time steps for all vehicle states
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        # Calculate/Predict the vehicle states/motion for the next T time steps
        state = State(
            x=x0[0], y=x0[1], delta=x0[2], v=x0[3], yaw=x0[4], yawrate=x0[5], beta=x0[6]
        )
        for (ai, d_vi, i) in zip(oa, od_v, range(1, self.config.T + 1)):
            state = self.update_state(state, ai, d_vi, vehicle_params)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.delta
            path_predict[3, i] = state.v
            path_predict[4, i] = state.yaw
            path_predict[5, i] = state.yawrate
            path_predict[6, i] = state.beta

        return path_predict

    def predict_motion_kinematic(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state_kinematic(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta_v, vehicle_params):
        # Extract Vehicle parameter to calculate vehicle dynamic model
        mass = vehicle_params[0]  # mass in [kg]
        l_f = vehicle_params[1]  # Distance CoG to front in [m]
        l_r = vehicle_params[2]  # Distance CoG to back in [m]
        h_CoG = vehicle_params[3]  # Height of the vehicle in [m]
        c_f = vehicle_params[4]  # Cornering Stiffness front in [N]
        c_r = vehicle_params[5]  # Cornering Stiffness back in [N]
        Iz = vehicle_params[6]  # Vehicle Inertia [kg m2]
        mu = vehicle_params[7]  # friction coefficient [-]
        g = 9.81  # Vertical acceleration  [m/s2]

        # Input check
        if delta_v >= self.config.MAX_STEER_V:
            delta_v = self.config.MAX_STEER_V
        elif delta_v <= -self.config.MAX_STEER_V:
            delta_v = -self.config.MAX_STEER_V

        # Input check
        if a >= self.config.MAX_ACCEL:
            a = self.config.MAX_ACCEL
        elif a <= -self.config.MAX_ACCEL:
            a = -self.config.MAX_ACCEL

        # Calculate substitute/helper variables for all functions
        K = (mu * mass) / ((l_f + l_r) * Iz)
        T = (g * l_r) - (a * h_CoG)
        V = (g * l_f) + (a * h_CoG)
        F = l_f * c_f
        R = l_r * c_r
        M = (mu * c_f) / (l_f + l_r)
        N = (mu * c_r) / (l_f + l_r)

        A1 = K * F * T
        A2 = K * (R * V - F * T)
        A3 = K * (l_f * l_f * c_f * T + l_r * l_r * c_r * V)
        A4 = M * T
        A5 = N * V + M * T
        A6 = N * V * l_r - M * T * l_f

        # Dynamic Motion calculation; State = [x, y, delta, v, yaw, yaw rate, beta]
        x_new = state.x + state.v * math.cos(state.yaw + state.beta) * self.config.DT
        y_new = state.y + state.v * math.sin(state.yaw + state.beta) * self.config.DT
        delta_new = state.delta + delta_v * self.config.DT
        v_new = state.v + a * self.config.DT
        yaw_new = (
            state.yaw
            + state.v / self.config.WB * math.tan(state.delta) * self.config.DT
        )

        yawrate_new = (
            state.yawrate
            + (A1 * state.delta + A2 * state.beta - A3 * (state.yawrate / state.v))
            * self.config.DT
        )
        beta_new = (
            state.beta
            + (
                A4 * (state.delta / state.v)
                - A5 * (state.beta / state.v)
                + A6 * (state.yawrate / (state.v * state.v))
                - state.yawrate
            )
            * self.config.DT
        )

        # Create an additional set of values so the current states are not overritten
        state.x = x_new
        state.y = y_new
        state.delta = delta_new
        state.v = v_new
        state.yaw = yaw_new
        state.yawrate = yawrate_new
        state.beta = beta_new

        # Output  check
        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        # Output  check
        if state.delta >= self.config.MAX_STEER:
            state.delta = self.config.MAX_STEER
        elif state.delta <= -self.config.MAX_STEER:
            state.delta = -self.config.MAX_STEER

        return state

    def update_state_kinematic(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + state.v / self.config.WB * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_dynamic_model_matrix(self, delta, v, yaw, yawrate, beta, a, vehicle_params):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, delta, v, yaw, yaw rate, beta]
        :param delta: steering angle
        :param v: speed
        :param phi: heading angle of the vehicle

        :return: A, B, C
        """
        # Extract Vehicle parameter to calculate vehicle dynamic model
        mass = vehicle_params[0]  # mass in [kg]
        l_f = vehicle_params[1]  # Distance CoG to front in [m]
        l_r = vehicle_params[2]  # Distance CoG to back in [m]
        h_CoG = vehicle_params[3]  # Height of the vehicle in [m]
        c_f = vehicle_params[4]  # Cornering Stiffness front in [N]
        c_r = vehicle_params[5]  # Cornering Stiffness back in [N]
        Iz = vehicle_params[6]  # Vehicle Inertia [kg m2]
        mu = vehicle_params[7]  # friction coefficient [-]
        g = 9.81  # Vertical acceleration  [m/s2]

        # Calculate substitute/helper variables for all functions
        K = (mu * mass) / ((l_f + l_r) * Iz)
        T = (g * l_r) - (a * h_CoG)
        V = (g * l_f) + (a * h_CoG)
        F = l_f * c_f
        R = l_r * c_r
        M = (mu * c_f) / (l_f + l_r)
        N = (mu * c_r) / (l_f + l_r)

        A1 = K * F * T
        A2 = K * (R * V - F * T)
        A3 = K * (l_f * l_f * c_f * T + l_r * l_r * c_r * V)
        A4 = M * T
        A5 = N * V + M * T
        A6 = N * V * l_r - M * T * l_f

        B1 = (
            (-h_CoG * F * K) * delta
            + (h_CoG * K * (F + R)) * beta
            - (h_CoG * K * ((l_r * l_r * c_r) - (l_f * l_f * c_f))) * (yawrate / v)
        )
        B2 = (
            (-h_CoG * M) * (delta / v)
            - h_CoG * (N - M) * (beta / v)
            + h_CoG * (l_f * M + l_r * N) * (yawrate / (v * v))
        )

        # --------------  State (or system) matrix A, 7x7
        A = np.zeros((self.config.NX, self.config.NX))
        # Diagonal
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[4, 4] = 1.0

        A[5, 5] = -self.config.DT * (A3 / v) + 1
        A[6, 6] = -self.config.DT * A5 + 1

        # Zero row
        A[0, 3] = self.config.DT * math.cos(yaw + beta)
        A[0, 4] = -self.config.DT * v * math.sin(yaw + beta)
        A[0, 6] = -self.config.DT * v * math.sin(yaw + beta)
        # First Row
        A[1, 3] = self.config.DT * math.sin(yaw + beta)
        A[1, 4] = self.config.DT * v * math.cos(yaw + beta)
        A[1, 6] = self.config.DT * v * math.cos(yaw + beta)
        # Fourth Row
        A[4, 5] = self.config.DT

        # Fifth Row
        A[5, 2] = self.config.DT * A1
        A[5, 3] = self.config.DT * A3 * (yawrate / (v * v))
        A[5, 6] = self.config.DT * A2
        # Sixth Row
        A[6, 2] = self.config.DT * (A4 / v)
        A[6, 3] = (
            self.config.DT
            * (-A4 * beta * v + A5 * beta * v - A6 * 2 * yawrate)
            / (v * v * v)
        )
        A[6, 5] = self.config.DT * ((A6 / (v * v)) - 1)

        # -------------- Input Matrix B; 7x2
        B = np.zeros((self.config.NX, self.config.NU))
        B[2, 0] = self.config.DT
        B[3, 1] = self.config.DT

        B[5, 1] = self.config.DT * B1
        B[6, 1] = self.config.DT * B2

        # -------------- Matrix C; 7x1
        C = np.zeros(self.config.NX)
        C[0] = self.config.DT * (
            v * math.sin(yaw + beta) * yaw + v * math.sin(yaw + beta) * beta
        )
        C[1] = self.config.DT * (
            -v * math.cos(yaw + beta) * yaw - v * math.cos(yaw + beta) * beta
        )

        C[5] = self.config.DT * (-A3 * (yawrate / v) - B1 * a)
        C[6] = self.config.DT * (
            ((A4 * delta * v - A5 * beta * v + A6 * 2 * yawrate) / (v * v)) - B2 * a
        )

        return A, B, C

    def get_kinematic_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def mpc_prob_init(self, ref_traj, state_predict, x0, oa, vehicle_params):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        """
        # Initialize and create vectors for the optimization problem
        self.x = cvxpy.Variable(
            (self.config.NX, self.config.T + 1)
        )  # Vehicle State Vector
        self.u = cvxpy.Variable((self.config.NU, self.config.T))  # Control Input vector
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0 = cvxpy.Parameter((self.config.NX,))
        self.x0.value = x0

        # Initialize reference trajectory parameter
        self.ref_traj = cvxpy.Parameter(ref_traj.shape)
        self.ref_traj.value = ref_traj

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.R] * self.config.T))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rd] * (self.config.T - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Q] * (self.config.T)
        Q_block.append(self.config.Qf)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.u), psd_wrap(R_block))

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.x - self.ref_traj), psd_wrap(Q_block))

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.u, axis=1)), psd_wrap(Rd_block))

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.T):
            A, B, C = self.get_dynamic_model_matrix(
                state_predict[2, t],
                state_predict[3, t],
                state_predict[4, t],
                state_predict[5, t],
                state_predict[6, t],
                oa[t],
                vehicle_params,
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz.size))

        # Setting sparse matrix data
        self.Annz.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.A_ = cvxpy.reshape(Indexer @ self.Annz, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz.size))
        self.B_ = cvxpy.reshape(Indexer @ self.Bnnz, (m, n), order="C")
        self.Bnnz.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.C_ = cvxpy.Parameter(C_block.shape)
        self.C_.value = C_block

        # Add dynamics constraints to the optimization problem
        constraints += [
            cvxpy.vec(self.x[:, 1:])
            == self.A_ @ cvxpy.vec(self.x[:, :-1])
            + self.B_ @ cvxpy.vec(self.u)
            + (self.C_)
        ]

        # Constraints 2: Steering Speed in each timestep must be lower than Max Steering Speed
        constraints += [cvxpy.abs(cvxpy.diff(self.u[0, :])) <= self.config.MAX_STEER_V]

        # Constraints 3: Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        constraints += [self.x[:, 0] == self.x0]  #
        constraints += [
            self.x[2, :] <= self.config.MAX_STEER
        ]  # State 2: Steering angle must be lower then Max-Steering
        constraints += [
            self.x[2, :] >= -self.config.MAX_STEER
        ]  # State 2: Steering angle must be higher then Min-Steering
        constraints += [
            self.x[3, :] <= self.config.MAX_SPEED
        ]  # State 3: Velocity must be lower than Max Velocity
        constraints += [
            self.x[3, :] >= self.config.MIN_SPEED
        ]  # State 3: Velocity must be higher than Min Velocity
        constraints += [
            cvxpy.abs(self.u[0, :]) <= self.config.MAX_STEER_V
        ]  # Input 1: Steering Speed must be lower than Max Steering Speed
        constraints += [
            cvxpy.abs(self.u[1, :]) <= self.config.MAX_ACCEL
        ]  # Input 2: Acceleration must be lower than max acceleration

        # CREATE: Define the optimization problem object in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function with given constraints
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def mpc_prob_init_kinematic(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        :return: optimal acceleration and steering strateg
        """
        # Initialize and create vectors for the optimization problem
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )  # Vehicle State Vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )  # Control Input vector
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), psd_wrap(R_block))

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), psd_wrap(Q_block))

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), psd_wrap(Rd_block))

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_kinematic_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block
        # Add dynamics constraints to the optimization problem
        constraints += [
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + (self.Ck_)
        ]

        constraints += [
            cvxpy.abs(cvxpy.diff(self.uk[1, :]))
            <= self.config.MAX_DSTEER * self.config.DTK
        ]

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        constraints += [self.xk[:, 0] == self.x0k]
        constraints += [self.xk[2, :] <= self.config.MAX_SPEED]
        constraints += [self.xk[2, :] >= self.config.MIN_SPEED]
        constraints += [cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL]
        constraints += [cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER]

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob_k = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def mpc_prob_solve(self, ref_traj, state_predict, x0, oa, vehicle_params):
        """
        Solves MPC quadratic optimization problem initialized by mpc_prob_init using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        :return: optimal acceleration and steering strateg
        """

        self.x0.value = x0
        self.ref_traj.value = ref_traj

        # Update vehicle dynamics matrices A,B,C
        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.T):
            A, B, C = self.get_dynamic_model_matrix(
                state_predict[2, t],
                state_predict[3, t],
                state_predict[4, t],
                state_predict[5, t],
                state_predict[6, t],
                oa[t],
                vehicle_params,
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz.value = A_block.data
        self.Bnnz.value = B_block.data
        self.C_.value = C_block

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(
            solver=cvxpy.OSQP,
            verbose=False,
            warm_start=True,
            enforce_dpp=True,
            eps_rel=1e-1,
            eps_abs=1e-1,
        )
        # print(f'optimal value with OSQP: {self.MPC_prob.value} | Took {self.MPC_prob._solve_time} seconds')

        # Save the output of the MPC (States and Input) into specific variables
        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            # MPC States output
            mpc_x = self.get_nparray_from_matrix(
                self.x.value[0, :]
            )  # MPC-State: x-position
            mpc_y = self.get_nparray_from_matrix(
                self.x.value[1, :]
            )  # MPC-State: y-position
            mpc_delta = self.get_nparray_from_matrix(
                self.x.value[2, :]
            )  # MPC-State: Steering Angle
            mpc_v = self.get_nparray_from_matrix(
                self.x.value[3, :]
            )  # MPC-State: Velocity
            mpc_yaw = self.get_nparray_from_matrix(
                self.x.value[4, :]
            )  # MPC-State: Heading
            mpc_yawrate = self.get_nparray_from_matrix(
                self.x.value[5, :]
            )  # MPC-State: yawrate
            mpc_beta = self.get_nparray_from_matrix(
                self.x.value[6, :]
            )  # MPC-State: Side Slip Angle

            # MPC Control output
            mpc_delta_v = self.get_nparray_from_matrix(
                self.u.value[0, :]
            )  # MPC-Control Input: Steering Velocitz
            mpc_a = self.get_nparray_from_matrix(
                self.u.value[1, :]
            )  # MPC-Control Input: Acceleration

        else:
            print("Error: Cannot solve mpc..")
            (
                mpc_a,
                mpc_delta_v,
                mpc_x,
                mpc_y,
                mpc_delta,
                mpc_v,
                mpc_yaw,
                mpc_yawrate,
                mpc_beta,
            ) = (None, None, None, None, None, None, None, None, None)

        return (
            mpc_a,
            mpc_delta_v,
            mpc_x,
            mpc_y,
            mpc_delta,
            mpc_v,
            mpc_yaw,
            mpc_yawrate,
            mpc_beta,
        )

    def mpc_prob_solve_kinematic(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_kinematic_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob_k.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob_k.status == cvxpy.OPTIMAL
            or self.MPC_prob_k.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = self.get_nparray_from_matrix(self.xk.value[0, :])
            oy = self.get_nparray_from_matrix(self.xk.value[1, :])
            ov = self.get_nparray_from_matrix(self.xk.value[2, :])
            oyaw = self.get_nparray_from_matrix(self.xk.value[3, :])
            oa = self.get_nparray_from_matrix(self.uk.value[0, :])
            odelta = self.get_nparray_from_matrix(self.uk.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od_v, vehicle_params):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od_v is None or oa.shape[0] < self.config.T:
            oa = [0.0] * self.config.T
            od_v = [0.0] * self.config.T

        # Call the Motion Prediction function: Predict the vehicle motion/states for T time steps
        state_predict = self.predict_motion(x0, oa, od_v, ref_path, vehicle_params)
        # -------------------- INITIALIZE MPC Problem ----------------------------------------
        if self.init_flag == 0:
            self.mpc_prob_init(ref_path, state_predict, x0, oa, vehicle_params)
            self.init_flag = 1

        # Run the MPC optimization: Create and solve the optimization problem
        (
            mpc_a,
            mpc_delta_v,
            mpc_x,
            mpc_y,
            mpc_delta,
            mpc_v,
            mpc_yaw,
            mpc_yawrate,
            mpc_beta,
        ) = self.mpc_prob_solve(ref_path, state_predict, x0, oa, vehicle_params)

        return (
            mpc_a,
            mpc_delta_v,
            mpc_x,
            mpc_y,
            mpc_delta,
            mpc_v,
            mpc_yaw,
            mpc_yawrate,
            mpc_beta,
            state_predict,
        )

    def linear_mpc_control_kinematic(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od is None or oa.shape[0] > self.config.TK:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion_kinematic(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve_kinematic(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def MPC_Control(self, vehicle_state, path, vehicle_params):
        # Extract information about the trajectory that needs to be followed
        cx = path[0]  # Trajectory x-Position
        cy = path[1]  # Trajectory y-Position
        cyaw = path[2]  # Trajectory Heading angle
        sp = path[3]  # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path = self.calc_ref_trajectory(vehicle_state, cx, cy, cyaw, sp)

        # Create state vector based on current vehicle state: [x, y, delta, v, yaw, yawrate, beta]
        x0 = [
            vehicle_state.x,
            vehicle_state.y,
            vehicle_state.delta,
            vehicle_state.v,
            vehicle_state.yaw,
            vehicle_state.yawrate,
            vehicle_state.beta,
        ]

        # Solve the Linear MPC Control problem and provide output
        # Acceleration, Steering Speed, x-pos, y-pos, steering angle, speed, yaw, yawrate, side slip angle and Predicted sates
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            odelta,
            ov,
            oyaw,
            oyawrate,
            obeta,
            state_predict,
        ) = self.linear_mpc_control(
            ref_path, x0, self.oa, self.odelta_v, vehicle_params
        )

        if self.odelta_v is not None:
            di_v, ai = self.odelta_v[0], self.oa[0]

        # Create the final steer and speed parameter that need to be sent out

        # Steering Output: First entry of the MPC steering speed output vector in rad/s
        # The F1TENTH Gym needs steering angle has a control input: Steering speed  -> Steering Angle
        steer_output = vehicle_state.delta + self.odelta_v[0] * self.config.DT

        # Acceleration Output: First entry of the MPC acceleration output in m/s2
        # The F1TENTH Gym needs velocity has a control input: Acceleration -> Velocity
        # accelerate
        speed_output = vehicle_state.v + self.oa[0] * self.config.DT

        if self.debug:
            plt.cla()
            plt.axis(
                [
                    vehicle_state.x - 6,
                    vehicle_state.x + 4.5,
                    vehicle_state.y - 2.5,
                    vehicle_state.y + 2.5,
                ]
            )
            plt.plot(
                cx,
                cy,
                linestyle="solid",
                linewidth=2,
                color="#005293",
                label="Raceline",
            )
            plt.plot(
                vehicle_state.x,
                vehicle_state.y,
                marker="o",
                markersize=10,
                color="red",
                label="CoG",
            )
            plt.scatter(
                ref_path[0],
                ref_path[1],
                marker="x",
                linewidth=4,
                color="purple",
                label="MPC Input: Ref. Trajectory for T steps",
            )
            plt.scatter(
                ox,
                oy,
                marker="o",
                linewidth=4,
                color="green",
                label="MPC Output: Trajectory for T steps",
            )
            plt.legend()
            plt.pause(0.000001)
            plt.axis("equal")

        return (
            speed_output,
            steer_output,
            ref_path[0],
            ref_path[1],
            state_predict[0],
            state_predict[1],
            ox,
            oy,
        )

    def MPC_Control_kinematic(self, vehicle_state, path):
        # Extract information about the trajectory that needs to be followed
        cx = path[0]  # Trajectory x-Position
        cy = path[1]  # Trajectory y-Position
        cyaw = path[2]  # Trajectory Heading angle
        sp = path[3]  # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path = self.calc_ref_trajectory_kinematic(vehicle_state, cx, cy, cyaw, sp)
        # Create state vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control_kinematic(ref_path, x0, self.oa, self.odelta_v)

        if self.odelta_v is not None:
            di, ai = self.odelta_v[0], self.oa[0]

        # Create the final steer and speed parameter that need to be sent out

        # Steering Output: First entry of the MPC steering angle output vector in degree
        steer_output = self.odelta_v[0]

        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        if self.debug:
            plt.cla()
            plt.axis(
                [
                    vehicle_state.x - 6,
                    vehicle_state.x + 4.5,
                    vehicle_state.y - 2.5,
                    vehicle_state.y + 2.5,
                ]
            )
            plt.plot(
                cx,
                cy,
                linestyle="solid",
                linewidth=2,
                color="#005293",
                label="Raceline",
            )
            plt.plot(
                vehicle_state.x,
                vehicle_state.y,
                marker="o",
                markersize=10,
                color="red",
                label="CoG",
            )
            plt.scatter(
                ref_path[0],
                ref_path[1],
                marker="x",
                linewidth=4,
                color="purple",
                label="MPC Input: Ref. Trajectory for T steps",
            )
            plt.scatter(
                ox,
                oy,
                marker="o",
                linewidth=4,
                color="green",
                label="MPC Output: Trajectory for T steps",
            )
            plt.legend()
            plt.pause(0.000001)
            plt.axis("equal")

        return (
            speed_output,
            steer_output,
            ref_path[0],
            ref_path[1],
            state_predict[0],
            state_predict[1],
            ox,
            oy,
        )
