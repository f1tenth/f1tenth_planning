# MIT License

# Copyright (c) Ahmad Amine

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
Minimum-time Learning-MPC

Author: Ahmad Amine
Last Modified: 01/01/2024
"""

import math
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import cvxpy
import numpy as np
from f1tenth_planning.utils.utils import nearest_point
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from cvxpy.atoms.affine.wraps import psd_wrap

@dataclass
class lmpc_config:
    NX: int = 4  # length of dynamic state vector: z = [s, ey, epsi, v]
    NU: int = 2  # length of input vector: u = = [steering_angle, accl]
    T: int = 10  # finite time horizon length
    Rk: list = field(
        default_factory=lambda: np.diag([1.0, 10.0])
    )  # input cost matrix, penalty for inputs - [steering_angle, accl]
    Qk: list = field(
        default_factory=lambda:  np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) 
    )  # state error cost matrix, for the the next (T) prediction time steps [s, ey, epsi, v]
    N_IND_SEARCH: int = 20  # Search index number
    DT: float = 0.1  # time step [s]
    dl: float = 0.03  # dist step [m]
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MIN_STEER: float = -0.4189  # minimum steering angle [rad]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum absolute acceleration [m/ss]
    TRACK_WIDTH: float = 0.31  # Width of the track [m]
    num_SS_it: int = 4  # Number of trajectories used at each iteration to build the safe set
    numSS_Points_Per_it = 12    # Number of points to select from each trajectory to build the safe set


@dataclass
class Kinematic_State_Frenet:
    s : float = 0.0
    ey : float = 0.0
    epsi : float = 0.0
    v : float = 0.0


class LMPCPlanner:
    """
    Learning Single Track MPC Controller, uses the Frenet ST model from 

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (Track): F1tenth track object to plan on
        config (lmpc_config): configuration for the planner, includes weights and model parameters
    """

    def __init__(
        self,
        track,
        config=lmpc_config(),
    ):
        self.track = track
        self.raceline = track.raceline # used for cartesian to frenet
        self.centerline = track.centerline # used for curvature, cartesian to frenet, etc
        self.waypoints = [track.raceline.ss,                 # s_ref
                          np.zeros_like(track.raceline.vxs), # ey_ref
                          np.zeros_like(track.raceline.vxs), # epsi_ref
                          track.raceline.vxs]                # vx_ref 

        self.zt = None

        self.SS_trajectories = []  # iterations, length, states
        self.uSS_trajectories = [] # iterations, length, inputs
        self.vSS_trajectories = [] # iterations, length, 1 (values)

        self.config : lmpc_config = config
        self.odelta = None
        self.oa = None
        self.ref_path = None
        self.ox = None
        self.oy = None
        self.lmpc_prob_initialized = False
        self.lmpc_prob_init()
        self.old_s = 0.0

    def plan(self, state):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            state (dict): current state of the vehicle
        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle

        """
        if state["linear_vel_x"] < 0.1:
            return 0.0, self.config.MAX_ACCEL

        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan()"
            )
        curr_state = np.array([
                                [state["pose_x"],
                                state["pose_y"],
                                state["delta"],
                                state["linear_vel_x"],
                                state["pose_theta"],
                                state["ang_vel_z"],
                                state["beta"]]
                              ])        
        frenet_kinematic_pose = self.track.cartesian_to_frenet(curr_state[0, 0], curr_state[0, 1], curr_state[0, 4], s_guess=self.old_s)
        vehicle_state = Kinematic_State_Frenet(
            s=frenet_kinematic_pose[0], # curvilinear abscissa
            ey=frenet_kinematic_pose[1], # lateral error
            epsi=frenet_kinematic_pose[2], # heading error
            v=state["linear_vel_x"]
        )

        (
            accl,
            steering_vel,
            mpc_pred_s,
            mpc_pred_ey,
            mpc_os,
            mpc_oey,
        ) = self.LMPC_Control(vehicle_state, state["delta"])

        return steering_vel, accl

    def update_zt(self, mpc_solution):
        """
        Update the reference target point based on the last mpc solution
        """
        self.zt = mpc_solution[:, -1]

    def select_safe_subset(self, numSS_Points_Per_it, num_SS_it, zt):
        """
        Select the safe subset from the safe set
        """
        xSS_selected = []
        vSS_selected = []
        uSS_selected = []

        # select from last num_SS_it trajectories
        for iteration in range(len(self.SS_trajectories)-num_SS_it, len(self.SS_trajectories)):
            # select the closes numSS_Points to zt
            curr_trajectory = self.SS_trajectories[iteration]
            curr_values = self.vSS_trajectories[iteration][0] # converst it to flat array
            curr_control = self.uSS_trajectories[iteration]

            curr_distances = np.abs(curr_trajectory[:,0] - zt[0]) # Only compare distances in curvilinear abscissa s, faster than calculating the whole distance
            curr_selected_indices = np.argpartition(curr_distances, numSS_Points_Per_it)[:numSS_Points_Per_it] # argpartition is faster than argsort

            traj_selected_states = curr_trajectory[curr_selected_indices]
            traj_selected_values = curr_values[curr_selected_indices]
            traj_selected_control = curr_control[curr_selected_indices]

            xSS_selected.append(traj_selected_states)
            vSS_selected.append(traj_selected_values)
            uSS_selected.append(traj_selected_control)

        # Convert to numpy arrays
        xSS_selected = np.concatenate(xSS_selected)
        vSS_selected = np.concatenate(vSS_selected).reshape(-1,1)
        uSS_selected = np.concatenate(uSS_selected)
        return xSS_selected, vSS_selected, uSS_selected
    
    def update_safe_set(self, xSS_new, uSS_new, vSS_new):
        """
        Update the safe set with the new trajectories
        """
        self.SS_trajectories.append(xSS_new)
        self.uSS_trajectories.append(uSS_new)
        self.vSS_trajectories.append(vSS_new)
        return

    def get_dynamic_model_matrix(self, current_state, current_control_input):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[s, ey, epsi, v]
        :param: current_state: current state of the vehicle [s, ey, epsi, v]
        :return: A, B, C matrices
        """
        # Get control input
        a = current_control_input[0]
        delta = current_control_input[1]

        # linearized dynamic model of the vehicle
        A = np.zeros((self.config.NX, self.config.NX))
        B = np.zeros((self.config.NX, self.config.NU))
        C = np.zeros((self.config.NX,))

        s, ey, epsi, v = current_state

        curvature = self.centerline.spline.calc_curvature(s)

        # ∇_x(s); s = vcos(epsi)/(1 - ey*curvature)
        A[0, 1] = curvature * (v * np.cos(epsi)) / (1 - ey * curvature)**2 # ds/dey
        A[0, 2] = - v * np.sin(epsi) / (1 - ey * curvature)                # ds/depsi
        A[0, 3] = np.cos(epsi) / (1 - ey * curvature)                                    # ds/dv
        
        # ∇_x(ey); ey = vsin(epsi)
        A[1, 0] = 0.0                                                                                                 # dey/ds
        A[1, 2] = v * np.cos(epsi)                                                        # dey/depsi
        A[1, 3] = np.sin(epsi)                                                                          # dey/dv

        # ∇_x(epsi); epsi = v*tan(delta)/WB - curvature*(vcos(epsi)/(1 - ey*curvature))
        A[2, 0] = 0.0                                                                                                 # depsi/ds
        A[2, 1] = - v * np.cos(epsi) / (1 - ey * curvature)                 # depsi/dey
        A[2, 3] = np.tan(current_control_input[1]) / self.config.WB                                                   # depsi/dv

        # ∇_x(v)
        A[3, 0] = 0.0                                                                                                 # dv/ds
        A[3, 1] = 0.0                                                                                                 # dv/dey
        A[3, 2] = 0.0                                                                                                 # dv/depsi

        # Forward Euler Discretization, maybe replace with exact discretization ?
        A = np.eye(self.config.NX) + A * self.config.DT

        # ∇_u(s); s = vcos(epsi)/(1 - ey*curvature)
        B[0, 0] = 0.0                                                                                                 # ds/da
        B[0, 1] = 0.0                                                                                                 # ds/ddelta

        # ∇_u(ey); ey = vsin(epsi)
        B[1, 0] = 0.0                                                                                                 # dey/da
        B[1, 1] = 0.0                                                                                                 # dey/ddelta

        # ∇_u(epsi); epsi = v*tan(delta)/WB - curvature*(vcos(epsi)/(1 - ey*curvature))
        B[2, 0] = 0.0                                                                                                 # depsi/da
        B[2, 1] = v / (self.config.WB * np.cos(current_control_input[1])**2)                            # depsi/ddelta

        # ∇_u(v)
        B[3, 0] = 1.0                                                                                                 # dv/da
        B[3, 1] = 0.0                                                                                                 # dv/ddelta

        # Forward Euler Discretization
        B = B * self.config.DT

        # ∇_x(epsi); epsi = v*tan(delta)/WB - curvature*(vcos(epsi)/(1 - ey*curvature))
        A[2, 0] = 0.0                                                                                                 # depsi/ds
        A[2, 1] = - v * np.cos(epsi) / (1 - ey * curvature)                 # depsi/dey
        A[2, 3] = np.tan(current_control_input[1]) / self.config.WB                                                   # depsi/dv

        # C = A*x + B*u - (x + f(x,u))*dt)
        C[0] = -(ey * (curvature * (v * np.cos(epsi)) / (1 - ey * curvature)**2) + epsi * (- v * np.sin(epsi) / (1 - ey * curvature))) * self.config.DT
        C[1] = -(epsi * (v * np.cos(epsi))) * self.config.DT
        C[2] = -(delta * v / (self.config.WB * np.cos(current_control_input[1])**2) - ey * v * np.cos(epsi) / (1 - ey * curvature) + v * np.tan(current_control_input[1]) / self.config.WB)* self.config.DT
        C[3] = 0.0

        return A, B, C

    def predict_motion(self, x0, oa, od):

        # Create Vector that includes the predicted path for the next T time steps for all vehicle states
        path_predict = np.zeros((self.config.NX, self.config.T + 1))
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        # Calculate/Predict the vehicle states/motion for the next T time steps
        state : Kinematic_State_Frenet = Kinematic_State_Frenet(
             s=x0[0], ey=x0[1], epsi=x0[2], v=x0[3],
        )
        for (ai, delta_i, i) in zip(oa, od, range(1, self.config.T + 1)):
            state = self.update_state(state, ai, delta_i)
            path_predict[0, i] = state.s
            path_predict[1, i] = state.ey
            path_predict[2, i] = state.epsi
            path_predict[3, i] = state.v

        return path_predict

    def update_state(self, state : Kinematic_State_Frenet, a, delta):
        """
        Uses the kinematic bicycle model in frenet frame to update the state of the vehicle
        Dynamics reference: https://arxiv.org/pdf/2005.07691.pdf
        :param state: current state of the vehicle [s, ey, epsi, v]
        :param a: acceleration
        :param delta: steering angle
        """
        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= self.config.MIN_STEER:
            delta = self.config.MIN_STEER

        curvature = self.centerline.spline.calc_curvature(state.s)

        state.s = state.s + ((state.v * np.cos(state.epsi)) / (1 - state.ey * curvature)) * self.config.DT
        state.ey = state.ey + (state.v * np.sin(state.epsi)) * self.config.DT
        state.epsi = state.epsi + (state.v * np.tan(delta) / self.config.WB - curvature * ((state.v * np.cos(state.epsi))/(1 - curvature * state.ey))) * self.config.DT
        state.v = state.v + a * self.config.DT

        return state

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def lmpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        :param xSS: Safe Set States (self.config.numSS_Points_Per_it * self.config.num_SS_it, NX)
        :param vSS: Safe Set Values (self.config.numSS_Points_Per_it * self.config.num_SS_it, 1)
        :param state_predict: predicted states in T steps
        """
        # Initialize and create vectors for the optimization problem
        self.x = cvxpy.Variable(
            (self.config.NX, self.config.T + 1)
        )  # Vehicle State Vector
        self.u = cvxpy.Variable((self.config.NU, self.config.T))  # Control Input vector
        self.lambda_f = cvxpy.Variable((self.config.numSS_Points_Per_it * self.config.num_SS_it, 1)) # Lambda vector to calculate terminal cost
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0 = cvxpy.Parameter((self.config.NX,))
        self.x0.value = np.zeros((self.config.NX,))

        # Initialize safe set trajectories parameter
        self.safe_set_states = cvxpy.Parameter((self.config.numSS_Points_Per_it * self.config.num_SS_it, self.config.NX))
        self.safe_set_states.value = np.zeros((self.config.numSS_Points_Per_it * self.config.num_SS_it, self.config.NX))

        # Initialize safe set trajectories parameter
        self.safe_set_values = cvxpy.Parameter((self.config.numSS_Points_Per_it * self.config.num_SS_it, 1))
        self.safe_set_values.value = np.zeros((self.config.numSS_Points_Per_it * self.config.num_SS_it, 1))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Running Cost sum(1) over T steps
        objective += self.config.T

        # Objective 2: Terminal Cost derived from the terminal state lambda_f.T * vSS
        objective += self.lambda_f.T @ self.safe_set_values

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        state_predict = np.zeros((self.config.NX, self.config.T + 1))
        for t in range(self.config.T):
            current_state = np.array([
                state_predict[0, t],
                state_predict[1, t],
                state_predict[2, t],
                state_predict[3, t]]
            )
            A, B, C = self.get_dynamic_model_matrix(
                current_state,
                np.zeros((self.config.NU,)),
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

        # Constraints 2: Initial state constraint x0 = x[0]
        constraints += [self.x[:, 0] == self.x0]  # [s, ey, epsi, v]

        # Constraint 3: Terminal state constraint (lambda_f.T * xSS) == x[N]
        constraints += [cvxpy.vec(self.lambda_f.T @ self.safe_set_states) == self.x[:, -1]]

        # Constraints 5: State and Input Constraints
        constraints += [
            self.x[1, :] <= self.config.TRACK_WIDTH/2
        ]  # State 1: Signed Lateral Error must be lower than half of the track width
        constraints += [
            self.x[1, :] >= -self.config.TRACK_WIDTH/2
        ]  # State 1: Signed Lateral Error must be higher than negative half of the track width
        constraints += [
            self.x[3, :] <= self.config.MAX_SPEED
        ]  # State 3: Velocity must be lower than Max Velocity
        constraints += [
            self.x[3, :] >= self.config.MIN_SPEED
        ]  # State 3: Velocity must be higher than Min Velocity
        constraints += [
            cvxpy.abs(self.u[0, :]) <= self.config.MAX_ACCEL
        ]  # Input 1: Acceleration must be lower than max acceleration
        constraints += [
            cvxpy.abs(self.u[1, :]) <= self.config.MAX_STEER
        ]  # Input 2: Steering Speed must be lower than Max Steering Speed

        # CREATE: Define the optimization problem object in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function with given constraints
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def lmpc_prob_solve(self, xSS, vSS, state_predict, x0, oa, odelta):
        """
        Solves MPC quadratic optimization problem initialized by mpc_prob_init using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [s, ey, epsi, v])
        path_predict: predicted states in T steps
        x0: initial state
        :return: optimal acceleration and steering strateg
        """
        self.safe_set_states.value = xSS
        self.safe_set_values.value = vSS
        self.x0.value = x0

        # Update vehicle dynamics matrices A,B,C
        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.T):
            A, B, C = self.get_dynamic_model_matrix(
                state_predict[:, t],
                np.array([oa[t], odelta[t]]),
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
            # MPC States output z = [s, ey, epsi, v]
            mpc_s = self.get_nparray_from_matrix(
                self.x.value[0, :]
            )  # MPC-State: curvilinear abscissa
            mpc_ey = self.get_nparray_from_matrix(
                self.x.value[1, :]
            )  # MPC-State: lateral error
            mpc_epsi = self.get_nparray_from_matrix(
                self.x.value[2, :]
            )  # MPC-State: heading error
            mpc_vx = self.get_nparray_from_matrix(
                self.x.value[3, :]
            )  # MPC-State: longitudinal velocity
            
            # MPC Control output
            mpc_delta = self.get_nparray_from_matrix(
                self.u.value[0, :]
            )  # MPC-Control Input: Steering Angle
            mpc_a = self.get_nparray_from_matrix(
                self.u.value[1, :]
            )  # MPC-Control Input: Acceleration

            self.update_zt(self.x.value) 
        else:
            print("Error: Cannot solve mpc..")
            (
                mpc_s,
                mpc_ey,
                mpc_epsi,
                mpc_vx,
                mpc_delta,
                mpc_a,
            ) = (None, None, None, None, None, None, None, None)

        return (
            mpc_s,
            mpc_ey,
            mpc_epsi,
            mpc_vx,
            mpc_delta,
            mpc_a,
        )

    def linear_lmpc_control(self, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od is None or oa.shape[0] < self.config.T:
            oa = [0.0] * self.config.T
            od = [0.0] * self.config.T

        # Call the Motion Prediction function: Predict the vehicle motion/states for T time steps
        state_predict = self.predict_motion(x0, oa, od)

        # Get xSS subset and vSS subset from the safe set
        xSS, vSS, uSS = self.select_safe_subset(self.config.numSS_Points_Per_it, self.config.num_SS_it, self.zt)

        # -------------------- INITIALIZE MPC Problem ----------------------------------------
        if self.lmpc_prob_initialized == False:
            self.lmpc_prob_init()
            self.lmpc_prob_initialized = True

        # Run the MPC optimization: Create and solve the optimization problem
        (
            mpc_s,
            mpc_ey,
            mpc_epsi,
            mpc_vx,
            mpc_delta,
            mpc_a,
        ) = self.lmpc_prob_solve(xSS, vSS, state_predict, x0, oa, od)

        return (
            mpc_s,
            mpc_ey,
            mpc_epsi,
            mpc_vx,
            mpc_delta,
            mpc_a,
            state_predict,
        )

    def LMPC_Control(self, vehicle_state : Kinematic_State_Frenet, curr_delta):
        
        # Create state vector based on current vehicle state: [s, ey, epsi, v]
        x0 = [
            vehicle_state.s,
            vehicle_state.ey,
            vehicle_state.epsi,
            vehicle_state.v
        ]

        # Solve the Linear MPC Control problem and provide output
        # Acceleration, Steering Speed, s-coordinate, lateral error, heading error, longitudinal velocity
        (
            os,
            oey,
            oepsi,
            ovx,
            self.odelta,
            self.oa,
            state_predict,
        ) = self.linear_lmpc_control(
            x0, self.oa, self.odelta
        )

        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]

        # Steering Output: First entry of the MPC steering speed output vector in rad/s
        # The F1TENTH Gym needs steering angle has a control input: Steering speed  -> Steering Angle
        # Steering velocity is computed as (desired_steering_angle - current_steering_angle) / DT
        svel_output = (di - curr_delta) / self.config.DT
        accl_output = self.oa[0]

        return (
            accl_output,
            svel_output,
            state_predict[0],
            state_predict[1],
            os,
            oey,
        )
