from f1tenth_planning.utils.utils import nearest_point

from f1tenth_planning.control.controller import Controller
from f1tenth_gym.envs.track import Track

import yaml
import math
import cvxpy
import pathlib
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix

@dataclass
class mpc_config:
    """
    MPC Configurations

    Attributes
    ----------
        NXK : int
            length of kinematic state vector: z = [x, y, v, yaw]
        NU : int
            length of input vector: u = = [steering speed, acceleration]
        TK : int
            finite time horizon length for the kinematic MPC problem
        Rk : list
            input cost matrix, penalty for inputs - [accel, steering_speed]
        Rdk : list
            input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
        Qk : list
            state error cost matrix, for the the next (T) prediction time steps - [x, y, v, yaw]
        Qfk : list
            final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
        DTK : float
            discretization time step [s]
        dlk : float
            distance between waypoints [m]
        WB : float
            Wheelbase [m]
        MIN_STEER : float
            minimum steering angle [rad]
        MAX_STEER : float
            maximum steering angle [rad]
        MAX_DSTEER : float
            maximum steering speed [rad/s]
        MIN_DSTEER : float
            minimum steering speed [rad/s]
        MAX_SPEED : float
            maximum speed [m/s]
        MIN_SPEED : float
            minimum speed [m/s]
        MAX_ACCEL : float
            maximum acceleration [m/s^2]
        MIN_ACCEL : float
            minimum acceleration [m/s^2]
    """
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
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
    
    DTK: float = 0.1
    dlk: float = 0.03
    WB: float = 0.33
    MIN_STEER: float = -0.4189
    MAX_STEER: float = 0.4189
    MAX_DSTEER: float = np.deg2rad(180.0)
    MIN_DSTEER: float = -np.deg2rad(180.0)
    MAX_SPEED: float = 6.0
    MIN_SPEED: float = 0.0
    MAX_ACCEL: float = 3.0
    MIN_ACCEL: float = -3.0


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class KMPCController(Controller):
    """
    Dynamic Single Track MPC Controller, uses the ST model from Common Road

    All vehicle pose used by the controller should be in the map frame.

    Attributes
    ----------
        waypoints : list
            list of waypoints [x, y, yaw, velocity]
        config : mpc_config
            mpc_config object with controller specific parameters, check mpc_config class for details
        odelta_v : float
            last solved for steering velocity
        oa : float
            last solved for acceleration
        odelta : float
            last solved for steering angle
        ref_path : numpy.ndarray
            reference path
        ox : numpy.ndarray
            last solved for optimal x-position states
        oy : numpy.ndarray
    """

    def __init__(self, track: Track, config: dict | str | pathlib.Path = None) -> None:
        """Controller init

        Parameters
        ----------
        track : Track
            track object with raceline/centerline
        config : dict | str, optional
            dictionary or path to yaml with controller specific parameters, by default None
            expects the following key:
            - mpc_config: mpc_config object, check mpc_config class for details

        Raises
        ------
        ValueError
            if track is None or does not have waypoints (raceline or centerline)
        ValueError
            if config file does not exist
        """
        if track is None or (track.raceline is None and track.centerline is None):
            raise ValueError("Track object with waypoints is required for the controller")
        
        reference = track.raceline if track.raceline is not None else track.centerline
        self.waypoints = np.array([
            reference.xs,
            reference.ys,
            reference.yaws,
            reference.vxs,
        ]).T

        if config is not None:
            if isinstance(config, (str, pathlib.Path)):
                if isinstance(config, str):
                    config = pathlib.Path(config)
                if not config.exists():
                    raise ValueError(f"Config file {config} does not exist")
                config = self.load_config(config)
        else:
            config = {}
            
        # Setting controller parameters
        self.config = mpc_config(**config.get("mpc_config", {}))

        self.config.dlk = (
            reference.ss[1] - reference.ss[0]
        )  # waypoint spacing

        # Initialize control variables
        self.odelta_v = None
        self.oa = None

        # Initialize visualization variables
        self.ref_path = None
        self.ox = None
        self.oy = None

        # Initialize MPC problem
        self.init_flag = 0
        self.mpc_prob_init_kinematic()

    def update(self, config: dict) -> None:
        """Updates setting of controller

        Parameters
        ----------
        config : dict
            configurations to update
        """
        self.config = mpc_config(**config.get("mpc_config", {}))
       
    def load_config(self, path: str | pathlib.Path) -> dict:
        """Load configuration from yaml file

        Parameters
        ----------
        path : str | pathlib.Path
            path to yaml file

        Returns
        -------
        dict
            configuration dictionary

        Raises
        ------
        ValueError
            if path does not exist
        """
        if type(path) == str:
            path = pathlib.Path(path)
        if not path.exists():
            raise ValueError(f"Config file {path} does not exist")
        with open(path, "r") as f:
            return yaml.safe_load(f)
        
    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.ref_path is not None:
            points = self.ref_path[:2].T
            e.render_lines(points, color=(0, 128, 0), size=2)

    def render_mpc_sol(self, e):
        """
        Callback to render the lookahead point.

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.ox is not None and self.oy is not None:
            e.render_lines(np.array([self.ox, self.oy]).T, color=(0, 0, 128), size=2)

    def plan(self, state: dict) -> np.ndarray:
        """
        Calculate the desired steering angle and speed based on the current vehicle state

        Parameters
        ----------
        state : dict
            observation as returned from the environment.

        Returns
        -------
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle

        Raises
        ------
        ValueError
            if waypoints are not set
        """
        if self.waypoints is None:
            raise ValueError(
                "Please set waypoints to track during planner instantiation or when calling plan()"
            )

        if state["linear_vel_x"] < 0.1:
            steer, accl = 0.0, self.config.MAX_ACCEL
            return steer, accl

        vehicle_state = State(
            x=state["pose_x"],
            y=state["pose_y"],
            delta=state["delta"],
            v=state["linear_vel_x"],
            yaw=state["pose_theta"],
            yawrate=state["ang_vel_z"],
            beta=state["beta"],
        )

        (
            accl,
            svel,
            mpc_ref_path_x,
            mpc_ref_path_y,
            mpc_pred_x,
            mpc_pred_y,
            mpc_ox,
            mpc_oy,
        ) = self.MPC_Control_kinematic(vehicle_state, self.waypoints)

        return svel, accl

    def calc_ref_trajectory_kinematic(self, state, cx, cy, cyaw, sp):
        """
        Calculate reference trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calculate the T points along the reference path

        Parameters
        ----------
            state (State): current vehicle state
            cx (numpy.ndarray): course x-position
            cy (numpy.ndarray): course y-position
            cyaw (numpy.ndarray): course heading
            sp (numpy.ndarray): speed profile

        Returns
        -------
            ref_traj (numpy.ndarray): reference trajectory [x, y, v, yaw]
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
        ref_yaw = cyaw[ind_list].copy() # Avoid manipulating the original array
        ref_yaw[ref_yaw - state.yaw > 4.5] = np.abs(
            ref_yaw[ref_yaw - state.yaw > 4.5] - (2 * np.pi)
        )
        ref_yaw[ref_yaw - state.yaw < -4.5] = np.abs(
            ref_yaw[ref_yaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = ref_yaw

        return ref_traj

    def predict_motion_kinematic(self, x0, oa, od, xref):
        """
        Predict the vehicle motion for T steps using the kinematic model
        
        Parameters
        ----------
            x0 (numpy.ndarray): initial state vector
            oa (numpy.ndarray): acceleration strategy
            od (numpy.ndarray): steering strategy
            xref (numpy.ndarray): reference trajectory

        Returns
        -------
            path_predict (numpy.ndarray): predicted states in T steps
        """

        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state_kinematic(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state_kinematic(self, state, a, delta):
        """
        Update the vehicle state using the kinematic model

        Parameters
        ----------
            state (State): current vehicle state
            a (float): acceleration
            delta (float): steering angle

        Returns
        -------
            state (State): updated vehicle state
        """
        
        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_kinematic_model_matrix(self, v, phi, delta):
        """
        Calculate the linear and discrete time dynamic model for the kinematic model.

        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]

        Parameters:
        v (float): speed
        phi (float): heading angle of the vehicle
        delta (float): steering angle

        Returns:
        A (numpy.ndarray): state matrix A
        B (numpy.ndarray): input matrix B
        C (numpy.ndarray): constant matrix C
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
        """
        Convert matrix to numpy array
        
        Parameters
        ----------
            x (cvxpy.Variable): matrix

        Returns
        -------
            np.array: numpy array
        """
        return np.array(x).flatten()

    def mpc_prob_init_kinematic(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP

        Parameters:
        ----------
        xref : numpy.ndarray
            Reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict : numpy.ndarray
            Predicted states in T steps
        x0 : numpy.ndarray
            Initial state
        dref : float
            Reference steer angle
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
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

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
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def mpc_prob_solve_kinematic(self, ref_traj, path_predict, x0):
        """
        Solve the MPC optimization problem using cvxpy

        Parameters
        ----------
            ref_traj (numpy.ndarray): reference trajectory in T steps
            path_predict (numpy.ndarray): predicted states in T steps
            x0 (numpy.ndarray): initial state vector

        Returns
        -------
            oa (numpy.ndarray): acceleration strategy
            od (numpy.ndarray): steering strategy
            ox (numpy.ndarray): x-position strategy
            oy (numpy.ndarray): y-position strategy
            oyaw (numpy.ndarray): yaw strategy
            ov (numpy.ndarray): velocity strategy
        """

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
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
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

    def linear_mpc_control_kinematic(self, ref_path, x0, oa, od):
        """
        Solve the MPC optimization problem using cvxpy

        Parameters
        ----------
        ref_traj : numpy.ndarray
            Reference trajectory in T steps
        path_predict : numpy.ndarray
            Predicted states in T steps
        x0 : numpy.ndarray
            Initial state vector

        Returns
        -------
        oa : numpy.ndarray
            Acceleration strategy
        od : numpy.ndarray
            Steering strategy
        ox : numpy.ndarray
            X-position strategy
        oy : numpy.ndarray
            Y-position strategy
        oyaw : numpy.ndarray
            Yaw strategy
        ov : numpy.ndarray
            Velocity strategy
        """

        if oa is None or od is None:
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

    def MPC_Control_kinematic(self, vehicle_state, path):
        """
        MPC Control function for the Kinematic Model

        Parameters
        ----------
            vehicle_state : State
                Current vehicle state
            path : numpy.ndarray
                Reference trajectory in T steps

        Returns
        -------
            accl_output : float
                Acceleration output
            sv_output : float
                Steering velocity output
            ref_x : float
                Reference x-position
            ref_y : float
                Reference y-position
            pred_x : float
                Predicted x-position
            pred_y : float
                Predicted y-position
            ox : numpy.ndarray
                X-position strategy
            oy : numpy.ndarray
                Y-position strategy
        """
        # Extract information about the trajectory that needs to be followed
        cx   = path[:, 0]  # Trajectory x-Position
        cy   = path[:, 1]  # Trajectory y-Position
        cyaw = path[:, 2]  # Trajectory Heading angle
        sp   = path[:, 3]  # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        self.ref_path = self.calc_ref_trajectory_kinematic(
            vehicle_state, cx, cy, cyaw, sp
        )
        # Create state vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        (
            self.oa,
            self.odelta_v,
            self.ox,
            self.oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control_kinematic(self.ref_path, x0, self.oa, self.odelta_v)

        if self.odelta_v is not None:
            di, ai = self.odelta_v[0], self.oa[0]

        accl_output = self.oa[0]
        sv_output = (self.odelta_v[0] - vehicle_state.delta) / self.config.DTK

        return (
            accl_output,
            sv_output,
            self.ref_path[0],
            self.ref_path[1],
            state_predict[0],
            state_predict[1],
            self.ox,
            self.oy,
        )
