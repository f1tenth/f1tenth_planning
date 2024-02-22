"""
NMPC waypoint tracker using CasADi. On init, takes in model equation. 
"""
from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.utils.utils import nearest_point


@dataclass
class mpc_config:
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
        track,
        config=mpc_config(),
        debug=False,
    ):
        self.waypoints = [
            track.raceline.xs,
            track.raceline.ys,
            track.raceline.yaws,
            track.raceline.vxs,
        ]
        self.config = config
        self.odelta_v = None
        self.oa = None
        self.odelta = None
        self.ref_path = None
        self.ox = None
        self.oy = None
        self.init_flag = 0
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
        return
    
    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        return
