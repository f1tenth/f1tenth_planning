
from .jax_mppi.jax_mppi.mppi import MPPI
from dataclasses import dataclass, field
import jax
from functools import partial
import jax.numpy as jnp
import numpy as np

@dataclass
class mppi_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, delta, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic
    R: list = field(
        default_factory=lambda: jnp.diag(np.array([0.01, 100.0]))
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rd: list = field(
        default_factory=lambda: jnp.diag(np.array([0.01, 100.0]))
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Q: list = field(
        default_factory=lambda: jnp.diag(np.array([18.5, 18.5, 0.0, 3.5, 0.1]))
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw]
    Qf: list = field(
        default_factory=lambda: jnp.diag(np.array([18.5, 18.5, 0.0, 3.5, 0.1]))
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = jnp.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]

    # MPPI Specific Parameters
    n_iterations: int = 1 # Number of MPPI iterations
    n_samples: int = 16 # Number of sampled trajectories
    temperature: float = 0.01 # Temperature parameter
    damping: float = 0.001 # Damping parameter
    a_noise: float = 0.1 # Noise in the action
    adaptive_covariance: bool = True # Adaptive covariance


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

# Create env class with .step and .reward methods
class KinemaitcMPPIEnv:
    def __init__(self, lwb=0.3302, time_step=0.05):
        self.a_shape = 2 # acceleration and steering speed
        self.lwb = lwb # calculated from F110Env.default_config()["params"] lf + lr
        self.time_step = time_step

    def vehicle_dynamics_ks(self, x, u):
        """
        Single Track Kinematic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (5, 1)): vehicle state vector (x1, x2, x3, x4, x5)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # system dynamics
        f = jnp.array([[x[3, 0]*jnp.cos(x[4, 0]),
            x[3, 0]*jnp.sin(x[4, 0]), 
            u[0],
            u[1],
            x[3, 0]/self.lwb*jnp.tan(x[2, 0])]]).T
        return f
    
    def update_fn(self, x, u):
        dt = self.time_step
        # RK45
        k1 = self.vehicle_dynamics_ks(x, u)
        k2 = self.vehicle_dynamics_ks(x + k1 * 0.5 * dt, u)
        k3 = self.vehicle_dynamics_ks(x + k2 * 0.5 * dt, u)
        k4 = self.vehicle_dynamics_ks(x + k3 * dt, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self.update_fn(state, action)

class KMPPIPlanner:
    def __init__(self, track, config=mppi_config(), debug=False):
        self.track = track
        self.debug = debug
        self.config = config
        self.mppi_env = KinemaitcMPPIEnv()
        self.mppi_env.reward = self.reward # Overwrite / add reward function to the environment
        self.mppi = MPPI(n_iterations=self.config.n_iterations, 
                         n_steps=self.config.TK,
                         n_samples=self.config.n_samples,
                         temperature=self.config.temperature,
                         damping=self.config.damping,
                         a_noise=self.config.a_noise,
                         adaptive_covariance=self.config.adaptive_covariance)

        # initialize the state
        self.rng = jax.random.key(0)
        self.control_state = self.mppi.init_state(self.mppi_env.a_shape, self.rng)
        
        # For visualization
        self.sampled_trajectories = None
        self.optimal_trajectory = None

    @jax.jit
    def _nearest_point(point, trajectory):
        """
        Return the nearest point along the given piecewise linear trajectory.

        Args:
            point (numpy.ndarray, (2, )): (x, y) of current pose
            trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
                NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

        Returns:
            nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
            nearest_dist (float): distance to the nearest point
            t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
            i (int): index of nearest point in the array of trajectory waypoints
        """
        diffs = trajectory[1:,:] - trajectory[:-1,:]
        l2s   = diffs[:,0]**2 + diffs[:,1]**2
        # this is equivalent to the elementwise dot product
        dots = jnp.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
        t = jnp.clip(dots / l2s, 0.0, 1.0)
        projections = trajectory[:-1,:] + (t*diffs.T).T
        dists = jnp.linalg.norm(point - projections,axis=1)
        min_dist_segment = jnp.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    @partial(jax.jit, static_argnums=(0,))
    def get_reference_trajectory(self, state):
        cx, cy, sp, cyaw = self.track.raceline.xs, self.track.raceline.ys, self.track.raceline.yaws, self.track.raceline.vxs

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        self.ref_path = self.calc_ref_trajectory_kinematic(
            state, cx, cy, cyaw, sp
        )

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = jnp.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = KMPPIPlanner._nearest_point(jnp.array([state.x, state.y]), jnp.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + jnp.insert(
            jnp.cumsum(jnp.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = jnp.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * jnp.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = jnp.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * jnp.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    
    @partial(jax.jit, static_argnums=(0,))
    def reward(self, state, action, ref_state, terminal_state=0):
        return ((1- terminal_state) * (-jnp.dot((state - ref_state).T, jnp.dot(self.config.Q, (state - ref_state))) -
                   (terminal_state)  * (-jnp.dot((state - ref_state).T, jnp.dot(self.config.Qf, (state - ref_state)))) - 
                                      jnp.dot(action.T, jnp.dot(self.config.R, action)))[0,0])[0]

    def plan(self, state):
        state = jnp.array([state["pose_x"], state["pose_y"], state["delta"], state["linear_vel_x"], state["pose_theta"]]).reshape(-1, 1)

        # do work to identify a good action
        self.control_state, self.sampled_trajectories = self.mppi.update(self.control_state, self.mppi_env, state, self.rng)

        # get the optimal action
        self.optimal_trajectory, _, uOpt = self.mppi.get_mppi_output(self.mppi_env, state, self.control_state)

        steerv, accl = uOpt[0, 0], uOpt[1, 0]

        return steerv, accl
    
    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.debug:
            points = np.array([self.track.raceline.xs, self.track.raceline.ys]).T
            e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_sampled_trajectories(self, e):
        if self.debug:
            if self.sampled_trajectories is not None:
                for traj in self.sampled_trajectories[-1]:
                    traj_xy = np.array([traj[:, 0, 0], traj[:, 1, 0]]).T
                    e.render_closed_lines(traj_xy, color=(128, 0, 0), size=1)

    def render_optimal_trajectory(self, e):
        if self.debug:
            if self.optimal_trajectory is not None:
                traj_xy = np.array([self.optimal_trajectory[:, 0, 0], self.optimal_trajectory[:, 1, 0]]).T
                e.render_closed_lines(traj_xy, color=(0, 128, 0), size=1)