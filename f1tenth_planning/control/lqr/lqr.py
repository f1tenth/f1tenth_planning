from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import update_matrix
from f1tenth_planning.utils.utils import solve_lqr
from f1tenth_planning.utils.utils import pi_2_pi

from f1tenth_planning.control.controller import Controller
from f1tenth_gym.envs.track import Track

import yaml
import pathlib
import numpy as np
import math

class LQRController(Controller):
    """
    Lateral Controller using LQR

    Parameters
    ----------
        track : Track
            track object with raceline/centerline
        config : dict | str, optional
            dictionary or path to yaml with controller specific parameters, by default None 
            expects the following keys : 
                 "wheelbase": float, wheelbase of the vehicle
                 "matrrix_q": list, weights on the states
                 "matrix_r": list, weights on control input
                 "iterations": int, maximum iteration for solving
                 "eps": float, error tolerance for solving
                 "timestep": float, discretization time step

    Attributes
    ----------
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 5], optional, default=None): waypoints to track, columns are [x, y, velocity, heading, curvature]
        vehicle_control_e_cog (float): lateral error of cog to ref trajectory
        vehicle_control_theta_e (float): yaw error to ref trajectory
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
            - wheelbase: float, wheelbase of the vehicle
            - matrix_q: list, weights on the states
            - matrix_r: list, weights on control input
            - iterations: int, maximum iteration for solving
            - eps: float, error tolerance for solving
            - timestep: float, discretization time step

        Raises
        ------
        ValueError
            if track is None or does not have waypoints (raceline or centerline)
        ValueError
            if config file does not exist
        """
        if track is None or (track.raceline is None and track.centerline is None):
            raise ValueError("Track object with waypoints is required for the controller")
        
        # Extract waypoints from track
        reference = track.raceline if track.raceline is not None else track.centerline
        self.waypoints = np.stack(
                            [reference.xs, reference.ys, reference.vxs, reference.yaws, reference.ks], axis=1
                        )
        
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
        self.timestep = config.get("timestep", 0.01)
        self.wheelbase = config.get("wheelbase", 0.33)
        self.matrix_q = config.get("matrix_q", [0.999, 0.0, 0.0066, 0.0])
        self.matrix_r = config.get("matrix_r", [0.75])
        self.iterations = config.get("iterations", 50)
        self.eps = config.get("eps", 0.001)

        # Initialize control errors
        self.vehicle_control_e_cog = 0  # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0  # theta_e: yaw error to ref trajectory
        self.closest_point = None
        self.target_index = None

    def update(self, config: dict) -> None:
        """Updates setting of controller

        Parameters
        ----------
        config : dict
            configurations to update
        """
        self.wheelbase = config.get("wheelbase", self.wheelbase)
        self.matrix_q = config.get("matrix_q", self.matrix_q)
        self.matrix_r = config.get("matrix_r", self.matrix_r)
        self.iterations = config.get("iterations", self.iterations)
        self.eps = config.get("eps", self.eps)
        self.timestep = config.get("timestep", self.timestep)

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
        
    def render_closest_point(self, e):
        """
        Callback to render the closest point.

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.closest_point is not None:
            points = self.closest_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.target_index is not None:
            points = self.waypoints[self.target_index : self.target_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=2)

    def calc_control_points(self, vehicle_state, waypoints):
        """
        Calculate the heading and cross-track errors and target velocity and curvature

        Parameters
        ----------
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track [x, y, velocity, heading, curvature]

        Returns
        -------
            theta_e (float): heading error
            e_cog (float): lateral crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_veloctiy (float): target velocity
        """

        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        self.closest_point, nearest_dist, t, self.target_index = nearest_point(
            position_front_axle, self.waypoints[:, 0:2]
        )
        vec_dist_nearest_point = position_front_axle - self.closest_point

        # crosstrack error
        front_axle_vec_rot_90 = np.array(
            [
                [math.cos(vehicle_state[2] - math.pi / 2.0)],
                [math.sin(vehicle_state[2] - math.pi / 2.0)],
            ]
        )
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        # heading error
        # NOTE: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[self.target_index, 3]
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # target velocity
        goal_veloctiy = waypoints[self.target_index, 2]

        # reference curvature
        kappa_ref = self.waypoints[self.target_index, 4]

        # saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy

    def controller(
        self, vehicle_state, waypoints, ts, matrix_q, matrix_r, max_iteration, eps
    ):
        """
        Compute lateral control command.

        Parameters
        ----------
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 5]): waypoints to track
            ts (float): discretization time step
            matrix_q ([float], len=4): weights on the states
            matrix_r ([float], len=1): weights on control input
            max_iteration (int): maximum iteration for solving
            eps (float): error tolerance for solving

        Returns
        -------
            steer_angle (float): desired steering angle
            v_ref (float): desired velocity
        """

        # size of controlled states
        state_size = 4

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cg, yaw_ref, k_ref, v_ref = self.calc_control_points(
            vehicle_state, waypoints
        )

        # Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = update_matrix(
            vehicle_state, state_size, ts, self.wheelbase
        )

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = solve_lqr(
            matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration
        )

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cog_old) / ts
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts

        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        # Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = k_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        return steer_angle, v_ref

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
        pose_x = state["pose_x"]
        pose_y = state["pose_y"]
        pose_theta = state["pose_theta"]
        velocity = state["linear_vel_x"]

        # Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        # Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(
            vehicle_state, self.waypoints, self.timestep, self.matrix_q, self.matrix_r, self.iterations, self.eps
        )

        return steering_angle, speed
