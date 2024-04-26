from f1tenth_planning.control.controller import Controller
from f1tenth_gym.envs.track import Track

from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import pi_2_pi

import yaml
import pathlib
import numpy as np
import math

class StanleyPlanner(Controller):
    """
    This is the class for the Front Weeel Feedback Controller (Stanley) for tracking the path of the vehicle
    References:
    - Stanley: The robot that won the DARPA grand challenge: http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf
    - Autonomous Automobile Path Tracking: https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

    Parameters
    ----------
        track : Track
            track object with raceline/centerline
        config : dict | str, optional
            dictionary or path to yaml with controller specific parameters, by default None
            expects key "wheelbase": float, wheelbase of the vehicle

    Attributes:
        wheelbase (float, optional, default=0.33): wheelbase of the vehicle
        waypoints (numpy.ndarray [N, 4], optional, default=None): waypoints to track, columns are [x, y, velocity, heading]
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

        Raises
        ------
        ValueError
            if track is None or does not have waypoints (raceline or centerline)

        """
        if track is None or (track.raceline is None and track.centerline is None):
            raise ValueError("Track object with waypoints is required for the controller")
        
        # Extract waypoints from track
        reference = track.raceline if track.raceline is not None else track.centerline
        self.waypoints = np.stack(
                            [reference.xs, reference.ys, reference.vxs, reference.yaws], axis=1
                        )

        if isinstance(config, str):
            config = self.load_config(config)
        self.wheelbase = config.get("wheelbase", 0.33)
        self.drawn_waypoints = []
        self.target_point = None
        self.target_index = None

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
        
    def render_waypoints(self, e):
        """
        Callback to render waypoints.
        """
        points = self.waypoints[:, :2]
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    def render_target_point(self, e):
        """
        Callback to render the target point.
        """
        if self.target_point is not None:
            points = self.target_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.target_index is not None:
            points = self.waypoints[self.target_index : self.target_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=2)

    def calc_theta_and_ef(self, vehicle_state, waypoints):
        """
        Calculate the heading and cross-track errors
        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 4]): waypoints to track [x, y, velocity, heading]
        """

        # distance to the closest point to the front axle center
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])
        self.target_point, nearest_dist, t, self.target_index = nearest_point(
            position_front_axle, self.waypoints[:, 0:2]
        )
        vec_dist_nearest_point = position_front_axle - self.target_point

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

        return theta_e, ef, self.target_index, goal_veloctiy

    def controller(self, vehicle_state, waypoints, k_path):
        """
        Front Wheel Feedback Controller to track the path
        Based on the heading error theta_e and the crosstrack error ef we calculate the steering angle
        Returns the optimal steering angle delta is P-Controller with the proportional gain k

        Args:
            vehicle_state (numpy.ndarray [4, ]): [x, y, heading, velocity] of the vehicle
            waypoints (numpy.ndarray [N, 4]): waypoints to track
            k_path (float): proportional gain

        Returns:
            theta_e (float): heading error
            ef (numpy.ndarray [2, ]): crosstrack error
            theta_raceline (float): target heading
            kappa_ref (float): target curvature
            goal_veloctiy (float): target velocity
        """

        theta_e, ef, target_index, goal_veloctiy = self.calc_theta_and_ef(
            vehicle_state, waypoints
        )

        # Calculate final steering angle/ control input in [rad]: Steering Angle based on error + heading error
        cte_front = math.atan2(k_path * ef, vehicle_state[3])
        delta = cte_front + theta_e

        return delta, goal_veloctiy

    def plan(self, pose_x, pose_y, pose_theta, velocity, k_path=5.0, waypoints=None):
        """
        Plan function

        Args:
            pose_x (float):
            pose_y (float):
            pose_theta (float):
            velocity (float):
            k_path (float, optional, default=5):
            waypoints (numpy.ndarray [N x 4], optional, default=None):

        Returns:
            steering_angle (float): desired steering angle
            speed (float): desired speed
        """
        if waypoints is not None:
            if waypoints.shape[1] < 4 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 4, numpy array!")
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )
        k_path = np.float32(k_path)
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])
        steering_angle, speed = self.controller(vehicle_state, self.waypoints, k_path)
        return steering_angle, speed
