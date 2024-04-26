from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation

from f1tenth_planning.control.controller import Controller
from f1tenth_gym.envs.track import Track

import yaml
import pathlib
import numpy as np
import warnings

class PurePursuitController(Controller):
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Parameters
    ----------
        track : Track
            track object with raceline/centerline
        config : dict | str, optional
            dictionary or path to yaml with controller specific parameters, by default None 
            expects the following keys : 
                 "wheelbase": float, wheelbase of the vehicle
                 "lookahead_distance": float, lookahead distance for the controller
                 "max_reacquire": float, maximum radius for reacquiring current waypoints

    Attributes
    ----------
        max_reacquire (float): maximum radius (meters) for reacquiring current waypoints
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
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
            - lookahead_distance: float, lookahead distance for the controller
            - max_reacquire: float, maximum radius for reacquiring current waypoints

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
                            [reference.xs, reference.ys, reference.vxs, reference.yaws], axis=1
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

        self.wheelbase = config.get("wheelbase", 0.33)
        self.lookahead_distance = config.get("lookahead_distance", 0.8)
        self.max_reacquire = config.get("max_reacquire", 5.0)

        self.lookahead_point = None
        self.current_index = None

    def update(self, config: dict) -> None:
        """Updates setting of controller

        Parameters
        ----------
        config : dict
            configurations to update
        """
        self.wheelbase = config.get("wheelbase", self.wheelbase)
        self.lookahead_distance = config.get("lookahead_distance", self.lookahead_distance)
        self.max_reacquire = config.get("max_reacquire", self.max_reacquire)

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
        
    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point.

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer

        Parameters
        ----------
        e : EnvRenderer
            environment renderer
        """
        if self.current_index is not None:
            points = self.waypoints[self.current_index : self.current_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=2)

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the look ahead circle intersection

        Parameters
        ----------
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)
            theta (float): current vehicle heading

        Returns
        -------
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """

        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.current_index, t2 = intersect_point(
                position,
                lookahead_distance,
                self.waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.current_index is None:
                return None
            current_waypoint = np.array(
                [
                    self.waypoints[self.current_index, 0],
                    self.waypoints[self.current_index, 1],
                    self.waypoints[i, 2],
                ]
            )
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return self.waypoints[i, :]
        else:
            return None

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
        position = np.array([state["pose_x"], state["pose_y"]])
        pose_theta = state["pose_theta"]

        self.lookahead_point = self._get_current_waypoint(
            self.lookahead_distance, position, pose_theta
        )

        if self.lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return 0.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            self.lookahead_distance,
            self.wheelbase,
        )

        return steering_angle, speed
