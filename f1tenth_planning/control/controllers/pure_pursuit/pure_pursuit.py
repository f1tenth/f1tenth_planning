"""
Pure Pursuit waypoint tracker

Author: Hongrui Zheng
Last Modified: 5/4/22
"""

from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_planning.utils.utils import intersect_point
from f1tenth_planning.utils.utils import get_actuation
from f1tenth_planning.control.controller import Controller
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)

import numpy as np
import warnings


class PurePursuitPlanner(Controller):
    """
    Pure pursuit tracking controller.
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    This controller uses a lookahead circle to determine the target waypoint and computes the
    required steering and speed commands based on the vehicle's pose and a set of static or dynamic waypoints.

    Args:
        track (Track): Track instance containing the raceline information.
        params (DynamicsConfig, optional): Vehicle dynamic parameters. Defaults to DynamicsConfig().
        max_reacquire (float, optional): Maximum radius (in meters) to reacquire the current waypoint in case the vehicle drifts. Defaults to 20.0.
        lookahead_distance (float, optional): Default lookahead distance (in meters) to use if not provided during planning. Defaults to 0.8.

    Attributes:
        waypoints (numpy.ndarray [N x 4]): Static list of waypoints to track; columns correspond to [x, y, velocity, heading].
        lookahead_point (numpy.ndarray or None): The current lookahead point computed on the track.
        target_index (int or None): Index of the current waypoint.
    """

    def __init__(
        self,
        track: Track,
        params: DynamicsConfig = f1tenth_params(),
        lookahead_distance=0.8,
        max_reacquire=20.0,
    ):
        super(PurePursuitPlanner, self).__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Angle, LongitudinalActionEnum.Speed),
        )
        self.waypoints = np.vstack(
            [
                track.raceline.xs,
                track.raceline.ys,
                track.raceline.vxs,
                track.raceline.yaws,
            ]
        ).T

        self.lookahead_distance = lookahead_distance
        self.max_reacquire = max_reacquire
        self.lookahead_point = None
        self.target_index = None

        self.control_solution = None
        self.local_plan = None

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the lookahead circle intersection.

        Args:
            lookahead_distance (float): The lookahead distance to pick the next tracking point.
            position (numpy.ndarray): Current position of the vehicle as [x, y].
            theta (float): Current heading angle of the vehicle in radians.

        Returns:
            numpy.ndarray or None: The selected waypoint as [x, y, velocity],
            or None if no waypoint is found within the constraints.
        """
        nearest_p, nearest_dist, t, i = nearest_point(position, self.waypoints[:, 0:2])
        if nearest_dist < lookahead_distance:
            self.lookahead_point, self.target_index, t2 = intersect_point(
                position.astype(np.float32),
                lookahead_distance,
                self.waypoints[:, 0:2],
                np.float32(i + t),
                wrap=True,
            )
            if self.target_index is None:
                return None
            current_waypoint = np.array(
                [
                    self.waypoints[self.target_index, 0],
                    self.waypoints[self.target_index, 1],
                    self.waypoints[i, 2],
                ]
            )
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            self.target_index = i
            return self.waypoints[i, :]
        else:
            return None

    def plan(self, state: dict, waypoints=None, lookahead_distance=None):
        """
        Computes the steering angle and speed command based on the current state of the vehicle
        and the target waypoint found using the lookahead method.

        This function uses either dynamic waypoints provided at call-time or, if absent,
        the static raceline defined during planner instantiation. It determines the appropriate
        waypoint based on the given lookahead distance and calculates the required actuation commands.

        Args:
            state (dict): Dictionary containing the vehicle's state with keys 'pose_x', 'pose_y', and 'pose_theta'.
            waypoints (numpy.ndarray [N x 4], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, velocity, heading]. Overrides the static raceline if provided.
            lookahead_distance (float): The lookahead distance to use for computing the target waypoint. Overrides the default if provided.

        Returns:
            control: A tuple (steering_angle, speed) representing the computed steering command and speed.
            If no valid lookahead point is found, returns (0.0, 0.0) after issuing a warning.
        """
        if waypoints is not None:
            if len(waypoints.shape) != 2 or waypoints.shape[1] < 3:
                raise ValueError(
                    "Waypoints need to be a (N x m) numpy array with m >= 3!"
                )
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )
        if lookahead_distance is not None:
            self.lookahead_distance = lookahead_distance

        pose_x = state["pose_x"]
        pose_y = state["pose_y"]
        pose_theta = state["pose_theta"]

        position = np.array([pose_x, pose_y])
        lookahead_distance = np.float32(self.lookahead_distance)
        self.lookahead_point = self._get_current_waypoint(
            lookahead_distance, position, pose_theta
        )

        if self.lookahead_point is None:
            warnings.warn("Cannot find lookahead point, stopping...")
            return 0.0, 0.0

        # Use the actual distance to the selected point (the chord length that
        # get_actuation's pursuit-radius formula assumes). On the reacquire path
        # _get_current_waypoint returns the nearest raceline point (up to
        # max_reacquire away), so the nominal lookahead_distance would give a wrong
        # radius; the actual distance is correct for both branches.
        actual_lookahead = np.float32(
            np.linalg.norm(np.asarray(self.lookahead_point[:2], dtype=np.float32) - position)
        )
        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            actual_lookahead,
            self.params.WHEELBASE,
        )

        end_index = self.target_index + 10
        if end_index > self.waypoints.shape[0]:
            self.local_plan = np.vstack(
                (
                    self.waypoints[self.target_index :, :2],
                    self.waypoints[: end_index % self.waypoints.shape[0], :2],
                )
            )
        else:
            self.local_plan = self.waypoints[self.target_index : end_index, :2]
        self.control_solution = self.lookahead_point[:2][None]

        return steering_angle, speed
