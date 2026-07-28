import argparse
import os
import pathlib

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_gym.envs.track import Track
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    LivelinessPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

from f1tenth_planning.control import (
    Nonlinear_Dynamic_MPPI_Planner as RoboracerController,
)
from f1tenth_planning.control.config.dynamics_config import (
    f1tenth_params,
)


class ControlRosWrapper(Node):
    def __init__(
        self,
        raceline="trajectory_logs.csv",
        odom_topic: str = "/pf/pose/odom",
        drive_topic: str = "/drive",
    ):
        super().__init__("control_node")
        # Waypoints from current directory
        waypoints_dir = pathlib.Path(os.path.join(os.path.dirname(__file__), raceline))
        # Load track waypoints
        waypoints_track: Track = Track.from_raceline_file(
            filepath=waypoints_dir,
            delimiter=";",
            skip_rows=3,
        )
        # For now, clip the velocity to 1m/s
        waypoints_track.raceline.vxs = np.clip(waypoints_track.raceline.vxs, None, 3.0)
        # Create planner
        self.planner = RoboracerController(
            track=waypoints_track, params=f1tenth_params()
        )
        # ROS publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # Create a QoS profile with KeepLast policy and depth of 1
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
            depth=1,
        )
        self.pose_sub = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, qos_profile
        )
        self.delta = 0.0
        self.local_plan_pub = self.create_publisher(
            MarkerArray, "/control/local_plan", 10
        )
        self.mpc_solution_pub = self.create_publisher(
            MarkerArray, "/control/mpc_solution", 10
        )
        self.global_plan_pub = self.create_publisher(
            MarkerArray, "/control/global_plan", 10
        )

        self.global_plan_marker_array = self._init_marker_array(
            self.planner.waypoints.shape[0], color=(0.0, 0.0, 1.0)
        )
        self.global_plan_marker_array = self._update_marker_array(
            self.global_plan_marker_array, self.planner.waypoints
        )

        self.local_plan_marker_array = self._init_marker_array(
            self.planner.solver.config.N + 1, color=(0.0, 1.0, 0.0)
        )
        self.mpc_solution_marker_array = self._init_marker_array(
            self.planner.solver.config.N, color=(1.0, 0.0, 0.0)
        )
        self.get_logger().info("Controller node initialized")

    def _init_marker_array(self, num_markers, color=(1.0, 0.0, 1.0)):
        marker_array = MarkerArray()
        for i in range(num_markers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 1.0
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.2
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker_array.markers.append(marker)
        return marker_array

    def _update_marker_array(self, marker_array: MarkerArray, points):
        num_update = len(points)
        if len(marker_array.markers) < len(points):
            num_update = len(marker_array.markers)

        for i in range(num_update):
            marker: Marker = marker_array.markers[i]
            marker.pose.position.x = float(points[i][0])
            marker.pose.position.y = float(points[i][1])
        for i in range(num_update, len(marker_array.markers)):
            marker = marker_array.markers[i]
            marker.action = Marker.DELETE

        return marker_array

    def publish_visualizations(self, local_plan, mpc_solution):
        # Check for subscribers before publishing
        if self.local_plan_pub.get_subscription_count() > 0:
            self.local_plan_pub.publish(
                self._update_marker_array(self.local_plan_marker_array, local_plan)
            )
        if self.mpc_solution_pub.get_subscription_count() > 0:
            self.mpc_solution_pub.publish(
                self._update_marker_array(self.mpc_solution_marker_array, mpc_solution)
            )
        if self.global_plan_pub.get_subscription_count() > 0:
            self.global_plan_pub.publish(self.global_plan_marker_array)

    def pose_callback(self, pose_msg: Odometry):
        # Get pose
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist
        beta = np.arctan2(twist.linear.y, twist.linear.x)
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        euler = R.from_quat(quaternion).as_euler("xyz", degrees=False)
        theta = euler[2]  # Yaw is the third element

        state_dict = {
            "pose_x": x,
            "pose_y": y,
            "delta": self.delta,
            "linear_vel_x": twist.linear.x,
            "pose_theta": theta,
            "ang_vel_z": twist.angular.z,
            "beta": beta,
        }

        # Plan control commands
        action, info = self.planner.compute_control(state_dict, params=self.params)
        steer_action = float(action[0])
        longitudtinal_action = float(action[1])

        self.publish_visualizations(
            np.array(self.planner.ref_traj[:2].T),
            np.array(self.planner.x_pred[:2, :].T),
        )

        # Convert steer_action to steering angle if needed
        steer = float(info["steering_angle"])

        # Convert longitudinal_action to speed if needed
        speed = float(info["velocity"])

        self.delta = steer
        # Publish control commands
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle_velocity = steer_action
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.acceleration = longitudtinal_action
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        # self.get_logger().info(
        #     f"Steering angle: {steer:.2f}, Speed: {speed:.2f}, Delta: {self.delta:.2f}"
        # )


def main(args=None):
    rclpy.init(args=args)
    print("Starting control node...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raceline",
        type=str,
        default="trajectory_logs.csv",
        help="Path to raceline file",
    )
    parser.add_argument(
        "--odom_topic", type=str, default="/pf/pose/odom", help="Odometry topic name"
    )
    parser.add_argument(
        "--drive_topic", type=str, default="/drive", help="Drive command topic name"
    )
    args = parser.parse_args()
    control_node = ControlRosWrapper(args.raceline, args.odom_topic, args.drive_topic)
    rclpy.spin(control_node)

    control_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
