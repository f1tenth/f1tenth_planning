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
Pure Pursuit waypoint tracker

Author: Hongrui Zheng
Last Modified: 5/27/21
"""

from planner_base.base import TrackingPlanner
from planner_base.utils import nearest_point
from planner_base.utils import intersect_point
from planner_base.utils import get_actuation
from planner_base.utils import quat_2_rpy

ROS2_PRESENT = True
try:
    # try importing ros2
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PoseStamped

except:
    # no ros2 import
    ROS2_PRESENT = False
    pass

import numpy as np

class PurePursuitPlanner(TrackingPlanner):
    """
    Pure pursuit tracking controller
    Reference: Coulter 1992, https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

    All vehicle pose used by the planner should be in the map frame.

    Configuration dictionary spec:
        wpt_xind: index for x values in waypoints
        wpt_yind: index for y values in waypoints
        wpt_vind: index for velocity values in waypoints
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.max_reacquire = 20.

    def _get_current_waypoint(self, lookahead_distance, position, theta):
        """
        Finds the current waypoint on the look ahead circle intersection

        Args:
            lookahead_distance (float): lookahead distance to find next point to track
            position (numpy.ndarray (2, )): current position of the vehicle (x, y)
            theta (float): current vehicle heading

        Returns:
            current_waypoint (numpy.ndarray (3, )): selected waypoint (x, y, velocity), None if no point is found
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind],
                          self.waypoints[:, self.conf.wpt_yind])).T
        nearest_p, nearest_dist, t, i = nearest_point(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(position,
                                                      lookahead_distance,
                                                      wpts,
                                                      i + t,
                                                      wrap=True)
            if i2 is None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(lookahead_distance,
                                                     position,
                                                     pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta,
                                              lookahead_point,
                                              position,
                                              lookahead_distance,
                                              self.wheelbase)

        return speed, steering_angle


if ROS2_PRESENT:
    class ROS2PurePursuit(Node):
        """
        ROS 2 node wrapper around pure pursuit
        """
        def __init__(self):
            super().__init__('pure_pursuit')

            # get parameters
            pose_topic = self.get_parameter('pose_topic').value
            drive_topic = self.get_parameter('drive_topic').value
            self.conf = {}
            self.conf['wpt_xind'] = self.get_parameter('wpt_xind').value
            self.conf['wpt_yind'] = self.get_parameter('wpt_yind').value
            self.conf['wpt_vind'] = self.get_parameter('wpt_vind').value
            self.conf['use_csv'] = self.get_parameter('use_csv').value
            self.conf['wpt_path'] = self.get_parameter('wpt_path').value
            self.conf['lookahead_distance'] = self.get_parameter('lookahead_distance').value
            self.conf['wheelbase'] = self.get_parameter('wheelbase').value

            # ROS 2 subscriptions and publishers
            self.pose_subscription = self.create_subscription(
                PoseStamped,
                pose_topic,
                self.pose_callback,
                1)
            self.pose_subscription
            self.drive_publisher = self.create_publisher(
                AckermannDriveStamped,
                drive_topic,
                10)

            # create pure pursuit planner
            self.planner = PurePursuitPlanner(self.conf)
            self.lh = self.conf['lookahead_distance']

        def pose_callback(self, msg):
            x = msg.pose.position.x
            y = msg.pose.position.y
            qx = msg.pose.orienetation.x
            qy = msg.pose.orienetation.y
            qz = msg.pose.orienetation.z
            qw = msg.pose.orienetation.w
            _, _, theta = quat_2_rpy(qx, qy, qz, qw)
            speed, steer = self.planner.plan(x, y, theta, self.lh)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = speed
            drive_msg.drive.steering_angle = steer
            self.drive_publisher.publish(drive_msg)

# node definition
def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = ROS2PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

# fall back main for no ROS2
def main_noros2():
    print('No ROS2 present, skipping node spin up.')

if __name__ == '__main__':
    if ROS2_PRESENT:
        main()
    else:
        main_noros2()