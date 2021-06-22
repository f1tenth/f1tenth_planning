# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz, Aman Sinha, Matthew O'Kelly

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
Author: Hongrui Zheng
Last Modified: 5/27/21
"""

from f1tenth_planner_base.base import TrackingPlanner
from f1tenth_planner_base.base import Planner
from f1tenth_pure_pursuit.pure_pursuit import PurePursuitPlanner
from f1tenth_planner_base.utils import nearest_point
from f1tenth_planner_base.utils import intersect_point

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

class LatticePlanner(Planner):
    """
    State Lattice Planner with cubic spirals

    Ref:
        1. Nagy, Bryan, and Alonzo Kelly. "Trajectory generation for car-like robots using cubic curvature polynomials." Field and Service Robots 11 (2001).
        2. A. Kelly and B. Nagy. Reactive nonholonomic trajectory generation via parametric optimal control. The International Journal of Robotics Research, 22(7-8):583–601, 2003.
        3. M. McNaughton. Parallel algorithms for real-time motion planning. 2011.

    TODO: the planner should take in three parameters for searching for splines:
            1. grid center look ahead, 2. grid width, 3. grid length
            resolution of the grid matches the resolution in the LUT
          also some parameter TBD for velocity control
    """

    # class variables
    lut_dict = np.load('lut_inuse.npz')
    lut_x = lut['x']
    lut_y = lut['y']
    lut_t = lut['theta']
    lut_k = lut['kappa']
    lut = lut['lut']

    tracker = PurePursuitPlanner()

    def __init__(self, conf):
        super().__init__(conf)

    def _get_current_waypoint(self):


if ROS2_PRESENT:
    class ROS2Lattice(Node):
        """
        ROS 2 node wrapper around Lattice planner
        # TODO: finish
        """
        def __init__(self):
            super().__init__('lattice_planner')

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