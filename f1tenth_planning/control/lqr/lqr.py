# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz, Atsushi Sakai

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
Stanley waypoint tracker
Implementation inspired by https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/lqr_steer_control/lqr_steer_control.py

Author: Hongrui Zheng, Johannes Betz, Atsushi Sakai
Last Modified: 5/1/22
"""

class LQRPlanner():
    """
    Lateral Controller using LQR
    """

    def __init__(self, conf, wb):
        super().__init__(conf, wb)
        self.vehicle_control_e_cog = 0       # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0     # theta_e: yaw error to ref trajectory

    def calc_control_points(self, vehicle_state, waypoints):
        """
        Calculate all the errors to trajectory frame + find desired curvature and heading
        calc theta and ef
        Theta is the heading of the car, this heading must be minimized
        ef = crosstrack error/The distance from the optimal path/ lateral distance in frenet frame (front wheel)
        """

        ############# Calculate closest point to the front axle based on minimum distance calculation ################
        # Calculate Position of the front axle of the vehicle based on current position
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])

        # Find target index for the correct waypoint by finding the index with the lowest distance value/hypothenuses
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point_front, nearest_dist, t, target_index = planner_utils.nearest_point_on_trajectory_py2(position_front_axle, wpts)

        # Calculate the Distances from the front axle to all the waypoints
        distance_nearest_point_x= fx - nearest_point_front[0]
        distance_nearest_point_y = fy - nearest_point_front[1]
        vec_dist_nearest_point = np.array([distance_nearest_point_x, distance_nearest_point_y])

        ###################  Calculate the current Cross-Track Error ef in [m]   ################
        # Project crosstrack error onto front axle vector
        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - math.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - math.pi / 2.0)]])

        #vec_target_2_front = np.array([dx[target_index], dy[target_index]])

        # Caculate the cross-track error ef by
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        #############  Calculate the heading error theta_e  normalized to an angle to [-pi, pi]     ##########
        # Extract heading for the optimal raceline
        # BE CAREFUL: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[target_index][self.conf.wpt_thind]

        # Calculate the heading error by taking the difference between current and goal + Normalize the angles
        theta_e = planner_utils.pi_2_pi(theta_raceline - vehicle_state[2])

        # Calculate the target Veloctiy for the desired state
        goal_veloctiy = waypoints[target_index][self.conf.wpt_vind]

        #Find Reference curvature
        kappa_ref = self.waypoints[target_index][self.conf.wpt_kapind]

        #Saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy

    def controller(self, vehicle_state, waypoints, timestep, matrix_q, matrix_r, max_iteration, eps):
        """
        ComputeControlCommand calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)f
        :return: steering angle (optimal u), theta_e, e_cg
        """
        state_size = 4


        # Use the timestep of the simulation as a controller input calculation
        ts_ = timestep

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cg, yaw_ref, k_ref, v_ref = self.calc_control_points(vehicle_state, waypoints)

        #Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = planner_utils.update_matrix(vehicle_state, state_size, timestep, self.wheelbase)

        matrix_state_ = np.zeros((state_size, 1))
        matrix_r_ = np.diag(matrix_r)
        matrix_q_ = np.diag(matrix_q)

        matrix_k_ = planner_utils.solve_lqr(matrix_ad_, matrix_bd_, matrix_q_, matrix_r_, eps, max_iteration)

        matrix_state_[0][0] = e_cg
        matrix_state_[1][0] = (e_cg - e_cog_old) / ts_
        matrix_state_[2][0] = theta_e
        matrix_state_[3][0] = (theta_e - theta_e_old) / ts_

        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        #Compute feed forward control term to decrease the steady error.
        steer_angle_feedforward = k_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        # Calculate final speed control input in [m/s]:

        return steer_angle, v_ref

    def plan(self, pose_x, pose_y, pose_theta, velocity, timestep,
             matrix_q_1, matrix_q_2, matrix_q_3, matrix_q_4, matrix_r, iterations, eps):

        #Define LQR Matrix and Parameter
        matrix_q = [matrix_q_1, matrix_q_2, matrix_q_3, matrix_q_4]
        matrix_r = [matrix_r]

        #Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        #Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(vehicle_state, self.waypoints, timestep, matrix_q, matrix_r, iterations, eps)

        return speed,steering_angle