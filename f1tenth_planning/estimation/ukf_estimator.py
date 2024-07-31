from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import numpy as np

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    
    return x

def steering_constraints(steering_angle, steering_velocity, delta_min, delta_max, deltav_min, deltav_max):
    """
    steering_constraints - adjusts the steering velocity based on steering

    Inputs:
        :param steering_angle - steering angle
        :param steering_velocity - steering velocity
        :param delta_min - minimum steering angle
        :param delta_max - maximum steering angle
        :param deltav_min - minimum steering velocity
        :param deltav_max - maximum steering velocity
        
    Outputs:
        :return steering_velocity - steering velocity

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: Ahmad Amine - 24/07/2024
    """
    # steering limit reached?
    if (steering_angle <= delta_min and steering_velocity <= 0) or (steering_angle >= delta_max and steering_velocity >= 0):
        steering_velocity = 0
    else: 
        steering_velocity = np.clip(steering_velocity, deltav_min, deltav_max)

    return steering_velocity

def acceleration_constraints(velocity, acceleration, v_min, v_max, a_min, a_max, v_switch):
    """
    acceleration_constraints - adjusts the acceleration based on velocity

    Inputs:
        :param velocity - velocity
        :param acceleration - acceleration
        :param v_min - minimum velocity
        :param v_max - maximum velocity
        :param a_min - minimum acceleration
        :param a_max - maximum acceleration
        :param v_switch - switching velocity
        
    Outputs:
        :return acceleration - acceleration

    """
    # positive acceleration limit
    if velocity > v_switch:
        posLimit = a_max * v_switch / velocity
    else:
        posLimit = a_max

    # velocity limit reached?
    if (velocity <= v_min and acceleration <= 0) or (velocity >= v_max and acceleration >= 0):
        acceleration = 0
    else:
        acceleration = np.clip(acceleration, a_min, posLimit)

    return acceleration

def vehicle_dynamics_ks_cog(x, u_init, lwb, lf, v_min, v_max, delta_min, delta_max, deltav_min, deltav_max, a_min, a_max, v_switch):
    """
    vehicle_dynamics_ks_cog - kinematic single-track vehicle dynamics
    reference point: center of mass

    Inputs:
        :param x: vehicle state vector
        :param u_init: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations

    Author: Gerald Würsching
    Written: 17-November-2020
    Last update: 17-November-2020
    Last revision: ---
    """
    # states
    # x1 = x-position in a global coordinate system
    # x2 = y-position in a global coordinate system
    # x3 = steering angle of front wheels
    # x4 = velocity at center of mass
    # x5 = yaw angle

    # consider steering constraints
    u = []
    u.append(steering_constraints(x[2], u_init[0], delta_min, delta_max, deltav_min, deltav_max)) # different name u_init/u due to side effects of u
    # consider acceleration constraints
    u.append(acceleration_constraints(x[3], u_init[1], v_min, v_max, a_min, a_max, v_switch)) # different name u_init/u due to side effects of u

    # slip angle (beta) from vehicle kinematics
    beta = np.atan2(np.tan(x[2]) * lf, lwb)

    # system dynamics
    f = [x[3] * np.cos(beta + x[4]),
         x[3] * np.sin(beta + x[4]),
         u[0],
         u[1],
         x[3] * np.cos(beta) * np.tan(x[2]) / lwb]

    return f

'''
    UKF which uses a Single-Track Dynamic Bicycle model as state space model.
    Switches to a KS_CoG model at low velocities (v < 0.1 m/s) to avoid numerical instabilities.
    https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models

    States:
        x0 = x-position in a global coordinate system
        x1 = y-position in a global coordinate system
        x2 = steering angle of front wheels
        x3 = velocity in x-direction
        x4 = yaw angle
        x5 = yaw rate
        x6 = slip angle at vehicle center

    Inputs:
        u0 = steering angle velocity of front wheels
        u1 = longitudinal acceleration

    Dynamic model:
        f = [v * math.cos(beta + phi),
             v * math.sin(beta + phi),
             deltav,
             a,
             omega_z,
             -mu * m / (v * I * (lr + lf)) * (
                         lf ** 2 * C_Sf * (g * lr - a * h) + lr ** 2 * C_Sr * (g * lf + a * h)) * omega_z \
             + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h)) * beta \
             + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta,
             (mu / (v ** 2 * (lr + lf)) * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf) - 1) *
             omega_z \
             - mu / (v * (lr + lf)) * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h)) * beta \
             + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta]

    Assumes noisy measurements of the following:
        z = [x, y, delta, vx, yaw, yaw_rate]
    
'''
class ST_UKF():
    def __init__(self, vehicle_params, dt,
                 P=np.eye(7), Q=np.eye(7), R=np.eye(6)):
        # Vehicle parameters
        self.lf   = vehicle_params['lf']
        self.lr   = vehicle_params['lr']
        self.m    = vehicle_params['m']
        self.I    = vehicle_params['I']
        self.h    = vehicle_params['h']
        self.C_Sf = vehicle_params['C_Sf']
        self.C_Sr = vehicle_params['C_Sr']
        self.mu   = vehicle_params['mu']
        self.g    = vehicle_params['g']

        # Constraints
        self.v_min      = vehicle_params['v_min']
        self.v_max      = vehicle_params['v_max']
        self.delta_min  = vehicle_params['s_min']
        self.delta_max  = vehicle_params['s_max']
        self.deltav_min = vehicle_params['sv_min']
        self.deltav_max = vehicle_params['sv_max']
        self.a_min      = vehicle_params['a_min']
        self.a_max      = vehicle_params['a_max']
        self.v_switch   = vehicle_params['v_switch']
        
        # State dimension
        self.dim_x = 7

        # Measurement dimension
        dim_z = 6

        # Initial state mean
        x = np.zeros(self.dim_x)

        # Initial state covariance
        self.P = P

        # Process noise covariance
        self.Q = Q

        # Measurement noise covariance
        self.R = R

        # UKF parameters
        # Recommended parameters as per the book
        beta = 2 # Van der Merwe suggests beta = 2 is a good choice for Gaussian problems
        kappa = max(0, 3 - self.dim_x) # kappa is an arbitrary constant
        alpha = 0.001 # alpha is a scaling parameter, 10^-3 is a common value
        self.points = MerweScaledSigmaPoints(n=self.dim_x, alpha=alpha, beta=beta, kappa=kappa)

        # UKF instance
        self.filter = UKF(dim_x=self.dim_x, dim_z=dim_z,
                          fx=self.vehicle_dynamics_st, hx=self.measurement,
                          residual_x=self.residual_x, residual_z=self.residual_h, x_mean_fn=self.state_mean, z_mean_fn=self.z_mean,
                          dt=dt, points=self.points)

        # Initialize the UKF
        self.filter.x = x
        self.filter.P = P
        self.filter.Q = Q
        self.filter.R = R

    def vehicle_dynamics_st(self, x, u):
        v = x[3]
        
        # Use KS_CoG for low speeds
        if abs(v) < 0.1:
            # wheelbase
            lwb = self.lf + self.lr
            
            # ks_cog state vector
            x_ks = [x[0], x[1], x[2], x[3], x[4]]

            # ks_cog dynamics
            f_ks = vehicle_dynamics_ks_cog(x_ks, u, lwb, self.lf, self.v_min, self.v_max, self.delta_min, self.delta_max, self.deltav_min, self.deltav_max, self.a_min, self.a_max, self.v_switch)

            f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
            # derivative of slip angle and yaw rate
            d_beta = (self.lr * u[0]) / (lwb * np.cos(x[2]) ** 2 * (1 + (np.tan(x[2]) ** 2 * self.lr / lwb) ** 2))
            dd_psi = 1 / lwb * (u[1] * np.cos(x[6]) * np.tan(x[2]) -
                                x[3] * np.sin(x[6]) * d_beta * np.tan(x[2]) +
                                x[3] * np.cos(x[6]) * u[0] / np.cos(x[2]) ** 2)
            f.append(dd_psi)
            f.append(d_beta)
        else:
            # system dynamics
            f = [v * np.cos(x[6] + x[4]),
                 v * np.sin(x[6] + x[4]),
                 u[0],
                 u[1],
                 x[5],
                 -self.mu * self.m / (x[3] * self.I * (self.lr + self.lf)) * (
                             self.lf ** 2 * self.C_Sf * (self.g * self.lr - u[1] * self.h) + self.lr ** 2 * self.C_Sr * (
                             self.g * self.lf + u[1] * self.h)) * x[5] \
                 + self.mu * self.m / (self.I * (self.lr + self.lf)) * (
                             self.lr * self.C_Sr * (self.g * self.lf + u[1] * self.h) - self.lf * self.C_Sf * (
                             self.g * self.lr - u[1] * self.h)) * x[6] \
                 + self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf * self.C_Sf * (
                             self.g * self.lr - u[1] * self.h) * x[2],
                 (self.mu / (x[3] ** 2 * (self.lr + self.lf)) * (self.C_Sr * (self.g * self.lf + u[1] * self.h) * self.lr - self.C_Sf * (self.g * self.lr - u[1] * self.h) * self.lf) - 1) *
                 x[5] \
                 - self.mu / (x[3] * (self.lr + self.lf)) * (self.C_Sr * (self.g * self.lf + u[1] * self.h) + self.C_Sf * (self.g * self.lr - u[1] * self.h)) * x[6] \
                 + self.mu / (x[3] * (self.lr + self.lf)) * (self.C_Sf * (self.g * self.lr - u[1] * self.h)) * x[2]]        
        
        return f

    def measurement(self, x):
        # measurement function - convert state to a measurement, all except slip angle
        return x[:6]
    
    def residual_h(self, a, b):
        y = a - b
        # Normalize heading error and slip angle error
        y[4] = normalize_angle(y[4])
        return y

    def residual_x(self, a, b):
        y = a - b

        # Normalize heading error and slip angle error
        y[4] = normalize_angle(y[4])
        y[6] = normalize_angle(y[6])
        return y

    def state_mean(self, sigmas, Wm):
        # state is [x, y, delta, vx, yaw, yaw_rate, slip_angle]
        x = np.zeros(self.dim_x)

        sum_sin_yaw = np.sum(np.dot(np.sin(sigmas[:, 4]), Wm))
        sum_cos_yaw = np.sum(np.dot(np.cos(sigmas[:, 4]), Wm))

        sum_sin_beta = np.sum(np.dot(np.sin(sigmas[:, 6]), Wm))
        sum_cos_beta = np.sum(np.dot(np.cos(sigmas[:, 6]), Wm))

        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = np.sum(np.dot(sigmas[:, 2], Wm))
        x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
        x[4] = np.atan2(sum_sin_yaw, sum_cos_yaw)
        x[5] = np.sum(np.dot(sigmas[:, 5], Wm))
        x[6] = np.atan2(sum_sin_beta, sum_cos_beta)
        return x

    def z_mean(self, sigmas, Wm):
        z_count = sigmas.shape[1]
        z = np.zeros(z_count)

        # Measurement is [x, y, delta, vx, yaw, yaw_rate]
        z[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        z[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        z[2] = np.sum(np.dot(sigmas[:, 2], Wm))
        z[3] = np.sum(np.dot(sigmas[:, 3], Wm))

        sum_sin_yaw = np.sum(np.dot(np.sin(sigmas[:, 4]), Wm))
        sum_cos_yaw = np.sum(np.dot(np.cos(sigmas[:, 4]), Wm))
        z[4] = np.arctan2(sum_sin_yaw, sum_cos_yaw)

        z[5] = np.sum(np.dot(sigmas[:, 5], Wm))
        return z

    def step(self, u, z):
        self.filter.predict(u)
        self.filter.update(z)

        return self.filter.x, self.filter.P
    
    def get_state(self):
        return self.filter.x
    
    def get_covariance(self):
        return self.filter.P
    
    def set_state(self, x):
        self.filter.x = x

    def set_covariance(self, P):
        self.filter.P = P

    def set_measurement_noise(self, R):
        self.filter.R = R

    def set_process_noise(self, Q):
        self.filter.Q = Q

    def set_initial_state(self, x):
        self.filter.x = x

    def set_initial_covariance(self, P):
        self.filter.P = P

    def get_measurement_noise(self):
        return self.filter.R
    
    def get_process_noise(self):
        return self.filter.Q