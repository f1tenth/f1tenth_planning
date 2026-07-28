import copy
from functools import partial
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig

import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp


class DynamicBicycleModel(DynamicsModel):
    """
    Dynamic single-track bicycle model for vehicle dynamics.

    Vehicle State:
        - [x, y, delta, v, yaw, yaw_rate, slip_angle]
    Control:
        - [delta_v, a]

    Reference:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads

    Args:
        DynamicsConfig: DynamicsConfig - vehicle dynamics configuration
    """

    STATE_NAMES = ("x", "y", "delta", "v", "yaw", "yaw_rate", "beta")
    CONTROL_NAMES = ("delta_v", "a")

    def __init__(self, params: DynamicsConfig):
        super().__init__(params)
        self.nx = len(self.STATE_NAMES)
        self.nu = len(self.CONTROL_NAMES)

    def f(
        self, state: dict, control: np.ndarray, params: DynamicsConfig = None
    ) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw, yaw_rate, slip_angle]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        # Use the passed-in params for this call only; never mutate self.params
        # (a one-off derivative evaluation must not rebind the model's config).
        p = self.params if params is None else params

        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw + slip_angle)
        dy = v * np.sin(yaw + slip_angle)
        ddelta = delta_v
        dv = a

        dyaw = 0
        ddyaw = 0
        dslip_angle = 0
        if v < 0.5:
            # kinematic (low-speed) regime; threshold matches the gym switch (V < 0.5)
            dyaw = v * np.cos(slip_angle) / p.WHEELBASE * np.tan(delta)

            # derivative of slip angle and yaw rate
            dslip_angle = (p.LR * delta_v) / (
                p.WHEELBASE
                * np.cos(delta) ** 2
                * (1 + (np.tan(delta) * p.LR / p.WHEELBASE) ** 2)
            )
            ddyaw = (
                1
                / p.WHEELBASE
                * (
                    a * np.cos(slip_angle) * np.tan(delta)
                    - v * np.sin(slip_angle) * np.tan(delta) * dslip_angle
                    + v * np.cos(slip_angle) * delta_v / np.cos(delta) ** 2
                )
            )
        else:
            dyaw = yaw_rate

            # Extract params for more readable equations
            mu = p.MU
            m = p.M
            I = p.I
            lr = p.LR
            lf = p.LF
            C_Sf = p.C_SF
            C_Sr = p.C_SR
            h = p.H
            g = 9.81

            ddyaw = (
                -mu
                * m
                / (v * I * (lr + lf))
                * (lf**2 * C_Sf * (g * lr - a * h) + lr**2 * C_Sr * (g * lf + a * h))
                * yaw_rate
                + mu
                * m
                / (I * (lr + lf))
                * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h))
                * slip_angle
                + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta
            )

            dslip_angle = (
                (
                    mu
                    / (v**2 * (lr + lf))
                    * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf)
                    - 1
                )
                * yaw_rate
                - mu
                / (v * (lr + lf))
                * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h))
                * slip_angle
                + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta
            )

        return np.array([dx, dy, ddelta, dv, dyaw, ddyaw, dslip_angle])

    def f_casadi(self) -> ca.Function:
        # State symbolic variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        delta = ca.SX.sym("delta")
        v = ca.SX.sym("v")
        yaw = ca.SX.sym("yaw")
        yaw_rate = ca.SX.sym("yaw_rate")
        slip_angle = ca.SX.sym("slip_angle")
        states = ca.vertcat(x, y, delta, v, yaw, yaw_rate, slip_angle)
        # control symbolic variables
        a = ca.SX.sym("a")
        delta_v = ca.SX.sym("delta_v")
        controls = ca.vertcat(delta_v, a)

        # parameters symbolic variables
        mu = ca.SX.sym("mu")
        m = ca.SX.sym("m")
        I = ca.SX.sym("I")
        lr = ca.SX.sym("lr")
        lf = ca.SX.sym("lf")
        C_Sf = ca.SX.sym("C_Sf")
        C_Sr = ca.SX.sym("C_Sr")
        h = ca.SX.sym("h")
        g = ca.SX.sym("g")
        params = ca.vertcat(mu, m, I, lr, lf, C_Sf, C_Sr, h, g)

        # right-hand side of the equation
        RHS = self.f_casadi_opti(states, controls, params)

        # maps controls, states and parameters to the right-hand side of the equation
        f = ca.Function("f", [states, controls, params], [RHS])
        return f

    @partial(jax.jit, static_argnums=(0))
    def f_jax(
        self, state: jnp.ndarray, control: jnp.ndarray, params: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Single Track Dynamic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                    x6: yaw rate
                    x7: slip angle at vehicle center
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # Extract params for more readable equations
        mu = params[0, 0]
        m = params[1, 0]
        I = params[2, 0]
        lr = params[3, 0]
        lf = params[4, 0]
        C_Sf = params[5, 0]
        C_Sr = params[6, 0]
        h = params[7, 0]
        g = params[8, 0]
        wheelbase = lf + lr

        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * jnp.cos(yaw + slip_angle)
        dy = v * jnp.sin(yaw + slip_angle)
        ddelta = delta_v
        dv = a

        dyaw = 0
        ddyaw = 0
        dslip_angle = 0

        # derivative of yaw "kinemaitcally"
        dyaw_ks = v * jnp.cos(slip_angle) * jnp.tan(delta) / wheelbase

        # derivative of slip angle and yaw rate
        dslip_angle_ks = (lr * delta_v) / (
            wheelbase
            * jnp.cos(delta) ** 2
            * (1 + (jnp.tan(delta) * lr / wheelbase) ** 2)
        )
        ddyaw_ks = (
            1
            / wheelbase
            * (
                a * jnp.cos(slip_angle) * jnp.tan(delta)
                - v * jnp.sin(slip_angle) * jnp.tan(delta) * dslip_angle_ks
                + v * jnp.cos(slip_angle) * delta_v / jnp.cos(delta) ** 2
            )
        )

        dyaw_st = yaw_rate

        # lax.select evaluates BOTH branches, so guard the 1/v terms to keep the
        # (discarded) high-speed branch finite at v=0 — otherwise jax.grad -> NaN.
        epsilon = 1e-4
        v_eps = v + epsilon

        ddyaw_st = (
            -mu
            * m
            / (v_eps * I * (lr + lf))
            * (lf**2 * C_Sf * (g * lr - a * h) + lr**2 * C_Sr * (g * lf + a * h))
            * yaw_rate
            + mu
            * m
            / (I * (lr + lf))
            * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h))
            * slip_angle
            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta
        )

        dslip_angle_st = (
            (
                mu
                / (v_eps**2 * (lr + lf))
                * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf)
                - 1
            )
            * yaw_rate
            - mu
            / (v_eps * (lr + lf))
            * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h))
            * slip_angle
            + mu / (v_eps * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta
        )

        return jax.lax.select(
            v < 0.5,
            jnp.array([dx, dy, ddelta, dv, dyaw_ks, ddyaw_ks, dslip_angle_ks]),
            jnp.array([dx, dy, ddelta, dv, dyaw_st, ddyaw_st, dslip_angle_st]),
        )

    def f_casadi_opti(self, state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
        # Extract params for more readable equations
        mu = params[0]
        m = params[1]
        I = params[2]
        lr = params[3]
        lf = params[4]
        C_Sf = params[5]
        C_Sr = params[6]
        h = params[7]
        g = params[8]

        # Extract state variables from x
        x = state[0]
        y = state[1]
        delta = state[2]
        v = state[3]
        yaw = state[4]
        yaw_rate = state[5]
        slip_angle = state[6]

        # Extract control variables from u
        delta_v = control[0]
        a = control[1]

        dyaw_slow = v * ca.cos(slip_angle) * ca.tan(delta) / (lr + lf)
        d_beta_slow = (lr * delta_v) / (
            (lr + lf)
            * ca.cos(delta) ** 2
            * (1 + (ca.tan(delta) * lr / (lr + lf)) ** 2)
        )
        dyaw_rate_slow = (
            1
            / (lr + lf)
            * (
                a * ca.cos(slip_angle) * ca.tan(delta)
                - v * ca.sin(slip_angle) * ca.tan(delta) * d_beta_slow
                + v * ca.cos(slip_angle) * delta_v / (ca.cos(delta) ** 2)
            )
        )

        epsilon = 1e-4
        dyaw_fast = yaw_rate  # dyaw/dt = yaw_rate
        glr = g * lr - a * h
        glf = g * lf + a * h
        # system dynamics
        dyaw_rate_fast = (mu * m / (I * (lr + lf))) * (
            lf * C_Sf * (glr) * delta
            + (lr * C_Sr * (glf) - lf * C_Sf * (glr)) * slip_angle
            - (lf**2 * C_Sf * (glr) + lr**2 * C_Sr * (glf)) * (yaw_rate / (v + epsilon))
        )
        d_beta_fast = (mu / ((v + epsilon) * (lf + lr))) * (
            C_Sf * (glr) * delta
            - (C_Sr * (glf) + C_Sf * (glr)) * slip_angle
            + (C_Sr * (glf) * lr - C_Sf * (glr) * lf) * (yaw_rate / (v + epsilon))
        ) - yaw_rate

        RHS_LOW_SPEED = ca.vertcat(
            v * ca.cos(yaw + slip_angle),  # dx/dt = v * cos(yaw + slip_angle)
            v * ca.sin(yaw + slip_angle),  # dy/dt = v * sin(yaw + slip_angle)
            delta_v,  # d(delta)/dt = delta_v
            a,  # dv/dt = a
            dyaw_slow,  # dyaw/dt = yaw_rate
            dyaw_rate_slow,  # dyaw_rate/dt = RHS
            d_beta_slow,  # dbeta/dt = d_beta
        )  # dx/dt = f(x,u)

        RHS_HIGH_SPEED = ca.vertcat(
            v * ca.cos(yaw + slip_angle),  # dx/dt = v * cos(yaw + slip_angle)
            v * ca.sin(yaw + slip_angle),  # dy/dt = v * sin(yaw + slip_angle)
            delta_v,  # d(delta)/dt = delta_v
            a,  # dv/dt = a
            dyaw_fast,  # dyaw/dt = yaw_rate
            dyaw_rate_fast,  # dyaw_rate/dt = RHS
            d_beta_fast,  # dbeta/dt = d_beta
        )  # dx/dt = f(x,u)

        RHS = ca.if_else(v < 0.5, RHS_LOW_SPEED, RHS_HIGH_SPEED)

        return RHS

    def parameters_vector_from_config(self, params):
        return np.array(
            [
                params.MU,
                params.M,
                params.I,
                params.LR,
                params.LF,
                params.C_SF,
                params.C_SR,
                params.H,
                9.81,
            ]
        ).reshape(-1, 1)

    def config_from_parameters_vector(self, params):
        """
        Convert a vector of parameters into a dynamics configuration object. This function is useful for optimization problems where the
        parameters need to be passed as a vector.

        Args:
            params (np.ndarray): (num_params, 1) vector of parameters

        Returns:
            dynamics_config: vehicle dynamics parameters
        """
        # Return a copy; do not mutate the model's live config in place.
        current_params = copy.deepcopy(self.params)
        current_params.MU = params[0, 0]
        current_params.M = params[1, 0]
        current_params.I = params[2, 0]
        current_params.LR = params[3, 0]
        current_params.LF = params[4, 0]
        current_params.C_SF = params[5, 0]
        current_params.C_SR = params[6, 0]
        current_params.H = params[7, 0]
        # Keep WHEELBASE consistent with LR/LF: the numpy low-speed branch reads
        # p.WHEELBASE while jax/casadi recompute lf+lr, so without this they diverge
        # when LR/LF are perturbed (e.g. during system identification).
        current_params.WHEELBASE = params[3, 0] + params[4, 0]
        # params[8] is gravity, a fixed model constant, not a stored config field.
        return current_params

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters for the dynamic model.
        """
        # Must match parameters_vector_from_config (9 entries, gravity last).
        active_params = [
            self.params.MU,
            self.params.M,
            self.params.I,
            self.params.LR,
            self.params.LF,
            self.params.C_SF,
            self.params.C_SR,
            self.params.H,
            9.81,  # gravity
        ]
        return len(active_params)
