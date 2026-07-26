import copy

from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig

import jax.numpy as jnp
import numpy as np
import casadi as ca


def _extract_kinematic_state(x0, xref):
    """Ensure that the states correspond to the kinematic model states."""
    # Extract first 5 states: [x, y, steering_angle, vx, yaw], assuming full state is
    # [x, y, steering_angle, vx, yaw, yaw_rate, slip_angle]
    return x0[:5], xref[:5, :]


class KinematicBicycleModel(DynamicsModel):
    """
    Kinematic bicycle model for vehicle dynamics.

    Vehicle State:
        - [x, y, delta, v, yaw]
    Control:
        - [delta_v, a]

    Reference:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads

    Args:
        DynamicsConfig: DynamicsConfig - vehicle dynamics configuration
    """

    def __init__(self, params: DynamicsConfig):
        super().__init__(params)
        self.nx = 5
        self.nu = 2

    def f(
        self, state: dict, control: np.ndarray, params: DynamicsConfig = None
    ) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        # Use the passed-in params for this call only; never mutate self.params.
        p = self.params if params is None else params

        x, y, delta, v, yaw = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw)
        dy = v * np.sin(yaw)
        ddelta = delta_v
        dv = a
        dyaw = (v / p.WHEELBASE) * np.tan(delta)

        return np.array([dx, dy, ddelta, dv, dyaw])

    def f_casadi(self) -> ca.Function:
        # State symbolic variables
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        delta = ca.SX.sym("delta")
        v = ca.SX.sym("v")
        yaw = ca.SX.sym("yaw")
        states = ca.vertcat(x, y, delta, v, yaw)

        # control symbolic variables
        a = ca.SX.sym("a")
        delta_v = ca.SX.sym("delta_v")
        controls = ca.vertcat(delta_v, a)

        # parameters symbolic variables
        wheelbase = ca.SX.sym("wheelbase")
        params = ca.vertcat(wheelbase)

        # right-hand side of the equation
        RHS = self.f_casadi_opti(states, controls, params)

        # maps controls, states and parameters to the right-hand side of the equation
        f = ca.Function("f", [states, controls, params], [RHS])
        return f

    def f_casadi_opti(self, state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
        # Extract params for more readable equations
        wheelbase = params[0]

        # Extract state variables from x
        x = state[0]
        y = state[1]
        delta = state[2]
        v = state[3]
        yaw = state[4]

        # Extract control variables from u
        delta_v = control[0]
        a = control[1]

        RHS = ca.vertcat(
            v * ca.cos(yaw),  # dx/dt = v * cos(yaw)
            v * ca.sin(yaw),  # dy/dt = v * sin(yaw)
            delta_v,  # d(delta)/dt = delta_v
            a,  # dv/dt = a
            (v / (wheelbase)) * ca.tan(delta),  # dyaw/dt = (v/(Lx+Ly)) * tan(delta)
        )  # dx/dt = f(x,u)

        return RHS

    def f_jax(
        self, state: jnp.ndarray, control: jnp.ndarray, params: jnp.ndarray
    ) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input using JAX.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        # Extract params for more readable equations. params is (num_params, 1),
        # so index 2-D like the dynamic model (params[0] alone is a (1,) array
        # and would make dyaw a length-1 array -> shape mismatch on jnp.array).
        wheelbase = params[0, 0]

        x, y, delta, v, yaw = state
        delta_v, a = control

        # params is the (num_params, 1) vector from parameters_vector_from_config.
        # Compute the state derivative
        dx = v * jnp.cos(yaw)
        dy = v * jnp.sin(yaw)
        ddelta = delta_v
        dv = a
        dyaw = (v / wheelbase) * jnp.tan(delta)

        return jnp.array([dx, dy, ddelta, dv, dyaw])

    def parameters_vector_from_config(self, params):
        return np.array([params.WHEELBASE]).reshape(-1, 1)

    def config_from_parameters_vector(self, p):
        """
        Convert a parameter vector to a dynamics_config object.

        Args:
            p (np.ndarray): parameter vector

        Returns:
            dynamics_config: vehicle dynamics configuration
        """
        # Return a copy of the live config with WHEELBASE updated; constructing a
        # bare DynamicsConfig(WHEELBASE=...) would raise (22 other required fields).
        current_params = copy.deepcopy(self.params)
        current_params.WHEELBASE = p[0, 0]
        return current_params

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters for the dynamic model.
        """
        active_params = [self.params.WHEELBASE]
        return len(active_params)

    def linearize_around_state(
        self, state: np.ndarray, control: np.ndarray, params: DynamicsConfig = None
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y, delta, v, yaw = state

        # State (or system) matrix A, 5x5
        A = np.zeros((self.nx, self.nx))

        A[0, 3] = np.cos(yaw)  # dx/d(v)

        A[0, 4] = -v * np.sin(yaw)  # dx/d(yaw)

        A[1, 3] = np.sin(yaw)  # dy/d(v)
        A[1, 4] = v * np.cos(yaw)  # dy/d(yaw)

        A[4, 2] = (v / self.params.WHEELBASE) * (
            1 / np.cos(delta) ** 2
        )  # dyaw/d(delta)
        A[4, 3] = np.tan(delta) / self.params.WHEELBASE  # dyaw/d(v)

        # Input Matrix B; 5x2
        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = 1

        return A, B
