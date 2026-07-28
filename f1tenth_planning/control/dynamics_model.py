from __future__ import annotations
from abc import abstractmethod, ABC

import jax
from jax import numpy as jnp
import numpy as np
import casadi as ca
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig
from f1tenth_planning.control.spec import OBSERVATION_KEY_FOR_STATE, VariableSpec

class DynamicsModel(ABC):
    """Base class for vehicle dynamics models.

    **Backends are opt-in** (DESIGN.md §5.1). A model implements only the backends
    its solvers need -- a jax-only learned model implements ``f_jax`` and nothing
    else. Use :meth:`backends` / :meth:`supports` to query what a model provides;
    solvers validate their requirements at construction rather than failing deep
    inside a solve.

    **Layouts are self-describing** (DESIGN.md §5.2). Subclasses declare
    ``STATE_NAMES`` and ``CONTROL_NAMES``; generic code then addresses variables by
    name via ``model.state.idx.<name>`` instead of hardcoding vector positions.
    """

    #: Names of the state variables, in vector order. Subclasses must set this.
    STATE_NAMES: tuple[str, ...] = ()
    #: Names of the control variables, in vector order. Subclasses must set this.
    CONTROL_NAMES: tuple[str, ...] = ()

    #: Backend keys understood by :meth:`backends`, mapped to the defining method.
    _BACKEND_METHODS = {
        "numpy": "f",
        "jax": "f_jax",
        "casadi": "f_casadi",
        "casadi_opti": "f_casadi_opti",
        "jacobian": "linearize_around_state",
    }

    @abstractmethod
    def __init__(self, params: DynamicsConfig) -> None:
        """
        Initialize the dynamics model.

        Args:
            params (DynamicsConfig): vehicle dynamics parameters
        """
        self.params = params

    # ------------------------------------------------------------------ layouts
    @property
    def state(self) -> VariableSpec:
        """Named layout of the state vector (``model.state.idx.v`` -> index)."""
        spec = getattr(self, "_state_spec", None)
        if spec is None:
            if not self.STATE_NAMES:
                raise NotImplementedError(
                    f"{type(self).__name__} does not declare STATE_NAMES"
                )
            spec = VariableSpec(self.STATE_NAMES)
            self._state_spec = spec
        return spec

    @property
    def control(self) -> VariableSpec:
        """Named layout of the control vector (``model.control.idx.a`` -> index)."""
        spec = getattr(self, "_control_spec", None)
        if spec is None:
            if not self.CONTROL_NAMES:
                raise NotImplementedError(
                    f"{type(self).__name__} does not declare CONTROL_NAMES"
                )
            spec = VariableSpec(self.CONTROL_NAMES)
            self._control_spec = spec
        return spec

    def state_from_observation(self, observation: dict) -> np.ndarray:
        """Assemble this model's state vector from a gym observation dict.

        Each model pulls exactly the fields its own layout names, so a 5-state
        kinematic model and a 7-state dynamic model both build correctly from the
        same observation -- no shared-prefix slicing and no pre-processing hook.
        """
        values = []
        for name in self.state.names:
            key = OBSERVATION_KEY_FOR_STATE.get(name)
            if key is None:
                raise KeyError(
                    f"state variable {name!r} cannot be read from an observation; "
                    f"override state_from_observation in {type(self).__name__}"
                )
            if key not in observation:
                raise KeyError(
                    f"observation is missing {key!r} (needed for state {name!r}); "
                    f"available keys: {sorted(observation)}"
                )
            values.append(observation[key])
        return np.array(values, dtype=float)

    # ------------------------------------------------------------- capabilities
    @classmethod
    def backends(cls) -> set[str]:
        """Which backends this model actually implements.

        Detected by checking which methods the subclass overrides, so a model that
        simply does not define ``f_casadi`` reports no casadi support.
        """
        provided = set()
        for key, method in cls._BACKEND_METHODS.items():
            if getattr(cls, method, None) is not getattr(DynamicsModel, method, None):
                provided.add(key)
        return provided

    @classmethod
    def supports(cls, *required: str) -> bool:
        """True if every named backend is implemented by this model."""
        return set(required).issubset(cls.backends())

    def f(self, state: np.ndarray, control: np.ndarray, params: DynamicsConfig = None) -> np.ndarray:
        """
        (Non-)linear dynamics model. This function computes the state derivative given the current state and control input. Should be 
        paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state and control
        should be handled externally.
        
        Mathematically:
            \\dot{x} = f(x, u)

        Args:
            state (np.ndarray): observation as returned from the environment.
            control (np.ndarray): control input as (steering_angle, speed)
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        raise NotImplementedError("control method not implemented")

    def f_casadi(self, params: DynamicsConfig = None) -> ca.Function:
        """
        (Non-)linear dynamics model in CasADi symbolic form. This function computes the state derivative given the current state and control 
        input. Should be paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state 
        and control should be handled externally. This function will create symbolic variables for each state and control input, and return a
        CasADi function that can be used to compute the state derivative.
        
        Mathematically:
            \\dot{x} = f(x, u)
        
        Args:
            params (DynamicsConfig): vehicle dynamics parameters, overwrites self.params if not None

        Returns:
            ca.Function: CasADi function for the state derivative
        """
        raise NotImplementedError("control method not implemented")
    
    def f_casadi_opti(self, state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
        """
        Casadi OptiStack compatible function for the dynamic model. 
        
        Args:
            x (ca.SX): (nx, 1) state vector
            u (ca.SX): (nu, 1) control vector
            p (ca.SX): (num_p, 1) parameter vector
        Returns:
            ca.SX: (nx, 1) state derivative
        """
        raise NotImplementedError("control method not implemented")

    def f_jax(self, state: jnp.ndarray, control: jnp.ndarray, params: jnp.ndarray = None) -> jnp.ndarray:
        """
        (Non-)linear dynamics model in JAX. This function computes the state derivative given the current state and control input. Should be 
        paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state and control
        should be handled externally.
        
        Mathematically:
            \\dot{x} = f(x, u)

        Args:
            state (jnp.ndarray): observation as returned from the environment.
            control (jnp.ndarray): control input as (steering_angle, speed)
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            jnp.ndarray: state derivative
        """
        raise NotImplementedError("control method not implemented")
    
    def linearize_around_state(self, state: np.ndarray, control: np.ndarray, params: DynamicsConfig = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics model around a given state and control input. This function computes the state Jacobian and control Jacobian
        at the given state and control input. These Jacobians can be used in model-based controllers.

        Mathematically:
            A = df_dx(x, u)
            B = df_du(x, u)

        Args:
            state (np.ndarray): observation as returned from the environment.
            control (np.ndarray): control input as (steering_angle, speed)
            params (DynamicsConfig): vehicle dynamics parameters, overwrites self.params if not None

        Returns:
            tuple[np.ndarray, np.ndarray]: state Jacobian, control Jacobian
        """
        raise NotImplementedError("linearize_around_state method not implemented")
    
    def parameters_vector_from_config(self, params: DynamicsConfig) -> np.ndarray:
        """
        Convert the dynamics configuration parameters into a vector format. This function is useful for optimization problems where the
        parameters need to be passed as a vector.

        Args:
            params (DynamicsConfig): vehicle dynamics parameters

        Returns:
            np.ndarray: (num_params, 1) vector of parameters
        """
        raise NotImplementedError("parameters_vector_from_config method not implemented")
    
    def config_from_parameters_vector(self, params: np.ndarray) -> DynamicsConfig:
        """
        Convert a vector of parameters into a dynamics configuration object. This function is useful for optimization problems where the
        parameters need to be passed as a vector.

        Args:
            params (np.ndarray): (num_params, 1) vector of parameters

        Returns:
            dynamics_config: vehicle dynamics parameters
        """
        raise NotImplementedError("config_from_parameters_vector method not implemented")

    @property
    def num_params(self) -> int:
        """
        Get the number of parameters from dynamics_config that are actively used in the model.
        This is useful for determining the size of the parameter vector in optimization problems.
        """
        raise NotImplementedError("num_params method not implemented")

    @property
    def params(self) -> DynamicsConfig:
        """
        Get the dynamics configuration parameters.
        """
        return self._params

    @params.setter
    def params(self, value: DynamicsConfig) -> None:
        """
        Set the dynamics configuration parameters without updating nx and nu fields.
        """
        assert isinstance(value, DynamicsConfig), f"Expected DynamicsConfig, got {type(value)}"
        self._params = value

