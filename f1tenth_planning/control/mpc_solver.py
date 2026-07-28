from __future__ import annotations
from abc import abstractmethod, ABC
    
from f1tenth_planning.control.config.controller_config import MPCConfig
from f1tenth_planning.control.dynamics_model import DynamicsModel

class MPCSolver(ABC):
    """
    Abstract base class for Model Predictive Control (MPC) solvers. Implemented MPC solvers should inherit from this class and implement all abstract methods for plug-and-play compatibility with MPC controllers.

    Subclasses declare which model backends they need via :attr:`REQUIRED_BACKENDS`;
    the requirement is checked at construction so an incompatible model+solver pair
    fails immediately with a clear message instead of raising ``NotImplementedError``
    from deep inside a solve (DESIGN.md §5.1).
    """

    #: Model backends this solver needs, e.g. ``("casadi",)`` or ``("numpy", "jacobian")``.
    REQUIRED_BACKENDS: tuple[str, ...] = ()

    @abstractmethod
    def __init__(self, config: MPCConfig, model: DynamicsModel):
        self._require_backends(model)
        self.config = config
        self.model = model

    @classmethod
    def _require_backends(cls, model: DynamicsModel) -> None:
        """Fail fast if `model` cannot supply what this solver needs."""
        if not cls.REQUIRED_BACKENDS:
            return
        missing = sorted(set(cls.REQUIRED_BACKENDS) - model.backends())
        if missing:
            raise TypeError(
                f"{cls.__name__} requires model backend(s) {missing}, but "
                f"{type(model).__name__} only provides {sorted(model.backends())}. "
                f"Either use a different model or implement the missing method(s)."
            )

    @abstractmethod
    def update(self, x0, ref_traj, p=None, Q=None, R=None, P=None, Rd=None):
        """
        Update the parameters of the MPC solver for the next solve iteration.
        Optionally, custom dynamics parameters and cost matrices can be provided.
        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            p (np.ndarray, optional): custom dynamics parameters vector. If None, uses default.
            Q (np.ndarray, optional): custom state cost matrix. If None, uses default.
            R (np.ndarray, optional): custom control input cost matrix. If None, uses default
            P (np.ndarray, optional): custom terminal cost matrix. If None, uses default
            Rd (np.ndarray, optional): custom control input derivative cost matrix. If None, uses default
        Returns:
            None
        """
        self.p = p if p is not None else self.p
        self.config.Q = Q if Q is not None else self.config.Q
        self.config.R = R if R is not None else self.config.R
        self.config.P = P if P is not None else self.config.P
        self.config.Rd = Rd if Rd is not None else self.config.Rd
        return 

    @abstractmethod
    def solve(self, x0, ref_traj, p=None, Q=None, R=None):
        """
        Solve the MPC problem for the given initial state and reference trajectory.
        Optionally, custom dynamics parameters and cost matrices can be provided.

        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            p (np.ndarray, optional): custom dynamics parameters vector. If None, uses default.
            Q (np.ndarray, optional): custom state cost matrix. If None, uses default.
            R (np.ndarray, optional): custom control input cost matrix. If None, uses default

        Returns:
            np.ndarray: optimal state trajectory of shape (nx, N+1)
            np.ndarray: optimal control input of shape (nu, N)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
