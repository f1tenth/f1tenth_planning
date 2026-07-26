from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np

from f1tenth_planning.control.config.model_config import ModelConfig


@dataclass
class MPCConfig:
    """
    Configuration for the MPC controller. Includes the following parameters:

    Args:
        nx (int): Number of states.
        nu (int): Number of control inputs.
        N (int): Planning horizon for the MPC controller.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control input cost matrix.
        Rd (np.ndarray): Control input derivative cost matrix (action rate cost).
        P (np.ndarray): Terminal cost matrix.
        dt (float): Time discretization interval.
    """

    # Horizon and time step
    N: int
    dt: float

    # Dimensions
    nx: int
    nu: int

    # Cost matrices
    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray

    # Constraints (state and input bounds)
    x_min: np.ndarray = field(default=None)
    x_max: np.ndarray = field(default=None)
    u_min: np.ndarray = field(default=None)
    u_max: np.ndarray = field(default=None)
    ud_min: np.ndarray = field(default=None)
    ud_max: np.ndarray = field(default=None)

    def __post_init__(self):
        # Default x_min, x_max, u_min, u_max to infinities if not provided
        if self.x_min is None:
            self.x_min = -np.inf * np.ones(self.nx)
        if self.x_max is None:
            self.x_max = np.inf * np.ones(self.nx)
        if self.u_min is None:
            self.u_min = -np.inf * np.ones(self.nu)
        if self.u_max is None:
            self.u_max = np.inf * np.ones(self.nu)
        if self.ud_min is None:
            self.ud_min = -np.inf * np.ones(self.nu)
        if self.ud_max is None:
            self.ud_max = np.inf * np.ones(self.nu)

        # Check that the dimensions of nx and nu are consistent with Q, Qf, R
        assert self.Q.shape == (self.nx, self.nx), "Q matrix has incorrect dimensions"
        assert self.R.shape == (self.nu, self.nu), "R matrix has incorrect dimensions"
        assert self.P.shape == (self.nx, self.nx), "P matrix has incorrect dimensions"
        assert self.Rd.shape == (self.nu, self.nu), "Rd matrix has incorrect dimensions"
        assert self.x_min.shape == (self.nx,), "x_min has incorrect dimensions"
        assert self.x_max.shape == (self.nx,), "x_max has incorrect dimensions"
        assert self.u_min.shape == (self.nu,), "u_min has incorrect dimensions"
        assert self.u_max.shape == (self.nu,), "u_max has incorrect dimensions"
        assert self.ud_min.shape == (self.nu,), "ud_min has incorrect dimensions"
        assert self.ud_max.shape == (self.nu,), "ud_max has incorrect dimensions"


@dataclass
class MPPIConfig(MPCConfig):
    """
    Configuration for the MPPI controller, inheriting from MPCConfig and adding MPPI-specific parameters.

    Args:
        n_iterations (int): Number of iterations for the MPPI solver.
        n_samples (int): Number of samples for the MPPI solver.
        temperature (float): Temperature for the MPPI solver.
        damping (float): Damping for the MPPI solver.
        u_std (float): Standard deviation of the control noise.
        scan (bool): Whether to scan the control space.
        adaptive_covariance (bool): Whether to adapt the covariance matrix.
    """

    # MPPI specific parameters
    n_iterations: int = field(default=5)
    n_samples: int = field(default=16)
    temperature: float = field(default=0.01)
    damping: float = field(default=0.001)
    u_std: float = field(default=0.5)  # std of the control noise
    scan: bool = field(default=True)
    adaptive_covariance: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        # No additional checks needed for MPPI-specific fields


def kinematic_mpc_config():
    # [x, y, delta, v, yaw]
    return MPCConfig(
        nx=5,
        nu=2,
        N=15,
        Q=np.diag([18.0, 18.0, 0.0, 1.2, 18.0]),
        R=np.diag([0.01, 0.4]),
        Rd=np.diag([0.002, 0.01]),
        P=np.diag([18.0, 18.0, 0.0, 1.2, 18.0]),
        dt=0.1,
    )


def dynamic_mpc_config():
    # [x, y, delta, v, yaw, yaw_rate, beta]
    return MPCConfig(
        nx=7,
        nu=2,
        N=15,
        Q=np.diag([25.0, 25.0, 0.0, 7.0, 1000.0, 0.0, 100.0]),
        R=np.diag([0.01, 0.4]),
        Rd=np.diag([0.002, 0.01]),
        P=np.diag([25.0, 25.0, 0.0, 7.0, 1000.0, 0.0, 100.0]),
        dt=0.1,
    )


def dynamic_mppi_config():
    # [x, y, delta, v, yaw, yaw_rate, beta]
    return MPPIConfig(
        nx=7,
        nu=2,
        N=10,
        Q=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        R=np.diag([0.0, 0.00]),
        Rd=np.diag([0.0, 0.00]),
        P=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        dt=0.1,
        n_iterations=2,
        n_samples=1024,
        adaptive_covariance=True,
        scan=False,
    )


@dataclass
class LQRConfig:
    """
    Configuration for the LQR controller. Includes the following parameters:

    Args:
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control input cost matrix."
        max_iterations (int): Maximum number of iterations for the LQR solver.
        eps (float): Tolerance for convergence.
        dt (float): Time discretization interval.
    """

    Q: np.ndarray = field(default=None)
    R: np.ndarray = field(default=None)
    max_iterations: int = None
    eps: float = None
    dt: float = None

    def __post_init__(self):
        # Fill only unset fields so caller-provided values are respected
        # (previously every field was overwritten, making LQRConfig inert).
        if self.Q is None:
            self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])
        if self.R is None:
            self.R = np.array([[0.75]])
        if self.max_iterations is None:
            self.max_iterations = 50
        if self.eps is None:
            self.eps = 0.01
        if self.dt is None:
            self.dt = 0.01


@dataclass
class LMPCConfig:
    """
    Configuration for LMPC algorithm-level parameters.
    """

    N: int = 10
    dt: float = 0.1
    n_iterations: int = 1
    ss_size: int = 5
    retrain_every_lap: bool = True
    lap_extend_sec: float = 2.0


@dataclass
class APMPPIConfig(MPPIConfig):
    """
    Configuration for AP-MPPI solver (sampling, penalty multipliers, etc.).

    Args:
        n_lambdas (int): Number of lambda penalty multipliers to sample per constraint.
        lambdas_sample_range (np.ndarray): Range of the lambda penalty multipliers of size (n_constraints, 2).
        constraints (List[Callable]): List of constraint functions. Each function should have signature
            constraint(x, u) -> (N,) where positive values indicate violation.
    """

    n_lambdas: int = field(default=16)
    lambdas_sample_range: np.ndarray = field(default=None)
    constraints: List[Callable] = field(default_factory=list)
    n_constraints: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        self.n_constraints = len(self.constraints)
        # Default lambdas_sample_range if not provided
        if self.lambdas_sample_range is None:
            if self.n_constraints > 0:
                self.lambdas_sample_range = np.tile(np.array([[0.0, 5.0]]), (self.n_constraints, 1))
            else:
                self.lambdas_sample_range = np.zeros((0, 2))
        if self.n_constraints > 0:
            assert self.lambdas_sample_range.shape == (self.n_constraints, 2), (
                f"lambdas_sample_range has incorrect dimensions: {self.lambdas_sample_range.shape} != ({self.n_constraints}, 2)"
            )


def dynamic_ap_mppi_config(
    constraints: List[Callable] = None,
    lambdas_sample_range: np.ndarray = None,
    n_lambdas: int = 16,
):
    """
    Create a default AP-MPPI config for dynamic bicycle model.

    Args:
        constraints: List of constraint functions. Each should have signature
            constraint(x, u) -> (N,) where positive values indicate violation.
        lambdas_sample_range: Range for lambda sampling, shape (n_constraints, 2).
        n_lambdas: Number of lambda samples per constraint.

    Returns:
        APMPPIConfig configured for dynamic bicycle model.
    """
    if constraints is None:
        constraints = []
    # [x, y, delta, v, yaw, yaw_rate, beta]
    return APMPPIConfig(
        nx=7,
        nu=2,
        N=10,
        Q=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        R=np.diag([0.0, 0.00]),
        Rd=np.diag([0.0, 0.00]),
        P=np.diag([5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        dt=0.1,
        n_iterations=2,
        n_samples=1024,
        adaptive_covariance=True,
        scan=False,
        n_lambdas=n_lambdas,
        lambdas_sample_range=lambdas_sample_range,
        constraints=constraints,
    )


@dataclass
class SITLMPCConfig:
    """
    Combined configuration for Safe-MPPI + LMPC (IT-LMPC) controller.
    """

    lmpc: LMPCConfig = field(default_factory=LMPCConfig)
    ap_mppi: APMPPIConfig = field(default_factory=APMPPIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
