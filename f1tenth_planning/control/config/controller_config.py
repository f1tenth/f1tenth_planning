from dataclasses import dataclass, field
import numpy as np


@dataclass
class mpc_config:
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

    N: int
    dt: float

    nx: int
    nu: int

    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray

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
class mppi_config:
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

    nx: int
    nu: int
    N: int
    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray
    dt: float

    # Control limits
    u_min: np.ndarray = field(default=None)
    u_max: np.ndarray = field(default=None)

    # MPPI specific parameters
    n_iterations: int = field(default=5)
    n_samples: int = field(default=16)
    temperature: float = field(default=0.01)
    damping: float = field(default=0.001)
    u_std: float = field(default=0.5)  # std of the control noise
    scan: bool = field(default=True)
    adaptive_covariance: bool = field(default=False)

    def __post_init__(self):
        # Default u_min, u_max to infinities if not provided
        if self.u_min is None:
            self.u_min = -np.inf * np.ones(self.nu)
        if self.u_max is None:
            self.u_max = np.inf * np.ones(self.nu)

        # Check that the dimensions of nx and nu are consistent with Q, Qf, R
        assert self.Q.shape == (self.nx, self.nx), "Q matrix has incorrect dimensions"
        assert self.R.shape == (self.nu, self.nu), "R matrix has incorrect dimensions"
        assert self.P.shape == (self.nx, self.nx), "P matrix has incorrect dimensions"
        assert self.Rd.shape == (self.nu, self.nu), "Rd matrix has incorrect dimensions"
        assert self.u_min.shape == (self.nu,), "u_min has incorrect dimensions"
        assert self.u_max.shape == (self.nu,), "u_max has incorrect dimensions"


def kinematic_mpc_config():
    # [x, y, delta, v, yaw]
    return mpc_config(
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
    return mpc_config(
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
    return mppi_config(
        nx=7,
        nu=2,
        N=10,
        Q=np.diag([5.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        R=np.diag([0.0, 0.00]),
        Rd=np.diag([0.0, 0.00]),
        P=np.diag([25.0, 25.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        dt=0.1,
        n_iterations=5,
        n_samples=512,
        adaptive_covariance=True,
        scan=False,
    )


@dataclass
class lqr_config:
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
        self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])
        self.R = np.array([[0.75]])
        self.max_iterations = 50
        self.eps = 0.01
        self.dt = 0.01
