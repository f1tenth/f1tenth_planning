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
        n_ind_search (int): Number of index search iterations.
        dl (float): Spatial discretization length.
        dt (float): Time discretization interval.
    """
    nx: int
    nu: int
    N: int
    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray
    n_ind_search: int
    dl: float
    dt: float
    
def kinematic_mpc_config():
    return mpc_config(
        nx=4,
        nu=2,
        N=10,
        Q=np.diag([18.5, 18.5, 3.5, 0.1]),
        R=np.diag([0.01, 100.0]),
        Rd=np.diag([0.01, 100.0]),
        P=np.diag([18.5, 18.5, 3.5, 0.1]),
        n_ind_search=20,
        dt=0.1,
        dl=0.03
    )
    
def dynamic_mpc_config():
    return mpc_config(
        nx=7,
        nu=0,
        N=5,
        Q=np.diag([18.5, 18.5, 0.0, 1.5, 0.0, 0.0, 0.0]),
        R=np.diag([0.5, 4.0]),
        Rd=np.diag([0.3, 4.0]),
        P=np.diag([18.5, 18.5, 0.0, 1.5, 0.0, 0.0, 0.0]),
        n_ind_search=20,
        dt=0.1,
        dl=0.03
    )

