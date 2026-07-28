import warnings

import numpy as np
import casadi as ca

from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.config.controller_config import MPCConfig
from f1tenth_planning.control.discretizers import rk4_discretization
from f1tenth_planning.control.mpc_solver import MPCSolver


class NonlinearMPCSolver(MPCSolver):
    """
    NMPC Solver, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
        model (DynamicsModel): dynamics model object, contains the vehicle dynamics
        ipopt_opts (dict, optional): options for the IPOPT solver
    """

    # Builds a CasADi NLP from the symbolic dynamics.
    REQUIRED_BACKENDS = ("casadi",)

    def __init__(
        self,
        config: MPCConfig,
        model: DynamicsModel,
        ipopt_opts: dict = None,
        discretizer=rk4_discretization,
    ) -> None:
        super().__init__(config, model)
        if ipopt_opts is None:
            ipopt_opts = {
                "ipopt": {
                    "print_level": 0,
                    "max_iter": 200,
                    "acceptable_tol": 1e-2,
                    "acceptable_obj_change_tol": 1e-3,
                    "warm_start_init_point": "yes",
                },
                "print_time": 0,
            }
        self.ipopt_opts = ipopt_opts
        self.discretizer = discretizer
        self.init_problem()

    def init_problem(self):
        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym("X", self.config.nx, self.config.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym("U", self.config.nu, self.config.N)

        # coloumn vector for storing initial state and target state
        Params = ca.SX.sym(
            "Params", self.config.nx + self.model.num_params, self.config.N + 1
        )

        # state weights matrix (use the full matrix so off-diagonal coupling in a
        # non-diagonal Q is preserved rather than silently discarded)
        Q = ca.DM(self.config.Q)

        # controls weights matrix (full matrix)
        R = ca.DM(self.config.R)

        # System dynamics function
        f = self.model.f_casadi()

        cost_fn = 0  # cost function
        g = (
            X[:, 0] - Params[: self.config.nx, 0]
        )  # x(0) = x0 constraint in the equation

        # loop over all time steps
        for k in range(self.config.N):
            st = X[:, k]
            con = U[:, k]
            params = Params[self.config.nx :, k]

            # state tracking cost + input cost
            cost_fn = (
                cost_fn
                + (st - Params[: self.config.nx, k]).T
                @ Q
                @ (st - Params[: self.config.nx, k])
                + con.T @ R @ con
            )

            # state dynamics constraint
            st_next = X[:, k + 1]
            st_next_RK4 = self.discretizer(f, st, con, params, self.config.dt)
            g = ca.vertcat(g, st_next - st_next_RK4)

        # terminal cost
        st = X[:, self.config.N]
        cost_fn = cost_fn + (st - Params[: self.config.nx, self.config.N]).T @ Q @ (
            st - Params[: self.config.nx, self.config.N]
        )

        OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp_prob = {"f": cost_fn, "x": OPT_variables, "g": g, "p": Params}

        # Solver initialization, this is the main solver for the NMPC problem which will be called at each time step
        self.nlp_solver = ca.nlpsol("solver", "ipopt", nlp_prob, self.ipopt_opts)

        # Lower and upper bounds for state and control variables
        lbx = ca.vertcat(
            ca.repmat(self.config.x_min, self.config.N + 1, 1),
            ca.repmat(self.config.u_min, self.config.N, 1),
        )
        ubx = ca.vertcat(
            ca.repmat(self.config.x_max, self.config.N + 1, 1),
            ca.repmat(self.config.u_max, self.config.N, 1),
        )

        # lbg is all zeros
        lbg = ca.vertcat(
            ca.DM.zeros((self.config.nx * (self.config.N + 1), 1)),
        )
        ubg = ca.vertcat(
            ca.DM.zeros((self.config.nx * (self.config.N + 1), 1)),
        )

        # store the arguments for the solver, these are updated at each time step
        self.args = {
            "lbg": lbg,  # constraints lower bound
            "ubg": ubg,  # constraints upper bound
            "lbx": lbx,
            "ubx": ubx,
        }
        self.U0 = ca.DM.zeros((self.config.nu, self.config.N))

        return

    def update(self, x0, ref_traj, p=None, Q=None, R=None, P=None, Rd=None):
        super().update(x0, ref_traj, p, Q, R, P, Rd)

    def solve(self, x0, xref, p=None, Q=None, R=None):
        """
        Solve the NMPC problem using CasADi IPOPT solver.
        Args:
            x0 (numpy.ndarray): initial state of shape (1, nx)
            xref (numpy.ndarray): reference trajectory of shape (N+1, nx)
            p (numpy.ndarray, optional): parameters for the dynamics model of shape (num_params, N+1)
            Q (numpy.ndarray, optional): state weights matrix of shape (nx, nx)
            R (numpy.ndarray, optional): control weights matrix of shape (nu, nu)
        """
        if x0.shape != (self.config.nx,):
            raise ValueError(f"x0 must be of shape {(self.config.nx,)}, got {x0.shape}")
        if xref.shape != (self.config.nx, self.config.N + 1):
            raise ValueError(
                f"xref must be of shape {(self.config.nx, self.config.N + 1)}, got {xref.shape}"
            )

        # Only update parameters if they are provided
        if p is not None:
            if p.shape[0] != self.model.num_params:
                raise ValueError(
                    f"Expected {self.model.num_params} parameters, got {p.shape[0]}"
                )
            if p.shape[1] == 1:
                p = np.tile(np.array(p), (1, self.config.N + 1))
            elif p.shape[1] != self.config.N + 1:
                raise ValueError(
                    f"Expected {self.config.N + 1} columns for parameters, got {p.shape[1]}"
                )
        else:
            p = self.model.parameters_vector_from_config(self.model.params)
            p = np.tile(np.array(p), (1, self.config.N + 1))

        if Q is not None:
            if Q.shape != (
                self.config.nx,
                self.config.nx,
            ):
                raise ValueError(
                    f"Q must be of shape {(self.config.nx, self.config.nx)}, got {Q.shape}"
                )
            warnings.warn("Changing Q during solve is not yet implemented; it is ignored.")
        if R is not None:
            if R.shape != (
                self.config.nu,
                self.config.nu,
            ):
                raise ValueError(
                    f"R must be of shape {(self.config.nu, self.config.nu)}, got {R.shape}"
                )
            warnings.warn("Changing R during solve is not yet implemented; it is ignored.")
        self.update(x0, xref, p, Q, R)
        params_x = ca.horzcat(
            x0,  # initial state
            xref[:, 1:],  # reference states
        )
        params_p = ca.DM(p)
        self.args["p"] = ca.vertcat(params_x, params_p)

        # optimization variable current state
        self.args["x0"] = ca.vertcat(
            ca.reshape(
                ca.repmat(x0, self.config.N + 1, 1),
                self.config.nx * (self.config.N + 1),
                1,
            ),
            ca.reshape(self.U0, self.config.nu * self.config.N, 1),
        )
        sol = self.nlp_solver(
            x0=self.args["x0"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            p=self.args["p"],
        )

        # Warn instead of silently returning garbage if IPOPT did not converge.
        stats = self.nlp_solver.stats()
        if not stats.get("success", False):
            warnings.warn(
                f"IPOPT did not converge (return_status={stats.get('return_status')}); "
                "returning the last iterate."
            )

        u_sol = ca.reshape(
            sol["x"][self.config.nx * (self.config.N + 1) :],
            self.config.nu,
            self.config.N,
        )
        x_sol = ca.reshape(
            sol["x"][: self.config.nx * (self.config.N + 1)],
            self.config.nx,
            self.config.N + 1,
        )

        return x_sol, u_sol
