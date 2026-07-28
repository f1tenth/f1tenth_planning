import os
import jax
import jax.numpy as jnp
from pathlib import Path
from functools import partial

from f1tenth_planning.control.config.controller_config import MPPIConfig
from f1tenth_planning.control.discretizers import rk4_discretization
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.mpc_solver import MPCSolver

jax_cache_dir = Path.home() / "jax_cache"
jax_cache_dir.mkdir(exist_ok=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))


def truncated_gaussian_sampler(key, mean, low, high, cov):
    """
    Multivariate truncated Gaussian sampler using Cholesky decomposition.
    Generates samples from a truncated Gaussian distribution with given mean, covariance, and bounds.

    Parameters:
      key (jax.random.PRNGKey): Random key for sampling
      mean (numpy.ndarray): Mean of the distribution
      low (numpy.ndarray): Lower bounds for each dimension
      high (numpy.ndarray): Upper bounds for each dimension
      cov (numpy.ndarray): Covariance matrix (optional)
    Returns:
      numpy.ndarray: One sample from the truncated Gaussian distribution

    """
    R = jnp.linalg.cholesky(cov)

    # Adjust the bounds for the truncated normal distribution
    adjusted_low = (low - mean) / jnp.diag(R)
    adjusted_high = (high - mean) / jnp.diag(R)

    # Generate truncated standard normal samples
    samples = jax.random.truncated_normal(
        key,
        lower=adjusted_low,
        upper=adjusted_high,
    )

    # Transform back to original space
    return mean + R @ samples


class MPPISolver(MPCSolver):
    """
    Path-tracking Model Predictive Path Integral (MPPI) controller.
    paper: https://arxiv.org/pdf/1707.02342 | base code: https://github.com/google-research/google-research/tree/master/jax_mpc

    Args:
        config (MPPIConfig): MPPI configuration object, contains MPPI costs and constraints
        model (DynamicsModel): dynamics model object, used to compute the state derivative
    """

    # Rolls out sampled trajectories under jax.vmap/jit.
    REQUIRED_BACKENDS = ("jax",)

    def __init__(
        self,
        config: MPPIConfig,
        model: DynamicsModel,
        discretizer=rk4_discretization,
        step_function=None,
        reward_function=None,
    ) -> None:
        """
        Initialize the MPPI solver.
        Args:
            config (MPPIConfig): MPPI configuration object, contains MPPI costs and constraints
            model (DynamicsModel): dynamics model object, used to compute the state derivative
            discretizer (function, optional): function to discretize the continuous-time dynamics. Defaults to rk4_discretization.
            step_function (function, optional): function of the form _step(self, x, u, p) to compute the next state given current state and control input. If None, uses the discretizer with model's f_jax
            reward_function (function, optional): function of the form _reward(self, x, u, x_ref, Q, R) to compute the reward given current state, control input, reference state, Q, and R. If None, uses the default quadratic cost
        Returns:
            None
        """
        super().__init__(config, model)
        self.config: MPPIConfig = self.config  # For type hinting
        self.discretizer = discretizer
        if step_function is not None:
            self._step = step_function
        if reward_function is not None:
            self._reward = reward_function
        self.control_params = self._init_control()  # [N, nu]
        self.p = self.model.parameters_vector_from_config(self.model.params)
        self.nu_eye = jnp.eye(self.config.nu)  # [nu, nu]
        self.nu_zeros = jnp.zeros((self.config.nu,))  # [nu]
        self.samples = None  # (a_sampled, s_sampled, r_sampled); set on first solve()
        # Persist the PRNG key across solve() calls so exploration noise is
        # independent each step instead of resetting to the same seed every solve.
        self.rng = jax.random.PRNGKey(0)

    def _init_control(self):
        """
        Initialize the control parameters for MPPI.

        Returns:
            tuple: (a_opt, a_cov) where a_opt is the optimal control input and a_cov is the covariance matrix.
        """

        a_opt = jnp.zeros((self.config.N, self.config.nu))  # [N, nu]
        # a_cov: [N, nu, nu]
        if self.config.adaptive_covariance:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            a_cov = (self.config.u_std**2) * jnp.tile(
                jnp.eye(jnp.array(self.config.nu)), (self.config.N, 1, 1)
            )
        else:
            a_cov = None
        return (a_opt, a_cov)

    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, input_, env_state, ref_traj, p, Q, R):
        a_opt, a_cov, rng = input_
        rng_da, rng = jax.random.split(rng)
        # TODO: FLAG: Check if this is correct
        adjusted_lower = self.config.u_min - a_opt
        adjusted_upper = self.config.u_max - a_opt
        # TODO: Find a way to use the covariance matrix
        da = jax.random.truncated_normal(
            rng_da,
            lower=adjusted_lower,
            upper=adjusted_upper,
            shape=(self.config.n_samples, self.config.N, self.config.nu),
        )
        a = a_opt + da  # [n_samples, N, nu]
        a = jnp.clip(a, self.config.u_min, self.config.u_max)  # [n_samples, N, nu]

        s, r = jax.vmap(self._rollout, in_axes=(0, None, None, None, None, None))(
            a, env_state, ref_traj, p, Q, R
        )  # [n_samples, N]
        R = jax.vmap(self._returns)(r)  # [n_samples, N], pylint: disable=invalid-name
        w = jax.vmap(self._weights, 1, 1)(R)  # [n_samples, N]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [N, nu]
        a_opt = a_opt + da_opt  # [N, nu]
        if self.config.adaptive_covariance:
            a_cov = jax.vmap(jax.vmap(jnp.outer))(da, da)  # [n_samples, N, nu, nu]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(
                a_cov, 0, w
            )  # a_cov: [N, nu, nu]
            # prevent loss of rank when one sample is heavily weighted
            a_cov = a_cov + self.nu_eye * 0.00001
        return (a_opt, a_cov, rng), (a, s, r)

    def _step(self, x, u, p):
        """
        Single-step state prediction function.
        """
        return self.discretizer(self.model.f_jax, x, u, p, self.config.dt)

    def _reward(self, x, u, x_ref, Q, R):
        """
        Single-step reward calculated as the negative of the trajectory tracking error (x^T Q x + u^T R u).
        """
        return -(
            jnp.dot((x - x_ref).T, jnp.dot(Q, (x - x_ref)))
            + jnp.dot(u.T, jnp.dot(R, u))
        )

    def update(self, x0, ref_traj, p=None, Q=None, R=None):
        """
        Update the parameters of the MPPI solver for the next solve iteration.
        Optionally, custom dynamics parameters and cost matrices can be provided.
        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            p (np.ndarray, optional): custom dynamics parameters vector. If None, uses default.
            Q (np.ndarray, optional): custom state cost matrix. If None, uses default.
            R (np.ndarray, optional): custom control input cost matrix. If None, uses default
        Returns:
            None
        """
        super().update(x0, ref_traj, p=p, Q=Q, R=R)
        return

    def _returns(self, r):
        # r: [N]
        return jnp.dot(jnp.triu(jnp.ones((self.config.N, self.config.N))), r)  # R: [N]

    def _weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.config.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.config.temperature)  # [n_samples] np.float32
        w = w / jnp.sum(w)  # [n_samples] np.float32
        return w

    @partial(jax.jit, static_argnums=(0))
    def _rollout(self, u, x0, xref, p, Q, R):
        """
        Rollout the trajectory given the control inputs and initial state.

        Args:
            u (np.ndarray): control inputs of shape (N, nu)
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (N+1, nx)
        Returns:
            np.ndarray: state trajectory of shape (N+1, nx)
            np.ndarray: reward trajectory of shape (N+1,)
        """

        def rollout_step(x, u):
            state, ind = x
            u = jnp.reshape(u, (self.config.nu,))
            state = self._step(state, u, p)
            r = self._reward(state, u, xref[:, ind + 1], Q, R)
            x = (state, ind + 1)
            return x, (x, r)

        if not self.config.scan:
            # python equivalent of lax.scan
            scan_output = []
            for t in range(self.config.N):
                x0, output = rollout_step((x0, t), u[t, :])
                x0 = x0[0]
                scan_output.append(output)
            s, r = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
            s = s[0]
        else:
            state_and_index_init = (x0, 0)
            _, (state_and_index, r) = jax.lax.scan(
                rollout_step, state_and_index_init, u
            )
            s = state_and_index[0]

        return (s, r)

    def solve(self, x0, ref_traj, vis=True, p=None, Q=None, R=None):
        """
        Solve the MPPI problem for the given initial state and reference trajectory.
        WARNING: Returned arrays are on the GPU, use jax.device_get() to get them on the CPU.

        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)

        Returns:
            np.ndarray: optimal state trajectory of shape (nx, N+1)
            np.ndarray: optimal control input of shape (nu, N)
        """
        # Update the parameters of the optimization problem
        super().update(x0, ref_traj, p=p, Q=Q, R=R)

        # Run MPPI iterations (continue the PRNG stream from the previous solve)
        rng = self.rng
        jax_x0 = jnp.array(x0)
        jax_ref = jnp.array(ref_traj)
        a_opt, a_cov = self.control_params
        a_opt = jnp.concatenate(
            [a_opt[1:, :], jnp.expand_dims(self.nu_zeros, axis=0)]
        )  # [N, nu]
        if self.config.adaptive_covariance:
            a_cov = jnp.concatenate(
                [
                    a_cov[1:, :],
                    jnp.expand_dims((self.config.u_std**2) * self.nu_eye, axis=0),
                ]
            )
        if not self.config.scan or self.config.n_iterations == 1:
            for _ in range(self.config.n_iterations):
                (a_opt, a_cov, rng), (a_sampled, s_sampled, r_sampled) = (
                    self.iteration_step(
                        (a_opt, a_cov, rng),
                        jax_x0,
                        jax_ref,
                        self.p,
                        self.config.Q,
                        self.config.R,
                    )
                )
        else:
            (a_opt, a_cov, rng), (a_sampled, s_sampled, r_sampled) = jax.lax.scan(
                lambda carry, _: self.iteration_step(
                    carry, jax_x0, jax_ref, self.p, self.config.Q, self.config.R
                ),
                (a_opt, a_cov, rng),
                None,
                length=self.config.n_iterations,
            )
        # persist the evolved PRNG key for the next solve()
        self.rng = rng
        self.control_params, self.samples = (
            (a_opt, a_cov),
            (a_sampled, s_sampled, r_sampled),
        )

        # Get the solved for controls
        self.uk = self.control_params[0]  # [N, nu]

        # Optionally rollout the optimal trajectory for visualization
        if vis:
            self.xk, _ = self._rollout(
                self.uk, x0, jax_ref, self.p, self.config.Q, self.config.R
            )  # [N, nu]
            self.xk = jnp.concatenate([jnp.expand_dims(x0, axis=0), self.xk], axis=0)

            # Make sure xk and uk are in the right shape
            self.xk = jnp.transpose(self.xk)  # [nx, N+1]
        else:
            self.xk = jnp.zeros((self.config.nx, self.config.N + 1))  # [nx, N+1]
        self.uk = jnp.transpose(self.uk)  # [nu, N]
        return self.xk, self.uk
