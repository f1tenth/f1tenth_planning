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


# ===========================================================================
# Jitted kernels (DESIGN.md §7.4)
#
# These are module-level functions taking everything explicitly, rather than
# methods with `self` marked static. The split is:
#
#   static  -- values that determine the SHAPE of the computation:
#              N, n_samples, nu, the scan flag, and the step/reward callables.
#   traced  -- numbers that flow THROUGH it: costs, bounds, dynamics params,
#              temperature, damping, dt.
#
# A tuning value read off a static `self` is baked into the trace at first call
# and can never change again -- that is the bug this structure removes. Note
# `static_argnames` rather than `static_argnums`: a positional index list
# silently desynchronises from a long signature.
# ===========================================================================


@partial(jax.jit, static_argnames=("N", "nu", "scan", "step_fn", "reward_fn"))
def _rollout_kernel(
    u, x0, xref, p, Q, R, dt, *, N, nu, scan, step_fn, reward_fn
):
    """Roll a single control sequence forward and score it.

    Args:
        u: (N, nu) control sequence.
        x0: (nx,) initial state.
        xref: (nx, N+1) reference trajectory.
        p: dynamics parameter vector.
        Q, R: state and control cost matrices.
        dt: integration step.

    Returns:
        (s, r): (N, nx) state trajectory and (N,) per-step reward.
    """

    def rollout_step(carry, u_t):
        state, ind = carry
        u_t = jnp.reshape(u_t, (nu,))
        state = step_fn(state, u_t, p, dt)
        r = reward_fn(state, u_t, xref[:, ind + 1], Q, R)
        return (state, ind + 1), ((state, ind + 1), r)

    if not scan:
        # python equivalent of lax.scan
        scan_output = []
        carry = x0
        for t in range(N):
            carry, output = rollout_step((carry, t), u[t, :])
            carry = carry[0]
            scan_output.append(output)
        s, r = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
        s = s[0]
    else:
        _, (state_and_index, r) = jax.lax.scan(rollout_step, (x0, 0), u)
        s = state_and_index[0]

    return (s, r)


def _sample_perturbations(rng, a_opt, u_min, u_max, u_std, n_samples, N, nu):
    """Sample control perturbations ~ N(0, u_std^2), truncated to the control bounds.

    `jax.random.truncated_normal` draws from a *standard* normal truncated to
    [lower, upper] -- it takes no scale argument. Scaling the bounds by u_std before
    the draw and the samples by u_std after is what actually makes u_std the sampling
    standard deviation; without it the exploration std is ~1.0 in every control
    dimension no matter what the config says.

    Truncating the *perturbation* against the current nominal also guarantees
    a_opt + da stays inside the control bounds.
    """
    scale = jnp.maximum(jnp.asarray(u_std, dtype=jnp.float32), 1e-6)
    lower = (u_min - a_opt) / scale
    upper = (u_max - a_opt) / scale
    return scale * jax.random.truncated_normal(
        rng, lower=lower, upper=upper, shape=(n_samples, N, nu)
    )


@partial(jax.jit, static_argnames=("N",))
def _returns_kernel(r, *, N):
    """Reward-to-go: R[i] = sum_{j>=i} r[j]."""
    return jnp.dot(jnp.triu(jnp.ones((N, N))), r)


@jax.jit
def _weights_kernel(returns, temperature, damping):
    """Softmax weights over samples.

    `temperature` and `damping` are **traced** -- they are the tuning knobs that
    have to be changeable at runtime.
    """
    standardized = (returns - jnp.max(returns)) / (
        (jnp.max(returns) - jnp.min(returns)) + damping
    )
    w = jnp.exp(standardized / temperature)
    return w / jnp.sum(w)


@partial(
    jax.jit,
    static_argnames=(
        "N", "n_samples", "nu", "scan", "adaptive_cov", "step_fn", "reward_fn",
    ),
)
def _iteration_kernel(
    a_opt, a_cov, rng,                                  # carry   (traced)
    x0, ref_traj, p, Q, R,                              # problem (traced)
    u_min, u_max, temperature, damping, dt, u_std,      # tuning  (traced)
    *,
    N, n_samples, nu, scan, adaptive_cov, step_fn, reward_fn,   # structural (static)
):
    """One MPPI iteration: sample, roll out, weight, update the nominal control."""
    rng_da, rng = jax.random.split(rng)

    da = _sample_perturbations(rng_da, a_opt, u_min, u_max, u_std, n_samples, N, nu)
    a = jnp.clip(a_opt + da, u_min, u_max)  # [n_samples, N, nu]

    rollout = partial(
        _rollout_kernel, N=N, nu=nu, scan=scan, step_fn=step_fn, reward_fn=reward_fn
    )
    s, r = jax.vmap(rollout, in_axes=(0, None, None, None, None, None, None))(
        a, x0, ref_traj, p, Q, R, dt
    )  # s: [n_samples, N, nx], r: [n_samples, N]

    returns = jax.vmap(partial(_returns_kernel, N=N))(r)  # [n_samples, N]
    w = jax.vmap(partial(_weights_kernel, temperature=temperature, damping=damping),
                 1, 1)(returns)  # [n_samples, N]

    a_opt = a_opt + jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [N, nu]

    if adaptive_cov:
        a_cov = jax.vmap(jax.vmap(jnp.outer))(da, da)  # [n_samples, N, nu, nu]
        a_cov = jax.vmap(jnp.average, (1, None, 1))(a_cov, 0, w)  # [N, nu, nu]
        # prevent loss of rank when one sample is heavily weighted
        a_cov = a_cov + jnp.eye(nu) * 1e-5

    return (a_opt, a_cov, rng), (a, s, r)


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
            step_function (function, optional): ``step(x, u, p, dt) -> x_next``. Use this
                for a model that predicts the next state directly instead of a
                derivative. If None, the discretizer is applied to the model's f_jax.
            reward_function (function, optional): ``reward(x, u, x_ref, Q, R) -> float``.
                If None, uses the default quadratic cost.

        Note both callables are **static** to the jitted kernels, so they are built
        once here and their identity must stay stable; do not rebuild them per step.
        """
        super().__init__(config, model)
        self.config: MPPIConfig = self.config  # For type hinting
        self.discretizer = discretizer
        self._step_fn = step_function if step_function is not None else self._make_step()
        self._reward_fn = (
            reward_function if reward_function is not None else _default_reward
        )
        self.control_params = self._init_control()  # [N, nu]
        self.p = self.model.parameters_vector_from_config(self.model.params)
        self.nu_eye = jnp.eye(self.config.nu)  # [nu, nu]
        self.nu_zeros = jnp.zeros((self.config.nu,))  # [nu]
        self.samples = None  # (a_sampled, s_sampled, r_sampled); set on first solve()
        # Persist the PRNG key across solve() calls so exploration noise is
        # independent each step instead of resetting to the same seed every solve.
        self.rng = jax.random.PRNGKey(0)

    def _make_step(self):
        """Build the default step function once, with a stable identity.

        `self._step` as a bound method would be a fresh object on every access, and
        because the kernels take it as a *static* argument that would retrace on
        every call.
        """
        model_f = self.model.f_jax
        discretizer = self.discretizer

        def step(x, u, p, dt):
            return discretizer(model_f, x, u, p, dt)

        return step

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

    def _static_kwargs(self):
        """The structural (compile-time) arguments for the jitted kernels."""
        return dict(
            N=self.config.N,
            n_samples=self.config.n_samples,
            nu=self.config.nu,
            scan=self.config.scan,
            adaptive_cov=self.config.adaptive_covariance,
            step_fn=self._step_fn,
            reward_fn=self._reward_fn,
        )

    def iteration_step(self, carry, x0, ref_traj, p, Q, R):
        """One MPPI iteration. Thin wrapper that routes config to the kernel."""
        a_opt, a_cov, rng = carry
        return _iteration_kernel(
            a_opt, a_cov, rng,
            x0, ref_traj, p, Q, R,
            self.config.u_min, self.config.u_max,
            self.config.temperature, self.config.damping, self.config.dt,
            self.config.u_std,
            **self._static_kwargs(),
        )

    def _rollout(self, u, x0, xref, p, Q, R):
        """Roll out a single control sequence (used for the visualised solution)."""
        return _rollout_kernel(
            u, x0, xref, p, Q, R, self.config.dt,
            N=self.config.N, nu=self.config.nu, scan=self.config.scan,
            step_fn=self._step_fn, reward_fn=self._reward_fn,
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


def _default_reward(x, u, x_ref, Q, R):
    """Negative quadratic tracking cost: -( (x-xref)' Q (x-xref) + u' R u )."""
    return -(
        jnp.dot((x - x_ref).T, jnp.dot(Q, (x - x_ref))) + jnp.dot(u.T, jnp.dot(R, u))
    )
