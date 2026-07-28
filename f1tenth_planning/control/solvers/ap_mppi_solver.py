import jax
import jax.numpy as jnp
from functools import partial

from f1tenth_planning.control.config.controller_config import APMPPIConfig
from f1tenth_planning.control.discretizers import rk4_discretization
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.solvers.mppi_solver import (
    MPPISolver,
    _returns_kernel,
    _rollout_kernel,
    _sample_perturbations,
    _weights_kernel,
)


@partial(
    jax.jit,
    static_argnames=(
        "N", "n_samples", "nu", "scan", "adaptive_cov",
        "step_fn", "reward_fn", "constraints_fn",
    ),
)
def _ap_iteration_kernel(
    a_opt, a_cov, rng,                                  # carry   (traced)
    x0, ref_traj, p, Q, R,                              # problem (traced)
    u_min, u_max, temperature, damping, dt, u_std, lambdas,   # tuning  (traced)
    *,
    N, n_samples, nu, scan, adaptive_cov,
    step_fn, reward_fn, constraints_fn,                 # structural (static)
):
    """One AP-MPPI iteration.

    Adaptive Penalty: rather than tuning a single penalty weight, sample a whole
    *population* of weight vectors (`lambdas`), solve the MPPI update once per weight
    in parallel, re-roll each resulting candidate, and pick a winner with a
    feasibility-first rule -- best pure return among feasible candidates if any are
    feasible, otherwise the least-violating one.

    `lambdas` is traced, so the penalty population can be resampled or annealed at
    runtime without recompiling.
    """
    rng_da, rng = jax.random.split(rng)

    da = _sample_perturbations(rng_da, a_opt, u_min, u_max, u_std, n_samples, N, nu)
    a = jnp.clip(a_opt + da, u_min, u_max)  # [n_samples, N, nu]

    rollout = partial(
        _rollout_kernel, N=N, nu=nu, scan=scan, step_fn=step_fn, reward_fn=reward_fn
    )
    s, r = jax.vmap(rollout, in_axes=(0, None, None, None, None, None, None))(
        a, x0, ref_traj, p, Q, R, dt
    )  # s: [n_samples, N, nx], r: [n_samples, N]

    # Constraint costs per sample, then penalised under EVERY lambda at once.
    c = jax.vmap(constraints_fn)(s, a)                       # [n_samples, C, N]
    c_weighted = jnp.einsum("scn,cl->sln", c, lambdas)       # [n_samples, L, N]
    r_modified = r[:, None, :] - c_weighted                  # [n_samples, L, N]

    returns_fn = partial(_returns_kernel, N=N)
    R_modified = jax.vmap(jax.vmap(returns_fn))(r_modified)  # [n_samples, L, N]
    R_for_weights = jnp.transpose(R_modified, (1, 0, 2))     # [L, n_samples, N]

    weights_fn = partial(_weights_kernel, temperature=temperature, damping=damping)
    w_all = jax.vmap(lambda R_l: jax.vmap(weights_fn, 1, 1)(R_l))(
        R_for_weights
    )  # [L, n_samples, N]

    da_candidates = jax.vmap(
        lambda w_l: jax.vmap(jnp.average, (1, None, 1))(da, 0, w_l)
    )(w_all)                                                 # [L, N, nu]
    a_candidates = a_opt + da_candidates                     # [L, N, nu]

    # Re-roll each candidate so it can be scored on its own merits.
    s_candidates, r_candidates = jax.vmap(
        rollout, in_axes=(0, None, None, None, None, None, None)
    )(a_candidates, x0, ref_traj, p, Q, R, dt)

    c_candidates = jax.vmap(constraints_fn)(s_candidates, a_candidates)  # [L, C, N]
    violations = jnp.sum(jnp.maximum(0.0, c_candidates), axis=(1, 2))    # [L]
    pure_returns = jnp.sum(r_candidates, axis=1)                         # [L]

    # Feasibility-first selection, branchless so it stays jittable.
    feasible_mask = (violations == 0).astype(jnp.float32)
    has_any_feasible = jnp.any(violations == 0).astype(jnp.float32)
    feasible_score = feasible_mask * pure_returns + (1.0 - feasible_mask) * (-1e10)
    combined_score = (
        has_any_feasible * feasible_score + (1.0 - has_any_feasible) * (-violations)
    )
    best_idx = jnp.argmax(combined_score)
    a_opt_new = a_candidates[best_idx]  # [N, nu]

    if adaptive_cov:
        w_best = w_all[best_idx]  # [n_samples, N]
        a_cov_new = jax.vmap(jax.vmap(jnp.outer))(da, da)
        a_cov_new = jax.vmap(jnp.average, (1, None, 1))(a_cov_new, 0, w_best)
        a_cov_new = a_cov_new + jnp.eye(nu) * 1e-5
    else:
        a_cov_new = a_cov

    return (a_opt_new, a_cov_new, rng), (a, s, r)


class APMPPISolver(MPPISolver):
    """
    Adaptive-Penalty Model Predictive Path Integral (AP-MPPI) solver.
    paper: https://ieeexplore.ieee.org/document/11260933
    website: https://sites.google.com/view/sit-lmpc/
    base code: https://github.com/mlab-upenn/SIT-LMPC

    Subclasses :class:`MPPISolver`, overriding only the parts that differ: a penalty
    population (`lambdas`), constraint costs, the per-lambda selection rule, and
    optional stability clipping of rollout states.

    Args:
        config (APMPPIConfig): AP-MPPI configuration (costs, bounds, constraints, lambdas)
        model (DynamicsModel): dynamics model object, used to compute the state derivative
        discretizer (function, optional): continuous-to-discrete integrator. Defaults to rk4_discretization.
        step_function (function, optional): ``step(x, u, p, dt) -> x_next``, replacing
            the default discretizer+clip. If None, the default is used.
        reward_function (function, optional): ``reward(x, u, x_ref, Q, R) -> float``.
    """

    # Rolls out sampled trajectories under jax.vmap/jit.
    REQUIRED_BACKENDS = ("jax",)

    def __init__(
        self,
        config: APMPPIConfig,
        model: DynamicsModel,
        discretizer=rk4_discretization,
        step_function=None,
        reward_function=None,
    ) -> None:
        super().__init__(
            config,
            model,
            discretizer=discretizer,
            step_function=step_function,
            reward_function=reward_function,
        )
        self.config: APMPPIConfig = self.config  # For type hinting
        self.lambdas = self._init_lambdas()
        self.constraints_costs = self._init_constraints_costs()

    def _make_step(self):
        """Default step: integrate, then clip for numerical stability.

        The clip uses ``x_clip_min``/``x_clip_max`` -- deliberately NOT the constrained
        bounds. Clipping a quantity that a constraint also polices would make the
        rollouts unable to violate it, silently disabling the adaptive penalty.
        """
        model_f = self.model.f_jax
        discretizer = self.discretizer
        clip_min = jnp.asarray(self.config.x_clip_min)
        clip_max = jnp.asarray(self.config.x_clip_max)

        def step(x, u, p, dt):
            return jnp.clip(discretizer(model_f, x, u, p, dt), clip_min, clip_max)

        return step

    def _init_lambdas(self):
        """
        Initialize the lambda penalty multipliers. Samples n_constraints x n_lambdas over the meshgrid of the constraint ranges.
        Returns:
            np.ndarray: lambda penalty multipliers of shape (n_constraints, n_lambdas).
        """
        key = jax.random.PRNGKey(0)
        key, key_sample = jax.random.split(key)

        low = jnp.array(self.config.lambdas_sample_range[:, 0], dtype=jnp.float32)
        high = jnp.array(self.config.lambdas_sample_range[:, 1], dtype=jnp.float32)

        return jax.random.uniform(
            key_sample,
            (self.config.n_constraints, self.config.n_lambdas),
            dtype=jnp.float32,
            minval=low[:, None],
            maxval=high[:, None],
        )

    def _init_constraints_costs(self):
        """
        Returns a function constraints_costs(x, u) that computes raw constraint costs.
        Shapes:
            x: (N, nx)
            u: (N, nu)
            returns: (C, N) - raw constraint values for each constraint and timestep
        """
        constraints = tuple(self.config.constraints)  # freeze (JIT-friendly)
        N = self.config.N
        C = self.config.n_constraints

        def constraints_costs(x, u):
            if len(constraints) == 0:
                return jnp.zeros((C, N), dtype=x.dtype)
            return jnp.stack([c(x, u) for c in constraints], axis=0)  # (C, N)

        return constraints_costs

    def _static_kwargs(self):
        kwargs = super()._static_kwargs()
        kwargs["constraints_fn"] = self.constraints_costs
        return kwargs

    def iteration_step(self, carry, x0, ref_traj, p, Q, R):
        """One AP-MPPI iteration. Routes config into the jitted kernel."""
        a_opt, a_cov, rng = carry
        return _ap_iteration_kernel(
            a_opt, a_cov, rng,
            x0, ref_traj, p, Q, R,
            self.config.u_min, self.config.u_max,
            self.config.temperature, self.config.damping, self.config.dt,
            self.config.u_std, self.lambdas,
            **self._static_kwargs(),
        )
