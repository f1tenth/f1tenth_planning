"""Runtime configurability and JIT trace behaviour (DESIGN.md §7).

Every controller parameter must be settable before init AND at runtime. For the
jax solvers that means tuning values have to arrive as *traced arguments*; anything
read off a `static_argnums=(0)` `self` is baked into the trace at first call and can
never change again (DESIGN.md §7.4).

Phase D fixed this by moving the kernels to module-level functions that take every
tuning value as a traced argument, with only shape-determining values static
(`static_argnames`). The runtime-reconfiguration tests below are the regression guard
for that: they failed before Phase D and must keep passing after it.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from f1tenth_planning.control.config.controller_config import (  # noqa: E402
    LQRConfig,
    dynamic_mppi_config,
)
from f1tenth_planning.control.config.dynamics_config import f1tenth_params  # noqa: E402
from f1tenth_planning.control.dynamics_models.dynamic_model import (  # noqa: E402
    DynamicBicycleModel,
)
from f1tenth_planning.control.solvers.mppi_solver import (  # noqa: E402
    MPPISolver,
    _iteration_kernel,
)


# --------------------------------------------------------------------------
# plain dataclass configs
# --------------------------------------------------------------------------
def test_lqr_config_respects_caller_values():
    """__post_init__ must fill only UNSET fields, never overwrite what was passed."""
    custom_q = np.diag([1.0, 2.0, 3.0, 4.0])
    cfg = LQRConfig(Q=custom_q, dt=0.5, max_iterations=999, eps=1e-6)

    np.testing.assert_allclose(cfg.Q, custom_q)
    assert cfg.dt == 0.5
    assert cfg.max_iterations == 999
    assert cfg.eps == 1e-6


def test_lqr_config_defaults_still_applied():
    cfg = LQRConfig()
    assert cfg.dt == 0.01
    assert cfg.max_iterations == 50
    assert cfg.Q is not None and cfg.R is not None


def test_mpc_config_bounds_default_to_infinite():
    """Unset bounds become +/-inf so solvers can read them unconditionally."""
    cfg = dynamic_mppi_config()
    assert np.all(np.isneginf(cfg.x_min)) or cfg.x_min is not None
    assert cfg.u_min.shape == (cfg.nu,)
    assert cfg.Q.shape == (cfg.nx, cfg.nx)


# --------------------------------------------------------------------------
# JIT static-vs-traced semantics (the rule this design depends on)
# --------------------------------------------------------------------------
def test_static_argument_forces_a_retrace_per_value():
    """A value marked static is a compile-time constant: each distinct value is a
    separate compilation. Correct, but expensive -- so tuning values must not be
    static."""
    traces = {"n": 0}

    from functools import partial

    @partial(jax.jit, static_argnames=("temperature",))
    def kernel_static(R, *, temperature):
        traces["n"] += 1  # executes at TRACE time only
        return jnp.exp(R / temperature).sum()

    R = jnp.linspace(-1.0, 0.0, 32)
    for temp in (0.1, 0.5, 1.0, 2.0, 5.0):
        kernel_static(R, temperature=temp)

    assert traces["n"] == 5, "each distinct static value should compile separately"


def test_traced_argument_compiles_once_and_still_updates():
    """A traced value flows into the compiled kernel: one compile, values still
    take effect. This is what tuning parameters must be."""
    traces = {"n": 0}

    from functools import partial

    @partial(jax.jit, static_argnames=("n_samples",))
    def kernel_traced(R, temperature, *, n_samples):
        traces["n"] += 1
        return jnp.exp(R[:n_samples] / temperature).sum()

    R = jnp.linspace(-1.0, 0.0, 32)
    outs = [float(kernel_traced(R, t, n_samples=32)) for t in (0.1, 0.5, 1.0, 2.0, 5.0)]

    assert traces["n"] == 1, "a traced value must not trigger recompilation"
    assert len(set(outs)) == 5, "changing a traced value must change the output"


def test_structural_change_does_retrace():
    """Shape-determining values must be static, and changing one must retrace."""
    traces = {"n": 0}

    from functools import partial

    @partial(jax.jit, static_argnames=("n",))
    def kernel(key, *, n):
        traces["n"] += 1
        return jax.random.normal(key, (n, 2)).sum()

    key = jax.random.PRNGKey(0)
    kernel(key, n=8)
    kernel(key, n=8)
    assert traces["n"] == 1, "same structure should reuse the trace"
    kernel(key, n=16)
    assert traces["n"] == 2, "a structural change must trigger exactly one retrace"


def test_array_cannot_be_marked_static():
    """Safety net: array-valued tuning params can never be mis-marked static."""
    from functools import partial

    @partial(jax.jit, static_argnames=("u_min",))
    def kernel(x, *, u_min):
        return x + u_min

    with pytest.raises((ValueError, TypeError)):
        kernel(jnp.ones(2), u_min=jnp.array([-3.2, -9.51]))


# --------------------------------------------------------------------------
# the real solver -- these encode the Phase D bug
# --------------------------------------------------------------------------
@pytest.fixture
def mppi_solver():
    params = f1tenth_params()
    cfg = dynamic_mppi_config()
    cfg.N, cfg.n_samples, cfg.n_iterations = 8, 64, 1
    cfg.x_min = np.array([-np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED,
                          -np.inf, -np.inf, -np.inf])
    cfg.x_max = np.array([np.inf, np.inf, params.MAX_STEER, params.MAX_SPEED,
                          np.inf, np.inf, np.inf])
    cfg.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
    cfg.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])
    return MPPISolver(cfg, DynamicBicycleModel(params))


def _solve_once(solver):
    N = solver.config.N
    x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    ref = np.tile(x0.reshape(-1, 1), (1, N + 1))
    ref[3, :] = 3.0
    return solver.solve(x0, ref, vis=False)


def _reset_solver_state(solver):
    """Put the solver back to a pristine state.

    MPPI warm-starts from the previous solution (`control_params`) and advances its
    PRNG, so two consecutive solves differ for reasons that have nothing to do with
    config. Without this, a "did my config change take effect?" test passes for the
    wrong reason.
    """
    solver.control_params = solver._init_control()
    solver.rng = jax.random.PRNGKey(0)


def test_solver_solve_returns_state_then_control(mppi_solver):
    """Contract: solve() returns (x, u) -- the ABC docstring used to claim (u, x)."""
    xk, uk = _solve_once(mppi_solver)
    cfg = mppi_solver.config
    assert xk.shape == (cfg.nx, cfg.N + 1), f"state should be (nx, N+1), got {xk.shape}"
    assert uk.shape == (cfg.nu, cfg.N), f"control should be (nu, N), got {uk.shape}"


def test_prng_key_advances_between_solves(mppi_solver):
    """Exploration noise must be independent across steps, not reset to the seed."""
    _reset_solver_state(mppi_solver)
    _solve_once(mppi_solver)
    first = np.asarray(mppi_solver.rng).copy()
    _solve_once(mppi_solver)
    second = np.asarray(mppi_solver.rng).copy()
    assert not np.array_equal(first, second), "PRNG key was reset instead of advanced"


def test_tuning_change_takes_effect_at_runtime(mppi_solver):
    """Mutating a tuning value must change the solve result without a rebuild."""
    _reset_solver_state(mppi_solver)
    _, u_before = _solve_once(mppi_solver)

    mppi_solver.config.temperature = mppi_solver.config.temperature * 50.0
    _reset_solver_state(mppi_solver)  # identical noise + warm start; only temp differs
    _, u_after = _solve_once(mppi_solver)

    assert not np.allclose(np.asarray(u_before), np.asarray(u_after)), (
        "changing temperature had no effect -- it is baked into the JIT trace"
    )


def test_bounds_change_takes_effect_at_runtime(mppi_solver):
    """Tightening the control bounds must clamp the emitted control."""
    _reset_solver_state(mppi_solver)
    _solve_once(mppi_solver)

    mppi_solver.config.u_max = np.array([0.01, 0.01])
    mppi_solver.config.u_min = np.array([-0.01, -0.01])
    _reset_solver_state(mppi_solver)
    _, u_after = _solve_once(mppi_solver)

    assert np.max(np.abs(np.asarray(u_after))) <= 0.011, (
        "control exceeded the newly-set bounds -- they were baked into the trace"
    )


def test_tuning_change_does_not_recompile(mppi_solver):
    """Once Phase D lands this must hold: changing a tuning value reuses the trace.

    Recorded now as the companion to the xfail tests above -- today the kernel does
    not recompile either, but only because the value is ignored entirely.
    """
    _reset_solver_state(mppi_solver)
    _solve_once(mppi_solver)
    before = _iteration_kernel._cache_size()

    mppi_solver.config.temperature = mppi_solver.config.temperature * 2.0
    _reset_solver_state(mppi_solver)
    _solve_once(mppi_solver)
    after = _iteration_kernel._cache_size()

    assert after == before, "a tuning change must not trigger a recompilation"
