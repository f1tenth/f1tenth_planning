"""AP-MPPI: solver inheritance and the adaptive-penalty mechanism (DESIGN.md §6, §7).

The critical property here is that constraint violations must be *visible* in the
rollouts. If rollout states are clipped to the same bounds the constraints police,
no sample can ever violate anything, every penalty weight looks equally feasible,
and the adaptive penalty silently degenerates into plain MPPI.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from f1tenth_planning.control.config.controller_config import (  # noqa: E402
    dynamic_ap_mppi_config,
)
from f1tenth_planning.control.config.dynamics_config import f1tenth_params  # noqa: E402
from f1tenth_planning.control.dynamics_models.dynamic_model import (  # noqa: E402
    DynamicBicycleModel,
)
from f1tenth_planning.control.solvers import APMPPISolver, MPPISolver  # noqa: E402

V_LIMIT = 3.5


def _velocity_constraint(x, u):
    """Positive where velocity exceeds the limit (positive == violation)."""
    return jnp.maximum(0.0, x[:, 3] - V_LIMIT)


def _build(constraints, clip_velocity=None):
    params = f1tenth_params()
    n = max(len(constraints), 1)
    cfg = dynamic_ap_mppi_config(
        constraints=constraints,
        n_lambdas=16,
        lambdas_sample_range=np.array([[0.0, 1000.0]] * n),
    )
    cfg.N, cfg.n_samples, cfg.n_iterations = 15, 256, 1
    cfg.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
    cfg.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])
    if clip_velocity is not None:
        cfg.x_clip_max = np.array(
            [np.inf, np.inf, np.inf, clip_velocity, np.inf, np.inf, np.inf]
        )
    return APMPPISolver(cfg, DynamicBicycleModel(params))


def _solve(solver):
    x0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    ref = np.tile(x0.reshape(-1, 1), (1, solver.config.N + 1))
    ref[3, :] = 12.0                       # ask for far more speed than the limit
    return solver.solve(x0, ref, vis=False)


def _max_rollout_velocity(solver):
    _, states, _ = solver.samples          # (n_samples, N, nx)
    return float(jnp.max(states[:, :, 3]))


# ------------------------------------------------------------------ inheritance
def test_ap_mppi_inherits_from_mppi():
    """AP-MPPI is MPPI plus a penalty population -- not a parallel copy."""
    assert issubclass(APMPPISolver, MPPISolver)


def test_ap_mppi_reuses_the_mppi_machinery():
    """The shared pieces must come from the base class, not be duplicated."""
    for shared in ("solve", "update", "_init_control", "_rollout"):
        assert getattr(APMPPISolver, shared) is getattr(MPPISolver, shared), (
            f"{shared} is duplicated in APMPPISolver instead of inherited"
        )
    # and the parts that genuinely differ are overridden
    for specialised in ("iteration_step", "_make_step", "_static_kwargs"):
        assert getattr(APMPPISolver, specialised) is not getattr(
            MPPISolver, specialised, None
        )


# ------------------------------------------------------------- clip vs constrain
def test_clip_bounds_default_to_no_clipping():
    """Defaulting to +/-inf keeps violations visible, so penalties can act."""
    cfg = dynamic_ap_mppi_config()
    assert np.all(np.isneginf(cfg.x_clip_min))
    assert np.all(np.isposinf(cfg.x_clip_max))


def test_violations_are_visible_without_clipping():
    """With no stability clipping, rollouts may exceed the constrained bound --
    which is exactly what lets the adaptive penalty distinguish lambda values."""
    solver = _build([_velocity_constraint])
    _solve(solver)
    assert _max_rollout_velocity(solver) > V_LIMIT, (
        "no rollout exceeded the constrained limit, so the penalty has nothing to act on"
    )


def test_clipping_at_the_constrained_bound_hides_violations():
    """The failure mode this design guards against, pinned as a test.

    Clipping the same quantity a constraint polices makes every sample look feasible,
    which is why clip bounds are separate from constraint bounds.
    """
    solver = _build([_velocity_constraint], clip_velocity=V_LIMIT)
    _solve(solver)
    assert _max_rollout_velocity(solver) <= V_LIMIT + 1e-3
    _, states, _ = solver.samples
    violations = jnp.sum(jnp.maximum(0.0, _velocity_constraint(states[0], None)))
    assert float(violations) == pytest.approx(0.0, abs=1e-6)


def test_constraint_changes_the_emitted_command():
    """End-to-end: adding a constraint must change what the solver commands."""
    unconstrained = _build([])
    constrained = _build([_velocity_constraint])
    for s in (unconstrained, constrained):
        s.rng = jax.random.PRNGKey(0)      # identical noise; only the constraint differs

    _, u_free = _solve(unconstrained)
    _, u_constrained = _solve(constrained)

    assert not np.allclose(np.asarray(u_free), np.asarray(u_constrained)), (
        "the constraint had no effect on the command -- the penalty is a no-op"
    )


# ------------------------------------------------------------------- runtime cfg
def test_lambdas_are_traced_so_they_can_change_without_recompiling():
    """The penalty population is a runtime input, so it can be resampled/annealed."""
    from f1tenth_planning.control.solvers.ap_mppi_solver import _ap_iteration_kernel

    solver = _build([_velocity_constraint])
    _solve(solver)
    before = _ap_iteration_kernel._cache_size()

    solver.lambdas = solver.lambdas * 2.0
    _solve(solver)

    assert _ap_iteration_kernel._cache_size() == before, (
        "changing lambdas triggered a recompilation -- it is not traced"
    )


def test_ap_mppi_tuning_change_takes_effect():
    """Same runtime-reconfiguration guarantee as the base solver."""
    solver = _build([_velocity_constraint])
    solver.rng = jax.random.PRNGKey(0)
    solver.control_params = solver._init_control()
    _, u_before = _solve(solver)

    solver.config.temperature = solver.config.temperature * 50.0
    solver.rng = jax.random.PRNGKey(0)
    solver.control_params = solver._init_control()
    _, u_after = _solve(solver)

    assert not np.allclose(np.asarray(u_before), np.asarray(u_after)), (
        "temperature change had no effect -- it is baked into the trace"
    )
