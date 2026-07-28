"""Cross-backend agreement (DESIGN.md §5.5).

A model may implement its dynamics in numpy, jax and casadi. Those are hand-written
per backend on purpose (efficiency), so nothing structurally prevents them from
drifting apart. These tests are that guarantee: where a model provides more than one
backend, all of them must agree.

This is the check that caught the historical `tan(delta)**2`, dropped-slip-angle and
switch-threshold divergences.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from f1tenth_planning.control.config.dynamics_config import f1tenth_params  # noqa: E402
from f1tenth_planning.control.dynamics_models.dynamic_model import (  # noqa: E402
    DynamicBicycleModel,
)
from f1tenth_planning.control.dynamics_models.kinematic_model import (  # noqa: E402
    KinematicBicycleModel,
)

# Tolerance must be RELATIVE: yaw-rate derivatives reach ~1e2 rad/s^2, so a fixed
# atol is meaningless there. The jax/casadi backends deliberately guard their 1/v
# terms with epsilon=1e-4 (lax.select / if_else evaluate both branches, so the
# high-speed branch must stay finite at v=0); numpy uses a real `if` and needs no
# guard. That introduces a bounded relative difference of ~eps/v, i.e. <= 2e-4 at
# the v=0.5 switch. rtol below is comfortably above that, and still far below the
# ~1-4% errors produced by the historical `tan(delta)**2` bug this test guards.
RTOL = 2e-3
ATOL = 1e-8

# The low-speed/high-speed switch lives at v = 0.5; sample either side of it,
# plus reverse, plus the boundary itself.
SPEEDS = {
    "standstill": 0.05,
    "low_speed": 0.30,
    "just_below_switch": 0.49,
    "just_above_switch": 0.60,
    "mid_speed": 3.0,
    "high_speed": 8.0,
    "reverse": -3.0,
}


def _dynamic_state(v, rng):
    x, y, delta, yaw, yaw_rate, beta = rng.uniform(-0.3, 0.3, size=6)
    return np.array([x, y, delta, v, yaw, yaw_rate, beta], dtype=float)


@pytest.mark.parametrize("regime", list(SPEEDS))
def test_dynamic_model_backends_agree(regime, params, rng):
    """numpy / jax / casadi must produce the same state derivative."""
    model = DynamicBicycleModel(params)
    p_vec = model.parameters_vector_from_config(params)
    f_casadi = model.f_casadi()

    state = _dynamic_state(SPEEDS[regime], rng)
    control = np.array([0.2, 1.0])

    d_np = np.asarray(model.f(state, control, params), dtype=float)
    d_jax = np.asarray(
        model.f_jax(jnp.array(state), jnp.array(control), jnp.array(p_vec)), dtype=float
    )
    d_ca = np.asarray(f_casadi(state, control, p_vec)).flatten()

    np.testing.assert_allclose(
        d_np, d_jax, rtol=RTOL, atol=ATOL, err_msg=f"numpy vs jax ({regime})"
    )
    np.testing.assert_allclose(
        d_np, d_ca, rtol=RTOL, atol=ATOL, err_msg=f"numpy vs casadi ({regime})"
    )


def test_dynamic_model_jax_gradient_is_finite_at_zero_speed(params):
    """`lax.select` evaluates BOTH branches, so the high-speed 1/v terms must stay
    finite at v=0 or every gradient through f_jax is NaN."""
    model = DynamicBicycleModel(params)
    p_vec = jnp.array(model.parameters_vector_from_config(params))

    def total(v):
        state = jnp.array([1.0, 2.0, 0.1, v, 0.5, 0.3, 0.14])
        return jnp.sum(model.f_jax(state, jnp.array([0.2, 1.0]), p_vec))

    grad = float(jax.grad(total)(0.0))
    assert np.isfinite(grad), "jax.grad through f_jax is not finite at v=0"


def test_kinematic_model_backends_agree(params, rng):
    model = KinematicBicycleModel(params)
    p_vec = model.parameters_vector_from_config(params)
    f_casadi = model.f_casadi()

    state = np.array([1.0, 2.0, 0.1, 6.0, 0.4])
    control = np.array([0.2, 1.0])

    d_np = np.asarray(model.f(state, control, params), dtype=float)
    d_jax = np.asarray(
        model.f_jax(jnp.array(state), jnp.array(control), jnp.array(p_vec)), dtype=float
    )
    d_ca = np.asarray(f_casadi(state, control, p_vec)).flatten()

    np.testing.assert_allclose(d_np, d_jax, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(d_np, d_ca, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("model_cls", [DynamicBicycleModel, KinematicBicycleModel])
def test_parameter_vector_roundtrip(model_cls, params):
    """num_params must match the vector length, and config->vector->config must
    preserve the parameters the vector carries."""
    model = model_cls(params)
    p_vec = model.parameters_vector_from_config(params)

    assert p_vec.shape == (model.num_params, 1), "num_params disagrees with the vector"

    restored = model.config_from_parameters_vector(p_vec)
    assert restored is not model.params, "must return a copy, not mutate self.params"

    round_tripped = model.parameters_vector_from_config(restored)
    np.testing.assert_allclose(np.asarray(p_vec), np.asarray(round_tripped), atol=1e-12)


def test_config_from_parameters_vector_does_not_mutate_model(params):
    """A conversion helper must not silently overwrite the model's live config."""
    model = DynamicBicycleModel(params)
    before = model.params.MU

    p_vec = model.parameters_vector_from_config(params).copy()
    p_vec[0, 0] = before + 0.5
    model.config_from_parameters_vector(p_vec)

    assert model.params.MU == before, "config_from_parameters_vector mutated self.params"


def test_f_with_explicit_params_does_not_rebind_model_params(params):
    """Passing params for a single evaluation must not rebind the model's config."""
    import copy

    model = DynamicBicycleModel(params)
    other = copy.deepcopy(params)
    other.MU = params.MU + 0.4

    model.f(_dynamic_state(5.0, np.random.default_rng(1)), np.array([0.1, 0.5]), other)

    assert model.params.MU == params.MU, "f() permanently rebound self.params"
