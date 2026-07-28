"""Self-describing layouts and backend capability queries (DESIGN.md §5.1-5.2)."""
import numpy as np
import pytest

from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.dynamics_models.kinematic_model import (
    KinematicBicycleModel,
)
from f1tenth_planning.control.spec import VariableSpec


# ------------------------------------------------------------------ VariableSpec
def test_spec_exposes_named_indices():
    spec = VariableSpec(("x", "y", "delta", "v", "yaw"))
    assert spec.size == 5
    assert spec.idx.v == 3          # the point: no hardcoded literal, no string key
    assert spec.idx.yaw == 4
    assert spec.index("delta") == 2
    assert spec.has("v") and not spec.has("beta")


def test_spec_rejects_bad_names():
    with pytest.raises(ValueError):
        VariableSpec(("x", "x"))            # duplicate
    with pytest.raises(ValueError):
        VariableSpec(("x", "yaw rate"))     # not an identifier


def test_spec_unknown_name_raises_keyerror():
    spec = VariableSpec(("x", "y"))
    with pytest.raises(KeyError):
        spec.index("v")


def test_spec_view_is_readable():
    spec = VariableSpec(("x", "y", "v"))
    view = spec.view([1.0, 2.0, 8.0])
    assert view.v == 8.0
    with pytest.raises(ValueError):
        spec.view([1.0, 2.0])


# ------------------------------------------------------------------ model layouts
@pytest.mark.parametrize(
    "model_cls,expected_nx",
    [(KinematicBicycleModel, 5), (DynamicBicycleModel, 7)],
)
def test_model_layout_matches_nx(model_cls, expected_nx):
    model = model_cls(f1tenth_params())
    assert model.state.size == expected_nx == model.nx
    assert model.control.size == model.nu == 2
    # velocity must be addressable by name -- generic code relies on it
    assert model.state.has("v")
    assert model.control.names == ("delta_v", "a")


def test_state_from_observation_matches_each_layout():
    obs = {
        "pose_x": 1.0, "pose_y": 2.0, "delta": 0.1, "linear_vel_x": 8.0,
        "pose_theta": 0.5, "ang_vel_z": 0.3, "beta": 0.14,
    }
    kin = KinematicBicycleModel(f1tenth_params())
    dyn = DynamicBicycleModel(f1tenth_params())

    x_kin = kin.state_from_observation(obs)
    x_dyn = dyn.state_from_observation(obs)

    assert x_kin.shape == (5,)
    assert x_dyn.shape == (7,)
    # each model reads the fields ITS layout names -- no shared-prefix slicing
    np.testing.assert_allclose(x_kin, [1.0, 2.0, 0.1, 8.0, 0.5])
    np.testing.assert_allclose(x_dyn, [1.0, 2.0, 0.1, 8.0, 0.5, 0.3, 0.14])
    assert x_dyn[dyn.state.idx.v] == 8.0


def test_state_from_observation_reports_missing_key():
    dyn = DynamicBicycleModel(f1tenth_params())
    with pytest.raises(KeyError, match="beta"):
        dyn.state_from_observation({
            "pose_x": 0.0, "pose_y": 0.0, "delta": 0.0,
            "linear_vel_x": 0.0, "pose_theta": 0.0, "ang_vel_z": 0.0,
        })


# ------------------------------------------------------------------ capabilities
def test_backend_capability_reporting():
    dyn = DynamicBicycleModel.backends()
    kin = KinematicBicycleModel.backends()

    for backend in ("numpy", "jax", "casadi", "casadi_opti"):
        assert backend in dyn and backend in kin

    # only the kinematic model has a hand-written Jacobian; the dynamic one does not
    assert "jacobian" in kin
    assert "jacobian" not in dyn

    assert KinematicBicycleModel.supports("numpy", "jacobian")
    assert not DynamicBicycleModel.supports("jacobian")


def test_solver_rejects_model_missing_a_required_backend():
    """A solver+model mismatch must fail at construction, not mid-solve."""
    from f1tenth_planning.control.config.controller_config import kinematic_mpc_config
    from f1tenth_planning.control.solvers import LTVMPCSolver

    params = f1tenth_params()
    cfg = kinematic_mpc_config()

    # LTV linearises, so it needs a Jacobian the dynamic model does not provide
    with pytest.raises(TypeError, match="jacobian"):
        LTVMPCSolver(cfg, DynamicBicycleModel(params))


def test_solver_accepts_a_capable_model():
    from f1tenth_planning.control.config.controller_config import kinematic_mpc_config
    from f1tenth_planning.control.solvers import LTVMPCSolver

    solver = LTVMPCSolver(kinematic_mpc_config(), KinematicBicycleModel(f1tenth_params()))
    assert solver.model.supports(*LTVMPCSolver.REQUIRED_BACKENDS)


def test_model_without_declared_names_raises():
    class Bare(DynamicsModel):
        def __init__(self, params):
            super().__init__(params)

    model = Bare(f1tenth_params())
    with pytest.raises(NotImplementedError, match="STATE_NAMES"):
        _ = model.state


# ------------------------------------------------------- reference construction
def test_reference_waypoints_built_in_model_layout(track):
    """The waypoint matrix must match the model's own state width and ordering."""
    from f1tenth_planning.control.spec import waypoints_from_raceline

    params = f1tenth_params()
    for model_cls in (KinematicBicycleModel, DynamicBicycleModel):
        model = model_cls(params)
        wp = waypoints_from_raceline(track.raceline, model.state.names)
        idx = model.state.idx

        assert wp.shape[1] == model.nx
        np.testing.assert_allclose(wp[:, idx.x], np.asarray(track.raceline.xs))
        np.testing.assert_allclose(wp[:, idx.y], np.asarray(track.raceline.ys))
        np.testing.assert_allclose(wp[:, idx.v], np.asarray(track.raceline.vxs))
        np.testing.assert_allclose(wp[:, idx.yaw], np.asarray(track.raceline.yaws))
        # names the raceline does not carry are zero-filled
        assert np.all(wp[:, idx.delta] == 0.0)
