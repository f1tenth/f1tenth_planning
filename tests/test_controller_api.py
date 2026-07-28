"""The Controller / Planner interface contract (DESIGN.md §2-§4, §8)."""
import numpy as np
import pytest

from f1tenth_planning.control import (
    KinematicMPCPlanner,
    LQRController,
    NonlinearDynamicMPPIPlanner,
    PurePursuitPlanner,
    StanleyController,
)
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.planning.planner import Planner

OBS = {
    "pose_x": 0.0, "pose_y": 0.0, "delta": 0.0, "linear_vel_x": 3.0,
    "pose_theta": 0.0, "ang_vel_z": 0.0, "beta": 0.0,
}


def _classical(track):
    return [
        PurePursuitPlanner(track=track, params=f1tenth_params()),
        StanleyController(track=track),
        LQRController(track),
    ]


# ------------------------------------------------------------------- the contract
def test_every_controller_implements_compute_control(track):
    for c in _classical(track) + [KinematicMPCPlanner(track=track)]:
        assert hasattr(c, "compute_control")
        assert callable(c.compute_control)


def test_plan_is_gone(track):
    """Clean break: the entry point is compute_control(), with no plan() alias."""
    for c in _classical(track):
        assert not hasattr(c, "plan"), f"{type(c).__name__} still exposes plan()"


def test_lifecycle_hooks_exist_and_default_to_noops(track):
    """reset()/complete_iteration() must always be callable, so a runner can call
    them unconditionally regardless of which controller it was handed."""
    for c in _classical(track):
        assert c.reset() is None
        assert c.complete_iteration() is None


def test_controller_declares_its_control_mode(track):
    from f1tenth_gym.envs.action import LongitudinalActionEnum, SteerActionEnum

    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    assert pp.control_mode == (SteerActionEnum.Steering_Angle, LongitudinalActionEnum.Speed)

    mpc = KinematicMPCPlanner(track=track)
    assert mpc.control_mode == (SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl)


# ---------------------------------------------------------------- named reference
def test_reference_layouts_are_named_not_positional(track):
    """Each controller declares which reference fields it needs, so "column 3" is
    never ambiguous across controllers."""
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    lqr = LQRController(track)
    mpc = KinematicMPCPlanner(track=track)

    assert pp.reference.names == ("x", "y", "v", "yaw")
    assert lqr.reference.names == ("x", "y", "v", "yaw", "curvature")
    # the MPC family derives its layout from the model's state vector
    assert mpc.reference.names == mpc.model.state.names

    # the same *name* resolves to different columns per controller -- which is exactly
    # why generic code must ask by name
    assert pp.reference.idx.v == 2
    assert mpc.reference.idx.v == 3


def test_waypoints_match_the_declared_layout(track):
    lqr = LQRController(track)
    idx = lqr.reference.idx
    np.testing.assert_allclose(lqr.waypoints[:, idx.x], np.asarray(track.raceline.xs))
    np.testing.assert_allclose(lqr.waypoints[:, idx.v], np.asarray(track.raceline.vxs))
    np.testing.assert_allclose(lqr.waypoints[:, idx.curvature], np.asarray(track.raceline.ks))


def test_waypoints_keep_the_raceline_dtype(track):
    """Widening to float64 breaks the numba kernels' dtype dispatch."""
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    assert pp.waypoints.dtype == np.asarray(track.raceline.xs).dtype


# ------------------------------------------------------------- update() as the seam
def test_update_accepts_a_raceline_as_the_reference(track):
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    pp.waypoints = None
    pp.update(reference=track.raceline)
    assert pp.waypoints is not None
    assert pp.waypoints.shape[1] == len(pp.reference.names)


def test_update_accepts_a_track_as_the_reference(track):
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    pp.update(reference=track)
    np.testing.assert_allclose(
        pp.waypoints[:, pp.reference.idx.x], np.asarray(track.raceline.xs)
    )


def test_update_accepts_a_raw_waypoint_array(track):
    """A planner emitting its own array must be usable without conversion."""
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    custom = np.zeros((50, 4), dtype=np.float32)
    custom[:, 0] = np.linspace(0.0, 10.0, 50)
    pp.update(reference=custom)
    np.testing.assert_allclose(pp.waypoints, custom)


def test_update_rejects_a_nonsense_reference(track):
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    with pytest.raises(TypeError, match="Raceline"):
        pp.update(reference="not a reference")


def test_update_sets_vehicle_params(track):
    import copy

    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    new = copy.deepcopy(f1tenth_params())
    new.WHEELBASE = 0.5
    pp.update(params=new)
    assert pp.params.WHEELBASE == 0.5


def test_update_is_keyword_only(track):
    """Keyword-only keeps the signature from growing into positional soup."""
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    with pytest.raises(TypeError):
        pp.update(track.raceline)


def test_reference_update_changes_the_command(track):
    """The reference is a live input: swapping it must change what the controller does."""
    pp = PurePursuitPlanner(track=track, params=f1tenth_params())
    obs = dict(OBS, pose_x=float(track.raceline.xs[0]), pose_y=float(track.raceline.ys[0]),
               pose_theta=float(track.raceline.yaws[0]))
    before = np.asarray(pp.compute_control(obs), dtype=float)

    # shift the whole reference sideways; the steering command must react
    shifted = pp.waypoints.copy()
    shifted[:, pp.reference.idx.y] += 2.0
    pp.update(reference=shifted)
    after = np.asarray(pp.compute_control(obs), dtype=float)

    assert not np.allclose(before, after), "reference swap had no effect on the command"


# ------------------------------------------------------------------------ Planner
def test_planner_interface_is_minimal_and_separate():
    """Planner and Controller are separate entities -- no shared base class."""
    assert not issubclass(Planner, Controller)
    assert not issubclass(Controller, Planner)
    assert hasattr(Planner, "plan")


def test_a_planner_can_drive_a_controller(track):
    """The wiring the design mandates: planner -> update(reference) -> compute_control."""

    class ConstantPlanner(Planner):
        """Emits a fixed reference, ignoring the observation."""

        def __init__(self, raceline):
            self._raceline = raceline

        def plan(self, state, **context):
            return self._raceline

    planner = ConstantPlanner(track.raceline)
    controller = PurePursuitPlanner(track=track, params=f1tenth_params())

    reference = planner.plan(OBS)
    controller.update(reference=reference)
    action = controller.compute_control(
        dict(OBS, pose_x=float(track.raceline.xs[0]), pose_y=float(track.raceline.ys[0]))
    )
    assert np.all(np.isfinite(np.asarray(action, dtype=float)))


def test_mppi_controller_reset_clears_warm_start(track):
    """A controller holding state across steps should honour reset()."""
    c = NonlinearDynamicMPPIPlanner(track=track, params=f1tenth_params())
    c.compute_control(OBS)
    warm = np.asarray(c.solver.control_params[0]).copy()
    assert np.any(warm != 0.0), "expected a non-zero warm start after one solve"

    c.reset()
    assert np.all(np.asarray(c.solver.control_params[0]) == 0.0), (
        "reset() did not clear the solver's warm start"
    )
