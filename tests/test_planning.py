"""The planning subsystem's interface conformance (DESIGN.md §2, §4).

The algorithms themselves are still work in progress -- these tests pin the
*structure*: that planners are Planners, that reference-free reactive controllers are
Controllers, and that the modules import at all (they previously did not, because of
stale import paths left behind when Pure Pursuit moved).
"""
import numpy as np
import pytest

from f1tenth_planning.control.controller import Controller
from f1tenth_planning.planning.planner import Planner


def test_planning_modules_import():
    """These modules were unimportable: they referenced a pure_pursuit path that had
    moved, so the failure was at module scope."""
    from f1tenth_planning.planning.lane_switcher.lane_switcher import LaneSwitcher
    from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner

    assert LaneSwitcher is not None and LatticePlanner is not None


def test_reference_producers_are_planners():
    from f1tenth_planning.planning.lane_switcher.lane_switcher import LaneSwitcher
    from f1tenth_planning.planning.lattice_planner.lattice_planner import LatticePlanner

    assert issubclass(LatticePlanner, Planner)
    assert issubclass(LaneSwitcher, Planner)


def test_reactive_algorithms_are_controllers():
    """FGM and wall following read a scan and emit a command, so they are
    Controllers -- not Planners, and not a separate 'reactive controller' type."""
    from f1tenth_planning.planning.fgm.fgm import FollowTheGapController
    from f1tenth_planning.planning.wall_follow.wall_follow import WallFollowController

    assert issubclass(FollowTheGapController, Controller)
    assert issubclass(WallFollowController, Controller)
    assert not issubclass(FollowTheGapController, Planner)


def test_unimplemented_algorithms_say_so_clearly():
    """A placeholder must fail loudly, not pretend to work."""
    from f1tenth_planning.planning.fgm.fgm import FollowTheGapController
    from f1tenth_planning.planning.wall_follow.wall_follow import WallFollowController

    for cls in (FollowTheGapController, WallFollowController):
        with pytest.raises(NotImplementedError):
            cls()


def test_lattice_planner_selects_a_trajectory():
    """End-to-end through the Planner contract: sample -> fit -> score -> select.

    The planner returns a reference for a controller to track; it does not emit a
    command (it used to construct its own Pure Pursuit and return steer/speed).
    """
    pytest.importorskip("pyclothoids")
    from f1tenth_planning.planning.lattice_planner.lattice_planner import (
        LatticePlanner,
        sample_lookahead_square,
    )

    # a straight reference line: [x, y, v, yaw, curvature]
    n = 200
    waypoints = np.zeros((n, 5))
    waypoints[:, 0] = np.linspace(0.0, 40.0, n)
    waypoints[:, 2] = 3.0

    planner = LatticePlanner(waypoints=waypoints)
    planner.add_sample_function(sample_lookahead_square)
    planner.add_cost_function(lambda traj: float(np.sum(np.abs(traj[:, 3]))))

    state = {"pose_x": 0.0, "pose_y": 0.0, "pose_theta": 0.0, "linear_vel_x": 3.0}
    selected = planner.plan(state)

    assert selected.ndim == 2 and selected.shape[1] == 4, (
        f"expected an (M, 4) [x, y, theta, curvature] reference, got {selected.shape}"
    )
    assert np.all(np.isfinite(selected))


def test_lattice_sampler_pairs_every_centre_with_every_width():
    """The goal grid must be the full cross product of lookaheads x lateral offsets."""
    pytest.importorskip("pyclothoids")
    from f1tenth_planning.planning.lattice_planner.lattice_planner import (
        sample_lookahead_square,
    )

    n = 200
    waypoints = np.zeros((n, 5))
    waypoints[:, 0] = np.linspace(0.0, 40.0, n)

    lookaheads = [1.0, 2.0, 3.0]
    widths = np.array([-0.5, 0.0, 0.5])
    grid = sample_lookahead_square(
        0.0, 0.0, 0.0, 3.0, waypoints, lookahead_distances=lookaheads, widths=widths
    )

    assert grid.shape == (len(lookaheads) * len(widths), 3)
    assert np.all(np.isfinite(grid))
