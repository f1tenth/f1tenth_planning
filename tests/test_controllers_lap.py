"""End-to-end regression: every reference controller drives the Spielberg track.

This is the safety net for the API redesign. It is deliberately behavioural rather
than numerical -- it asserts that each controller *runs* and *goes somewhere*, so it
keeps working across refactors while still catching the failures that matter
(exceptions, NaN commands, a controller that stops making progress).

Two tiers of expectation, and the difference is by design, not a bug:

* the classical trackers (Pure Pursuit, Stanley, LQR) complete a full lap;
* the MPC/MPPI family are pure reference trackers with **no track-boundary or
  obstacle constraints**, so they leave the track at the first friction-limited
  corner. They are asserted to run cleanly and make forward progress, not to finish.
"""
import os

import numpy as np
import pytest

os.environ.pop("DISPLAY", None)

import gymnasium as gym  # noqa: E402

from f1tenth_planning.control.config.dynamics_config import f1tenth_params  # noqa: E402

KINEMATIC_ENV = {
    "map": "Spielberg",
    "num_agents": 1,
    "control_input": ["speed", "steering_angle"],
    "observation_config": {"type": "kinematic_state"},
}
DYNAMIC_ENV = {
    "map": "Spielberg",
    "num_agents": 1,
    "control_input": "accl",
    "observation_config": {"type": "dynamic_state"},
}

# progress (as a fraction of lap length) every controller must reach before it may
# stop making progress; the first sharp corner sits at ~11 %
MIN_PROGRESS_FRACTION = 0.08


def _default_plan(controller, obs):
    return controller.plan(obs)


def _stanley_plan(controller, obs):
    """Stanley is tuned at the call site in examples/control/stanley.py."""
    steer, speed = controller.plan(obs, k_path=7.0)
    return steer, 0.7 * speed


def _build(name):
    """(env_config, factory, plan_fn) per controller, mirroring examples/control/.

    The plan_fn matters: the shipped examples are not uniform -- Stanley passes a
    gain and scales its speed -- and a test that ignores that is not testing the
    reference configuration.
    """
    from f1tenth_planning.control import (
        KinematicMPCPlanner,
        LQRController,
        NonlinearDynamicMPCPlanner,
        NonlinearDynamicMPPIPlanner,
        NonlinearKinematicMPCPlanner,
        PurePursuitPlanner,
        StanleyController,
    )

    if name == "pure_pursuit":
        return (KINEMATIC_ENV,
                lambda t: PurePursuitPlanner(track=t, params=f1tenth_params()),
                _default_plan)
    if name == "stanley":
        return KINEMATIC_ENV, lambda t: StanleyController(track=t), _stanley_plan
    if name == "lqr":
        return KINEMATIC_ENV, lambda t: LQRController(t), _default_plan
    if name == "kinematic_mpc":
        return DYNAMIC_ENV, lambda t: KinematicMPCPlanner(track=t), _default_plan
    if name == "nonlinear_kmpc":
        return DYNAMIC_ENV, lambda t: NonlinearKinematicMPCPlanner(track=t), _default_plan
    if name == "nonlinear_dmpc":
        return (DYNAMIC_ENV,
                lambda t: NonlinearDynamicMPCPlanner(track=t, params=f1tenth_params()),
                _default_plan)
    if name == "dynamic_mppi":
        return (DYNAMIC_ENV,
                lambda t: NonlinearDynamicMPPIPlanner(track=t, params=f1tenth_params()),
                _default_plan)
    raise ValueError(name)


def _drive(name, max_sim_seconds):
    """Run one controller until it laps, crashes, stalls or runs out of sim time."""
    env_config, factory, plan_fn = _build(name)
    env = gym.make("f1tenth_gym:f1tenth-v0", config=env_config, render_mode=None)
    try:
        u = env.unwrapped
        rl = u.track.raceline
        xy = np.column_stack([np.asarray(rl.xs), np.asarray(rl.ys)])
        s_of_idx = np.asarray(rl.ss, dtype=float)
        length = float(rl.length)

        controller = factory(u.track)
        poses = np.array([[rl.xs[0], rl.ys[0], rl.yaws[0]]])
        obs, _ = env.reset(options={"poses": poses})

        def arclength(px, py):
            return s_of_idx[int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2))]

        prev_s = arclength(obs["agent_0"]["pose_x"], obs["agent_0"]["pose_y"])
        travelled = max_travelled = 0.0
        t0 = u.current_time
        steps = 0

        while u.current_time - t0 < max_sim_seconds:
            action = plan_fn(controller, obs["agent_0"])
            action = np.asarray(action, dtype=float).flatten()
            assert action.shape == (2,), f"{name}: expected a 2-vector, got {action.shape}"
            assert np.all(np.isfinite(action)), f"{name}: non-finite command {action}"

            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            steps += 1

            s = arclength(obs["agent_0"]["pose_x"], obs["agent_0"]["pose_y"])
            ds = s - prev_s
            if ds < -length / 2:      # wrapped past the start line
                ds += length
            elif ds > length / 2:     # wrapped backwards
                ds -= length
            travelled += ds
            prev_s = s
            max_travelled = max(max_travelled, travelled)

            if float(u.collisions[0]) > 0.0:
                return dict(outcome="crash", progress=max_travelled / length, steps=steps)
            if int(u.lap_counts[0]) >= 1 or max_travelled >= length:
                return dict(outcome="lap", progress=max_travelled / length, steps=steps)
            if terminated or truncated:
                return dict(outcome="done", progress=max_travelled / length, steps=steps)

        return dict(outcome="timeout", progress=max_travelled / length, steps=steps)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["pure_pursuit", "stanley", "lqr"])
def test_classical_controller_completes_a_lap(name):
    result = _drive(name, max_sim_seconds=120.0)
    assert result["outcome"] == "lap", (
        f"{name} did not finish a lap: {result['outcome']} at "
        f"{result['progress'] * 100:.1f}% of the track"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["kinematic_mpc", "nonlinear_kmpc", "nonlinear_dmpc", "dynamic_mppi"]
)
def test_mpc_controller_runs_and_makes_progress(name):
    """MPC/MPPI have no boundary constraints, so leaving the track is expected.

    What must hold is that they run without raising, emit finite commands, and get
    meaningfully down the track before doing so.
    """
    result = _drive(name, max_sim_seconds=30.0)
    assert result["steps"] > 0
    assert result["progress"] >= MIN_PROGRESS_FRACTION, (
        f"{name} only reached {result['progress'] * 100:.1f}% of the track "
        f"(outcome={result['outcome']}) -- expected at least "
        f"{MIN_PROGRESS_FRACTION * 100:.0f}%"
    )
