"""Shared fixtures. Everything here is headless — no display required."""
import os

import numpy as np
import pytest

# never open a render window during tests
os.environ.pop("DISPLAY", None)


@pytest.fixture(scope="session")
def params():
    from f1tenth_planning.control.config.dynamics_config import f1tenth_params

    return f1tenth_params()


@pytest.fixture(scope="session")
def track():
    """The Spielberg track from a headless env (session-scoped: construction is slow)."""
    import gymnasium as gym

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode=None,
    )
    yield env.unwrapped.track
    env.close()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
