import os

import gymnasium as gym
import f110_gym
import unittest

from f1tenth_planning.control.controller import load_params
from f1tenth_planning.control.stanley.stanley import StanleyPlanner


class TestLQR(unittest.TestCase):
    def test_load_params(self):
        env = gym.make(
            "f110_gym:f110-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "control_input": "speed",
                "observation_config": {"type": "kinematic_state"},
            },
            render_mode=None,
        )
        track = env.unwrapped.track

        # params as dict
        params = {"vgain": 0.123456789}
        planner = StanleyPlanner(track=track, params=params)
        vgain = planner.params["vgain"]
        self.assertTrue(
            abs(vgain - params["vgain"]) < 1e-6,
            f"vgain from dict params is not correct, got {vgain} != {params['vgain']}",
        )

        # params as path to yaml file
        cwd = os.path.dirname(os.path.realpath(__file__))
        filepath = f"{cwd}/../configs/control/pure_pursuit.yaml"
        planner = StanleyPlanner(track=track, params=filepath)
        expect_vgain = load_params(default_params={}, new_params=filepath)["vgain"]
        self.assertTrue(
            abs(planner.params["vgain"] - expect_vgain) < 1e-6,
            f"vgain from yaml params is not correct, got {planner.params['vgain']} != {expect_vgain}",
        )

        # none params
        expect_vgain = 1.0
        planner = StanleyPlanner(track=track, params=None)
        self.assertTrue(
            abs(planner.params["vgain"] - expect_vgain) < 1e-6,
            f"vgain from default params is not correct, got {planner.params['vgain']} != {expect_vgain}",
        )
