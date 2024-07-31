# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
UKF Estimator example, uses the UKF to estimate the state of the vehicle and uses the PurePursuit controller to track the waypoints.

Author: Ahmad Amine
"""

import numpy as np
import gymnasium as gym

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from f110_gym.envs.f110_env import F110Env
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner
from f1tenth_planning.estimation.ukf_estimator import ST_UKF


def main():
    """
    Pure Pursuit example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    num_agents = 1
    env : F110Env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": num_agents,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
            "reset_config": {"type": "random_static"},
        },
        render_mode="human",
    )
    track = env.unwrapped.track


    # create planner
    raceline = env.unwrapped.track.raceline
    waypoints = np.stack([raceline.xs, raceline.ys, raceline.vxs], axis=1)
    planner = PurePursuitPlanner(waypoints=waypoints)

    # create estimator
    params = env.default_config()["params"]
    params['g'] = 9.81 # Add gravity to the parameters
    params['a_min'] = -5.0 # Minimum acceleration
    estimator = ST_UKF(params, dt=0.01)
    
    env.add_render_callback(planner.render_waypoints)
    env.add_render_callback(planner.render_local_plan)
    env.add_render_callback(planner.render_lookahead_point)

    # reset environment
    track = env.unwrapped.track
    poses = np.array(
        [
            [
                track.raceline.xs[0],
                track.raceline.ys[0],
                track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    while not done:
        steer, speed = planner.plan(
            obs["agent_0"]["pose_x"],
            obs["agent_0"]["pose_y"],
            obs["agent_0"]["pose_theta"],
            lookahead_distance=0.8,
        )
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
