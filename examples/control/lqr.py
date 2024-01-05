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
LQR waypoint tracker example

Author: Hongrui Zheng
Last Modified: 5/5/22
"""

import numpy as np
import gymnasium as gym
import f110_gym
from f1tenth_planning.control.lqr.lqr import LQRPlanner


def main():
    """
    LQR example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "speed",
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="human",
    )

    # create planner
    track = env.unwrapped.track
    planner = LQRPlanner(track=track)

    env.add_render_callback(planner.render_waypoints)
    env.add_render_callback(planner.render_local_plan)
    env.add_render_callback(planner.render_closest_point)

    # reset environment
    poses = np.array(
        [
            [
                env.track.raceline.xs[0],
                env.track.raceline.ys[0],
                env.track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    while not done:
        action = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(np.array([action]))
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
