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
STMPC waypoint tracker example

Author: Hongrui Zheng
Last Modified: 8/1/22
"""

import numpy as np
import gymnasium as gym
from f110_gym.envs import F110Env
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from f1tenth_planning.control.kinematic_mpc.kinematic_mpc import KMPCPlanner


def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env : F110Env = gym.make('f110_gym:f110-v0',
                            config={
                                "map": "Spielberg",
                                "num_agents": 1,
                                "control_input": "accl",
                                "observation_config": {"type": "dynamic_state"},
                            },
                            render_mode='human')

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env.add_render_callback(render_callback)

    # create planner
    planner = KMPCPlanner(track=env.track, debug=False)

    # create environment
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

    laptime = 0.0
    start = time.time()
    while not done:
        steerv, accl = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(np.array([[steerv, accl]]))
        done = terminated or truncated
        laptime += timestep
        env.render()

        print("speed: {}, steer vel: {}, accl: {}".format(obs["agent_0"]['linear_vel_x'], steerv, accl))

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == '__main__':
    main()
