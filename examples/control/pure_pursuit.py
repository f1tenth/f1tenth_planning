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
Pure Pursuit waypoint tracker example

Author: Hongrui Zheng
Last Modified: 5/4/22
"""

import numpy as np
import gymnasium as gym
import f110_gym
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner


def main():
    """
    Pure Pursuit example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make('f110_gym:f110-v0',
                   config={
                       "map": "Spielberg",
                       "num_agents": 1,
                       "control_input": "speed",
                       "observation_config": {"type": "kinematic_state"},
                   },
                   render_mode='human')

    # create planner
    raceline = env.unwrapped.track.raceline
    waypoints = np.stack([raceline.xs, raceline.ys, raceline.vxs], axis=1)
    planner = PurePursuitPlanner(waypoints=waypoints)

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

    # run simulation
    laptime = 0.0
    done = False
    while not done:
        steer, speed = planner.plan(obs["agent_0"]['pose_x'],
                                    obs["agent_0"]['pose_y'],
                                    obs["agent_0"]['pose_theta'],
                                    lookahead_distance=0.8)
        obs, timestep, terminated, truncated, infos = env.step(np.array([[steer, speed]]))
        laptime += timestep
        env.render()
    print('Sim elapsed time:', laptime)


if __name__ == '__main__':
    main()
