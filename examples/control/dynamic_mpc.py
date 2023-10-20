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
import f110_gym

from f1tenth_planning.control.dynamic_mpc.dynamic_mpc import STMPCPlanner

def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make('f110_gym:f110-v0',
                   config={
                       "map": "Spielberg",
                       "num_agents": 1,
                       "control_input": "accl",
                       "observation_config": {"type": "dynamic_state"},
                   },
                   render_mode='human')

    # create planner
    raceline = env.unwrapped.track.centerline
    waypoints = [raceline.xs, raceline.ys, raceline.yaws, 2.0 * raceline.vxs]
    planner = STMPCPlanner(waypoints=waypoints, debug=True)

    first_pose = np.array([raceline.xs[0], raceline.ys[0], raceline.yaws[0]])
    obs, infos = env.reset(options={"poses": first_pose[None]})
    done = False

    laptime = 0.0
    while not done:
        steerv, accl = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, info = env.step(np.array([[steerv, accl]]))
        done = terminated or truncated
        laptime += timestep
        env.render()

    print('Sim elapsed time:', laptime)

if __name__ == '__main__':
    main()