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
Last Modified: 7/26/22
"""

import numpy as np
import gym

from f1tenth_planning.control.dynamic_mpc.dynamic_mpc import STMPCPlanner

def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # loading waypoints
    waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=0)
    planner = STMPCPlanner(waypoints=waypoints)

    # create environment
    env = gym.make('f110_gym:f110-v0', map='./Spielberg_map', map_ext='.png', num_agents=1)
    obs, _, done, _ = env.reset(np.array([[0.0, -0.84, 3.40]]))

    laptime = 0.0
    while not done:
        steer, speed = planner.plan(env.sim.agents[0].state)
        obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
        laptime += timestep
        env.render(mode='human')
    print('Sim elapsed time:', laptime)

if __name__ == '__main__':
    main()