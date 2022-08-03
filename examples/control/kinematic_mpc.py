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
import gym

from f1tenth_planning.control.kinematic_mpc.kinematic_mpc import KMPCPlanner

def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # loading waypoints
    waypoints = np.loadtxt('./levine_centerline.csv', delimiter=';', skiprows=3)
    # [x, y, yaw, v]
    mpc_line = [waypoints[:, 1], waypoints[:, 2], waypoints[:, 3], waypoints[:, 5]]
    planner = KMPCPlanner(waypoints=mpc_line)

    # create environment
    env = gym.make('f110_gym:f110-v0', map='./levine_slam', map_ext='.pgm', num_agents=1)
    obs, _, done, _ = env.reset(np.array([[2.51, 3.29, 1.58]]))

    laptime = 0.0
    up_to_speed = False
    while not done:
        if up_to_speed:
            steer, speed = planner.plan(env.sim.agents[0].state)
            obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
            laptime += timestep
            env.render(mode='human')
        else:
            steer = 0.0
            speed = 10.0
            obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
            laptime += timestep
            env.render(mode='human')
            if obs['linear_vels_x'][0] > 0.1:
                up_to_speed = True
    print('Sim elapsed time:', laptime)

if __name__ == '__main__':
    main()