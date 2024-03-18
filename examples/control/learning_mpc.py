# MIT License

# Copyright (c) Ahmad Amine

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
LMPC example

Author: Ahmad Amine
Last Modified: 8/1/22
"""

import numpy as np
import gymnasium as gym
from f110_gym.envs import F110Env
import time
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from f1tenth_planning.control.kinematic_mpc.kinematic_mpc import KMPCPlanner
from f1tenth_planning.control.learning_mpc.learning_mpc import LMPCPlanner

def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    features = [
        "pose_x",
        "pose_y",
        "delta",
        "linear_vel_x",
        "pose_theta",
        "ang_vel_z",
        "beta",
        "lap_time",
        "lap_count",
    ]
    # create environment
    env : F110Env = gym.make('f110_gym:f110-v0',
                            config={
                                "map": "IMS",
                                "num_agents": 1,
                                "control_input": "accl",
                                "observation_config": {"type": "features", 
                                                       "features": features},
                            },
                            render_mode='human')

    # create planner
    planner = KMPCPlanner(track=env.track, debug=False)
    planner.config.dlk = env.track.raceline.ss[1] - env.track.raceline.ss[0] # waypoint spacing
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_mpc_sol)

    # Initialize LMPC
    lmpc = LMPCPlanner(track=env.track)

    # reset environment
    poses = np.array(
        [
            [
                env.unwrapped.track.raceline.xs[0],
                env.unwrapped.track.raceline.ys[0],
                env.unwrapped.track.raceline.yaws[0],
                0.0
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    total_time = 0.0
    desired_laps = 1

    curr_trajectory = np.zeros((0, 7))
    curr_controls = np.zeros((0, planner.config.NU))

    lap_count = 0
    start = time.time()
    first_done = False
    while not done:
        curr_state = np.array([
                                [obs["agent_0"]["pose_x"],
                                obs["agent_0"]["pose_y"],
                                obs["agent_0"]["delta"],
                                obs["agent_0"]["linear_vel_x"],
                                obs["agent_0"]["pose_theta"],
                                obs["agent_0"]["ang_vel_z"],
                                obs["agent_0"]["beta"]]
                              ])        
        frenet_kinematic_pose = env.track.cartesian_to_frenet(curr_state[0, 0], curr_state[0, 1], curr_state[0, 4])
        curr_state_frenet = np.array([
                                [frenet_kinematic_pose[0],
                                frenet_kinematic_pose[1],
                                obs["agent_0"]["delta"],
                                obs["agent_0"]["linear_vel_x"],
                                frenet_kinematic_pose[2],
                                obs["agent_0"]["ang_vel_z"],
                                obs["agent_0"]["beta"]]
                              ])
        
        steerv, accl = planner.plan(obs["agent_0"])
        if planner.xk.value is not None:
            lmpc.update_zt(planner.xk.value)

        obs, timestep, terminated, truncated, infos = env.step(np.array([[steerv, accl]]))
        done = terminated or truncated
        env.render()

        curr_controls = np.vstack((curr_controls, np.array([[steerv, accl]])))
        curr_trajectory = np.vstack((curr_trajectory, curr_state_frenet))

        print("speed: {}, steer vel: {}, accl: {}".format(obs["agent_0"]['linear_vel_x'], steerv, accl))

        if lap_count != obs["agent_0"]["lap_count"]:
            lap_count = obs["agent_0"]["lap_count"]
            lap_time = obs["agent_0"]["lap_time"] - total_time
            print("laps_completed: {} | lap_time: {}".format(obs["agent_0"]["lap_count"], lap_time))
            total_time = obs["agent_0"]["lap_time"]
            print("Done: {}, Terminated: {}, Truncated: {}".format(done, terminated, truncated))

            # Increasing from 0 to length of trajectory 
            curr_values = np.array([np.arange(0, curr_trajectory.shape[0], 1)])
            lmpc.update_safe_set(curr_trajectory, curr_controls, curr_values)

            curr_trajectory = np.zeros((0, 7))
            curr_controls = np.zeros((0, planner.config.NU))

        if done and (obs["agent_0"]["lap_count"] != desired_laps):
            done = False
            terminated = False
            
    # Scatter xSS x,y colored with vSS
    for i in range(len(lmpc.SS_trajectories)):
        # CONVERT FRENET BACK TO CARTESIAN
        traj_x = []
        traj_y = []
        for state in lmpc.SS_trajectories[i]:
            cartesian_pose = env.track.frenet_to_cartesian(state[0], state[1], state[4])
            traj_x.append(cartesian_pose[0])
            traj_y.append(cartesian_pose[1])
        traj_x = np.array(traj_x)
        traj_y = np.array(traj_y)
        plt.scatter(traj_x, traj_y, c=lmpc.vSS_trajectories[i])
    # Plot zt for reference
    plt.scatter(lmpc.zt[0], lmpc.zt[1], c='r', s=50, marker='x')
    # Plot current position
    plt.scatter(curr_state[0,0], curr_state[0,1], c='g', s=50, marker='s')
    plt.show()

    print("Sim elapsed time:", total_time, "Real elapsed time:", time.time() - start)


if __name__ == '__main__':
    main()
