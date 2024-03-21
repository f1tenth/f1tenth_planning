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
                                "map": "Rounded_Rectangle",
                                "num_agents": 1,
                                "control_input": "accl",
                                "observation_config": {"type": "features", 
                                                       "features": features},
                            },
                            render_mode='human')

    # create planner
    planner = KMPCPlanner(track=env.unwrapped.track, debug=False)
    planner.config.dlk = env.unwrapped.track.raceline.ss[1] - env.unwrapped.track.raceline.ss[0] # waypoint spacing
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_mpc_sol)

    # Initialize LMPC
    lmpc = LMPCPlanner(track=env.unwrapped.track)

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
    desired_laps = 4

    curr_trajectory = np.zeros((0, lmpc.config.NX))
    curr_controls = np.zeros((0, lmpc.config.NU))

    lap_count = 0
    start = time.time()
    first_done = False
    completed_laps = 0
    curr_state = np.array([
                            [obs["agent_0"]["pose_x"],
                            obs["agent_0"]["pose_y"],
                            obs["agent_0"]["delta"],
                            obs["agent_0"]["linear_vel_x"],
                            obs["agent_0"]["pose_theta"],
                            obs["agent_0"]["ang_vel_z"],
                            obs["agent_0"]["beta"]]
                            ])        
    frenet_kinematic_pose = env.unwrapped.track.cartesian_to_frenet(curr_state[0, 0], curr_state[0, 1], curr_state[0, 4], s_guess=0.0)
    curr_state_frenet = np.array([
                            [frenet_kinematic_pose[0],
                            frenet_kinematic_pose[1],
                            frenet_kinematic_pose[2],
                            obs["agent_0"]["linear_vel_x"]]
                            ])
    while not done:
        old_s = curr_state_frenet[0, 0]
        curr_state = np.array([
                                [obs["agent_0"]["pose_x"],
                                obs["agent_0"]["pose_y"],
                                obs["agent_0"]["delta"],
                                obs["agent_0"]["linear_vel_x"],
                                obs["agent_0"]["pose_theta"],
                                obs["agent_0"]["ang_vel_z"],
                                obs["agent_0"]["beta"]]
                              ])        
        frenet_kinematic_pose = env.unwrapped.track.cartesian_to_frenet(curr_state[0, 0], curr_state[0, 1], curr_state[0, 4], s_guess=old_s)
        curr_state_frenet = np.array([
                                [frenet_kinematic_pose[0],
                                frenet_kinematic_pose[1],
                                frenet_kinematic_pose[2],
                                obs["agent_0"]["linear_vel_x"]]
                              ])
        
        if abs(old_s - curr_state_frenet[0, 0]) > 5:
            print("S is not continuous, old_s: {}, new_s: {}".format(old_s, curr_state_frenet[0, 0]))
            completed_laps += 1

        steerv, accl = planner.plan(obs["agent_0"])
        if planner.xk.value is not None:
            lmpc.update_zt(planner.xk.value)

        obs, timestep, terminated, truncated, infos = env.step(np.array([[steerv, accl]]))
        done = terminated or truncated
        env.render()

        curr_controls = np.vstack((curr_controls, np.array([[steerv, accl]])))
        curr_trajectory = np.vstack((curr_trajectory, curr_state_frenet))

        print("speed: {}, steer vel: {}, accl: {}".format(obs["agent_0"]['linear_vel_x'], steerv, accl))

        if lap_count != completed_laps:
            lap_count = completed_laps
            lap_time = obs["agent_0"]["lap_time"] - total_time
            print("laps_completed: {} | lap_time: {}".format(completed_laps, lap_time))
            total_time = obs["agent_0"]["lap_time"]
            print("Done: {}, Terminated: {}, Truncated: {}".format(done, terminated, truncated))

            # decreasing from length of trajectory to 0
            curr_values = np.arange(curr_trajectory.shape[0]-1, -1, -1).reshape(1,-1)
            lmpc.update_safe_set(curr_trajectory, curr_controls, curr_values)

            curr_trajectory = np.zeros((0, lmpc.config.NX))
            curr_controls = np.zeros((0, lmpc.config.NU))

        if done and (completed_laps != desired_laps):
            done = False
            terminated = False
        
        if completed_laps == desired_laps:
            done = True

            
    # Scatter xSS x,y colored with vSS
    for i in range(len(lmpc.SS_trajectories)):
        # CONVERT FRENET BACK TO CARTESIAN
        traj_x = []
        traj_y = []
        traj_s = []
        for state in lmpc.SS_trajectories[i]:
            cartesian_pose = env.unwrapped.track.frenet_to_cartesian(state[0], state[1], state[-1])
            traj_x.append(cartesian_pose[0])
            traj_y.append(cartesian_pose[1])
            traj_s.append(state[0])
        traj_x = np.array(traj_x)
        traj_y = np.array(traj_y)
        traj_s = np.array(traj_s)
        plt.scatter(traj_x, traj_y, c=traj_s)
                    #lmpc.vSS_trajectories[i])
    plt.colorbar() # Colorbar before other scatter to ensure correct range
    # Plot zt for reference
    plt.scatter(lmpc.zt[0], lmpc.zt[1], c='r', s=50, marker='x')
    # Plot current position
    plt.scatter(curr_state[0,0], curr_state[0,1], c='g', s=50, marker='s')
    # Plot raceline for reference
    plt.plot(env.unwrapped.track.raceline.xs, env.unwrapped.track.raceline.ys, c='k')
    plt.show()

    print("Sim elapsed time:", total_time, "Real elapsed time:", time.time() - start)

    # Now switch to LMPC planning
    done = False
    while not done:
        old_s = curr_state_frenet[0, 0]
        curr_state = np.array([
                                [obs["agent_0"]["pose_x"],
                                obs["agent_0"]["pose_y"],
                                obs["agent_0"]["delta"],
                                obs["agent_0"]["linear_vel_x"],
                                obs["agent_0"]["pose_theta"],
                                obs["agent_0"]["ang_vel_z"],
                                obs["agent_0"]["beta"]]
                              ])        
        frenet_kinematic_pose = env.unwrapped.track.cartesian_to_frenet(curr_state[0, 0], curr_state[0, 1], curr_state[0, 4], s_guess=old_s)
        curr_state_frenet = np.array([
                                [frenet_kinematic_pose[0],
                                frenet_kinematic_pose[1],
                                frenet_kinematic_pose[2],
                                obs["agent_0"]["linear_vel_x"]]
                              ])
        
        if abs(old_s - curr_state_frenet[0, 0]) > 5:
            print("S is not continuous, old_s: {}, new_s: {}".format(old_s, curr_state_frenet[0, 0]))
            completed_laps += 1

        steerv, accl = lmpc.plan(obs["agent_0"])

        obs, timestep, terminated, truncated, infos = env.step(np.array([[steerv, accl]]))
        done = terminated or truncated
        env.render()

        curr_controls = np.vstack((curr_controls, np.array([[steerv, accl]])))
        curr_trajectory = np.vstack((curr_trajectory, curr_state_frenet))

        print("speed: {}, steer vel: {}, accl: {}".format(obs["agent_0"]['linear_vel_x'], steerv, accl))

        if lap_count != completed_laps:
            lap_count = completed_laps
            lap_time = obs["agent_0"]["lap_time"] - total_time
            print("laps_completed: {} | lap_time: {}".format(completed_laps, lap_time))
            total_time = obs["agent_0"]["lap_time"]
            print("Done: {}, Terminated: {}, Truncated: {}".format(done, terminated, truncated))

            # decreasing from length of trajectory to 0
            curr_values = np.arange(curr_trajectory.shape[0]-1, -1, -1).reshape(1,-1)
            lmpc.update_safe_set(curr_trajectory, curr_controls, curr_values)

            curr_trajectory = np.zeros((0, 7))
            curr_controls = np.zeros((0, planner.config.NU))

        if done and (completed_laps != desired_laps):
            done = False
            terminated = False
        
        if completed_laps == desired_laps:
            done = True
    

if __name__ == '__main__':
    main()
