

import numpy as np
import gymnasium as gym
from f110_gym.envs import F110Env
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from f1tenth_planning.control.mppi.kinematic_mppi import KMPPIPlanner

def main():
    # create environment
    env: F110Env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
        },
        render_mode="human",
    )

    # create planner
    planner = KMPPIPlanner(track=env.track, debug=False)
    planner.config.dlk = (
        env.track.raceline.ss[1] - env.track.raceline.ss[0]
    )  # waypoint spacing
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_sampled_trajectories)
    env.unwrapped.add_render_callback(planner.render_optimal_trajectory)

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

    laptime = 0.0
    start = time.time()
    while not done:
        steerv, accl = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()

        print(
            "speed: {}, steer vel: {}, accl: {}".format(
                obs["agent_0"]["linear_vel_x"], steerv, accl
            )
        )

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
