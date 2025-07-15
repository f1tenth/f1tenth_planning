import time
import numpy as np
import gymnasium as gym

from f1tenth_planning.control import PurePursuitPlanner
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import (
    fullscale_params,
    f1fifth_params,
    update_config_from_dict,
)

from f1tenth_gym.envs.f110_env import F110Env
import os


def main():
    """
    Pure Pursuit example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg_blank",
            "num_agents": 1,
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="unlimited",
    )

    # create controller
    planner = PurePursuitPlanner(track=env.unwrapped.track)

    planner.render_waypoints(env.unwrapped.renderer)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

    # reset environment
    poses = np.array(
        [
            [
                env.unwrapped.track.raceline.xs[0],
                env.unwrapped.track.raceline.ys[0],
                env.unwrapped.track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    start = time.time()
    while not done:
        steer, speed = planner.plan(
            obs["agent_0"],
            lookahead_distance=0.8,
        )
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )

        print(
            "speed: {}, steering angle: {}".format(
                speed, steer
            )
        )

        done = terminated or truncated
        laptime += timestep
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
