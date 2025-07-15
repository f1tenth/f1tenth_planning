import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env
import time

from f1tenth_gym.envs.track import Track
from f1tenth_planning.control import Nonlinear_Dynamic_MPC_Planner
from f1tenth_planning.control.config.dynamics_config import (
    fullscale_params,
    f1fifth_params,
    update_config_from_dict,
)

from f1tenth_gym.envs.f110_env import F110Env
import os


def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg_blank",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
            "params": F110Env.f1fifth_vehicle_params(),
        },
        render_mode="unlimited",
    )

    # create planner
    planner = Nonlinear_Dynamic_MPC_Planner(track=env.unwrapped.track, params=f1fifth_params())
    planner.render_waypoints(env.unwrapped.renderer)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

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
        (steerv, accl), _ = planner.plan(obs["agent_0"])
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
