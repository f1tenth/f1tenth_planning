import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env
import time

from f1tenth_planning.control import NonlinearDynamicMPCPlanner
from f1tenth_planning.control.config.dynamics_config import (
    f1tenth_params,
)


from render_helpers import make_render_callbacks

def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
            "params": F110Env.f1tenth_vehicle_params(),
        },
        render_mode="human",
    )
    # create planner
    planner = NonlinearDynamicMPCPlanner(
        track=env.unwrapped.track, params=f1tenth_params()
    )
    for callback in make_render_callbacks(planner):
        env.unwrapped.add_render_callback(callback)

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

    laptime = 0.0
    start = time.time()
    while not done:
        steerv, accl = planner.compute_control(obs["agent_0"])
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
