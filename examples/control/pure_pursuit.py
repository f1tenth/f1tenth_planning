import numpy as np
import gymnasium as gym

from f1tenth_planning.control import PurePursuitPlanner
from f1tenth_planning.control.config.dynamics_config import (
    f1tenth_params,
)

from f1tenth_gym.envs.f110_env import F110Env


from render_helpers import make_render_callbacks

def main():
    """
    Pure Pursuit example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "kinematic_state"},
            "params": F110Env.f1tenth_vehicle_params(),
        },
        render_mode="human",
    )
    # create controller
    planner = PurePursuitPlanner(track=env.unwrapped.track, params=f1tenth_params())

    for callback in make_render_callbacks(planner):
        env.unwrapped.add_render_callback(callback)

    # reset environment
    track = env.unwrapped.track
    poses = np.array(
        [
            [
                track.raceline.xs[0],
                track.raceline.ys[0],
                track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    while not done:
        steer, speed = planner.plan(
            obs["agent_0"],
        )
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
