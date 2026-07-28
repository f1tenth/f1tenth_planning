from f1tenth_planning.control import StanleyController
import numpy as np
import gymnasium as gym

from render_helpers import make_render_callbacks

def main():
    """
    Stanley example. This example uses fixed waypoints throughout the 2 laps.
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
        },
        render_mode="human",
    )
    # create controller
    planner = StanleyController(track=env.unwrapped.track)

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

    # run simulation
    laptime = 0.0
    while not done:
        steer, speed = planner.plan(
            obs["agent_0"],
            k_path=7.0,
        )
        speed = 0.7 * speed
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
