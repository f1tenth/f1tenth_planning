import numpy as np
import gymnasium as gym
from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitController


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
            "control_input": "speed",
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="human",
    )

    # create planner
    raceline = env.unwrapped.track.raceline
    waypoints = np.stack([raceline.xs, raceline.ys, raceline.vxs], axis=1)
    controller = PurePursuitController(env.unwrapped.track)

    env.add_render_callback(controller.render_waypoints)
    env.add_render_callback(controller.render_local_plan)
    env.add_render_callback(controller.render_lookahead_point)

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
        steer, speed = controller.plan(obs['agent_0'])
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steer, speed]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
