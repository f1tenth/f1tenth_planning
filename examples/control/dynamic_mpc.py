import numpy as np
import gymnasium as gym
import time

from f1tenth_planning.control.dynamic_mpc.dynamic_mpc import STMPCPlanner


def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
            "reset_config": {"type": "rl_grid_random"},
        },
        render_mode="human",
    )

    # create planner
    track = env.unwrapped.track
    planner = STMPCPlanner(track=track, debug=False)
    planner.config.dlk = track.raceline.ss[1] - track.raceline.ss[0]
    planner.config.dl = planner.config.dlk
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_mpc_sol)

    env.add_render_callback(planner.render_waypoints)

    obs, info = env.reset()
    done = False
    env.render()

    laptime = 0.0
    start = time.time()
    while not done:
        steerv, accl = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, info = env.step(
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
