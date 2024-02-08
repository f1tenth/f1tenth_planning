import numpy as np
import gymnasium as gym
from f1tenth_planning.control.lqr.lqr import LQRPlanner


def main():
    """
    LQR example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # create environment
    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "speed",
            "observation_config": {"type": "kinematic_state"},
            "reset_config": {"type": "rl_grid_random"},
        },
        render_mode="human",
    )

    # create planner
    track = env.unwrapped.track
    planner = LQRPlanner(track=track)

    env.add_render_callback(planner.render_waypoints)
    env.add_render_callback(planner.render_local_plan)
    env.add_render_callback(planner.render_closest_point)

    obs, info = env.reset()
    done = False
    env.render()

    # run simulation
    laptime = 0.0
    while not done:
        action = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(np.array([action]))
        done = terminated or truncated
        laptime += timestep
        env.render()
    print("Sim elapsed time:", laptime)


if __name__ == "__main__":
    main()
