import numpy as np
import gymnasium as gym
from f1tenth_gym.envs import F110Env

from f1tenth_planning.control import SITLMPCPlanner, NonlinearDynamicMPPIPlanner


from render_helpers import make_render_callbacks

def main():
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "control_input": "accl",
            "observation_config": {"type": "dynamic_state"},
        },
        render_mode="human",
    )

    base_controller = NonlinearDynamicMPPIPlanner(track=env.unwrapped.track)
    planner = SITLMPCPlanner(track=env.unwrapped.track, base_controller=base_controller)
    for callback in make_render_callbacks(planner):
        env.unwrapped.add_render_callback(callback)

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

    while not done:
        steerv, accl = planner.plan(obs["agent_0"])
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        env.render()

    # Commit the rollout to the safe set/value function
    planner.complete_iteration()


if __name__ == "__main__":
    main()
