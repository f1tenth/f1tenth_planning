"""
AP-MPPI (Adaptive Penalty MPPI) Example with State Limit Constraints.

This example demonstrates the AP-MPPI controller with automatic state limit constraints
based on vehicle parameters (steering angle and velocity bounds).

The AP-MPPI solver samples over penalty multipliers (lambdas) to find trajectories that
satisfy constraints while maximizing returns. If feasible trajectories exist, it selects
the one with maximum returns; otherwise, it falls back to minimizing constraint violations.
"""

import os
import time

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control import NonlinearDynamicAPMPPIPlanner
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.controllers.mpc.ap_mppi.dynamic_ap_mppi import (
    make_state_min_constraint,
    make_state_max_constraint,
)


from render_helpers import make_render_callbacks

def main():
    """
    AP-MPPI example with state limit constraints.

    The planner automatically sets up constraints for:
    - Steering angle limits (from vehicle params)
    - Velocity limits (from vehicle params)

    You can also define custom constraints by creating functions with signature:
        constraint(x, u) -> (N,)
    where positive values indicate constraint violation.
    """

    # Create environment
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

    # Get track from environment
    waypoints_track = env.unwrapped.track

    # Get vehicle parameters
    params = f1tenth_params()

    # === STATE BOUNDS STRATEGY ===
    # There are TWO types of state bounds in AP-MPPI:
    #
    # 1. x_clip_min/x_clip_max: STABILITY CLIPPING during rollouts
    #    - Mirrors what the system (vehicle) physically enforces
    #      e.g., if we command too high a steering angle, the vehicle
    #      servo will saturate at its max steering angle.
    #    - Clipping prevents numerical issues during rollouts as 
    #      unbounded steering can quickly blow up to infinity.
    #    - Clipped states are NOT penalized in the cost function:
    #         - As the rollouts never exceed these bounds, no penalties are incurred.
    #
    # 2. Constraint functions: SOFT PENALTIES for limit violations
    #    - Constraint penalties guide the optimizer toward feasible trajectories
    #    - Violations are detected and penalized, not masked by clipping
    #    - Crucial to NOT clip states that we want to constrain!
    #
    # For velocity constraints to work, x_clip_max[3] must be > x_max[3]
    # so the optimizer can "see" velocity violations in the rollouts.

    # Build state constraints from vehicle parameters
    # x = [x, y, delta, v, yaw, yaw_rate, beta]
    x_min_constrained = np.array([
        -np.inf,           # x position: unconstrained
        -np.inf,           # y position: unconstrained  
        params.MIN_STEER,  # steering angle
        params.MIN_SPEED,  # velocity
        -np.inf,           # yaw: unconstrained
        -np.inf,           # yaw_rate: unconstrained
        -np.inf,           # beta: unconstrained
    ])
    x_max_constrained = np.array([
        np.inf,            # x position: unconstrained
        np.inf,            # y position: unconstrained
        params.MAX_STEER,  # steering angle
        5.0,               # for testing, set max speed to 5 m/s
        np.inf,            # yaw: unconstrained
        np.inf,            # yaw_rate: unconstrained
        np.inf,            # beta: unconstrained
    ])

    constraints = [
        make_state_min_constraint(x_min_constrained),
        make_state_max_constraint(x_max_constrained),
    ]

    # Lambda range must be (n_constraints, 2) - one range per constraint
    # Higher lambda values = stronger constraint enforcement
    # Use high values (0-1000) for strict constraint satisfaction
    n_constraints = len(constraints)
    lambdas_sample_range = np.array([[0.0, 1000.0]] * n_constraints)

    # Create config with custom N, dt and constraints
    config = dynamic_ap_mppi_config(
        constraints=constraints,
        n_lambdas=32,  # More lambda samples for better constraint handling
        lambdas_sample_range=lambdas_sample_range,
    )
    config.dt = 0.05  # 20 Hz control
    config.N = 20  # 20-step horizon
    config.n_samples = 1024  # More control samples for better coverage
    config.n_iterations = 3  # More iterations to converge

    # Set physical state/control bounds on the config (used for MPPI rollout clipping)
    # For velocity, use +inf so constraint violations are visible in rollouts
    config.x_min = np.array([
        -np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED,
        -np.inf, -np.inf, -np.inf,
    ])
    config.x_max = np.array([
        np.inf, np.inf, params.MAX_STEER, np.inf,  # velocity: +inf so constraints work
        np.inf, np.inf, np.inf,
    ])
    config.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
    config.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])

    planner = NonlinearDynamicAPMPPIPlanner(
        track=waypoints_track,
        params=params,
        config=config,
        # Operational speed limit for reference trajectory (separate from physical limits)
        ref_velocity_bounds=(params.MIN_SPEED, 5.0),
    )

    # Add render callbacks
    for callback in make_render_callbacks(planner, include_samples=True):
        env.unwrapped.add_render_callback(callback)

    # Reset environment at start of track
    poses = np.array(
        [
            [
                waypoints_track.raceline.xs[0],
                waypoints_track.raceline.ys[0],
                waypoints_track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    env.render()

    # Print constraint info
    n_constraints = planner.solver.config.n_constraints
    print(f"\n=== AP-MPPI Configuration ===")
    print(f"Number of constraints: {n_constraints}")
    print(f"Number of lambda samples: {planner.solver.config.n_lambdas}")
    print(f"Lambda sample range:\n{planner.solver.config.lambdas_sample_range}")
    print(f"Horizon: {planner.solver.config.N}")
    print(f"dt: {planner.solver.config.dt}")
    print(f"Max speed (reference): {planner.ref_v_max} m/s")
    print(f"Max steer: {np.rad2deg(params.MAX_STEER):.1f} deg")
    print("=" * 30 + "\n")

    laptime = 0.0
    start = time.time()

    while not done:
        # Plan next control action
        steerv, accl = planner.compute_control(obs["agent_0"])

        # Step environment
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        laptime += timestep
        env.render()

        # Print status
        speed = obs["agent_0"]["linear_vel_x"]
        print(
            f"speed: {speed:.2f} m/s, steer vel: {steerv:.3f}, accl: {accl:.2f}"
        )

    print(f"\nSim elapsed time: {laptime:.2f}s")
    print(f"Real elapsed time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
