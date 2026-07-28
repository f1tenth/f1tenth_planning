from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.config.controller_config import (
    MPCConfig,
    dynamic_mppi_config,
)
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.solvers import MPPISolver
import jax.numpy as jnp


class DynamicMPPIPlanner(MPCController):
    """
    Convenience class that uses MPPI solver with dynamic bicycle model.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        solver (MPPISolver, optional): MPPI solver object, contains MPPI parameters
        model (DynamicsModel, optional): dynamics model object, contains the vehicle dynamics
        params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none,
        config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        params: DynamicsConfig = None,
        model: DynamicsModel = None,
        config: MPCConfig = None,
        solver: MPPISolver = None,
    ):
        print("Initiailizing Dynamic MPPI Planner (convenience class)")
        if not isinstance(solver, MPPISolver) and solver is not None:
            raise ValueError("Solver must be an instance of MPPISolver")
        if not isinstance(model, DynamicsModel) and model is not None:
            raise ValueError("Model must be an instance of DynamicsModel")
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)
        if config is None:
            config = dynamic_mppi_config()
            # x = [x, y, delta, v, yaw, yaw_rate, beta]
            config.x_min = jnp.array(
                [
                    -jnp.inf,
                    -jnp.inf,
                    params.MIN_STEER,
                    params.MIN_SPEED,
                    -jnp.inf,
                    -jnp.inf,
                    -jnp.inf,
                ]
            )
            config.x_max = jnp.array(
                [
                    jnp.inf,
                    jnp.inf,
                    params.MAX_STEER,
                    params.MAX_SPEED,
                    jnp.inf,
                    jnp.inf,
                    jnp.inf,
                ]
            )
            # u = [delta_v, a]
            config.u_min = jnp.array(
                [
                    params.MIN_DSTEER,
                    params.MIN_ACCEL,
                ]
            )
            config.u_max = jnp.array(
                [
                    params.MAX_DSTEER,
                    params.MAX_ACCEL,
                ]
            )
        if solver is None:
            solver = MPPISolver(config, model)
        super(DynamicMPPIPlanner, self).__init__(
            track,
            solver,
            model,
            params,
        )
        self.sampled_trajectories_render = None

    def render_sampled_trajectories(self, e):
        """
        Render the sampled trajectories on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        # Render the sampled trajectories
        ORANGE = (255, 165, 0)
        if self.solver.samples is not None:
            sampled_trajectories = self.solver.samples[1]
            if self.sampled_trajectories_render is None:
                self.sampled_trajectories_render = [None] * len(
                    sampled_trajectories // 100
                )
            for idx in range(0, len(sampled_trajectories), 100):
                sampled_xy = sampled_trajectories[idx][:, :2]
                if self.sampled_trajectories_render[idx] is None:
                    self.sampled_trajectories_render[idx] = e.render_lines(
                        sampled_xy, color=ORANGE, size=4
                    )
                else:
                    self.sampled_trajectories_render[idx].setData(
                        sampled_xy[:, 0], sampled_xy[:, 1]
                    )
