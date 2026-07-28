from f1tenth_planning.control.controllers.lmpc.base import LMPCController
from f1tenth_planning.control.controllers.lmpc.components import (
    SimpleSafeSetStore,
    SimpleValueFunctionModel,
)
from f1tenth_planning.control.controllers.lmpc.manager import LMPCIterationManager
from f1tenth_planning.control.solvers import APMPPISolver
from f1tenth_planning.control.config.controller_config import (
    LMPCConfig,
    APMPPIConfig,
)
from f1tenth_planning.control.config.dynamics_config import f1tenth_params


class SITLMPCPlanner(LMPCController):
    """
    Convenience planner wiring LMPCController with the AP-MPPI stub components.
    """

    def __init__(
        self,
        track,
        params=None,
        lmpc_cfg: LMPCConfig = None,
        ap_mppi_cfg: APMPPIConfig = None,
        base_controller=None,
    ):
        lmpc_cfg = lmpc_cfg or LMPCConfig()
        ap_mppi_cfg = ap_mppi_cfg or APMPPIConfig()
        solver = APMPPISolver(ap_mppi_cfg)
        params = params or (
            track.vehicle_params
            if hasattr(track, "vehicle_params")
            else f1tenth_params()
        )
        super().__init__(
            track,
            solver,
            params=params,
            lmpc_cfg=lmpc_cfg,
            base_controller=base_controller,
        )


__all__ = [
    "LMPCController",
    "SITLMPCPlanner",
    "SimpleSafeSetStore",
    "SimpleValueFunctionModel",
    "LMPCIterationManager",
]
