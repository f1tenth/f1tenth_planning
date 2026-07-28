"""Wall following.

NOT YET IMPLEMENTED -- this is a deliberate placeholder for planned work.

Like FGM, wall following reads a laser scan and emits an actuation command directly,
so it is a **Controller**, not a Planner (DESIGN.md §2).

Sketch of the intended implementation:

1. take two range measurements at known angles off one wall;
2. from them, estimate the vehicle's angle to the wall and its current distance;
3. project that distance forward by a lookahead to get the error at the next step;
4. drive the error to a desired offset with a PID controller, and scale speed by the
   steering magnitude.

The previous contents of this file were a ``pyclothoids`` benchmark script that was
byte-identical to ``fgm.py`` and contained no wall-following code at all.
"""
from __future__ import annotations

import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionEnum, SteerActionEnum

from f1tenth_planning.control.controller import Controller


class WallFollowController(Controller):
    """Reactive wall-following controller. Not implemented yet."""

    # Consumes a laser scan rather than a raceline, so it declares no reference fields.
    REFERENCE_FIELDS = ()

    def __init__(self, track=None, params=None, **kwargs):
        super().__init__(
            track,
            params,
            control_mode=(
                SteerActionEnum.Steering_Angle,
                LongitudinalActionEnum.Speed,
            ),
        )
        raise NotImplementedError(
            "Wall following is not implemented yet. See the module docstring for the "
            "intended design."
        )

    def compute_control(self, state: dict) -> np.ndarray:
        raise NotImplementedError("Wall following is not implemented yet.")
