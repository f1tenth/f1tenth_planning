"""Follow-the-Gap Method (FGM).

NOT YET IMPLEMENTED -- this is a deliberate placeholder for planned work.

Note what FGM is in this library's terms: it reads a laser scan and emits an
actuation command directly, so it is a **Controller**, not a Planner (DESIGN.md §2).
There is exactly one Controller type; reading a scan instead of holding a reference
is an implementation detail, not a different kind of controller.

Sketch of the intended implementation:

1. preprocess the LiDAR ranges (clip to a max, smooth, zero a bubble around the
   closest return);
2. find the widest contiguous gap of non-zero ranges;
3. pick a target within that gap (deepest point, or its centre);
4. steer toward the target and scale speed by the steering magnitude.

The previous contents of this file were a ``pyclothoids`` benchmark script that was
byte-identical to ``wall_follow.py`` and contained no follow-the-gap code at all.
"""
from __future__ import annotations

import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionEnum, SteerActionEnum

from f1tenth_planning.control.controller import Controller


class FollowTheGapController(Controller):
    """Reactive gap-following controller. Not implemented yet."""

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
            "Follow-the-Gap is not implemented yet. See the module docstring for the "
            "intended design."
        )

    def compute_control(self, state: dict) -> np.ndarray:
        raise NotImplementedError("Follow-the-Gap is not implemented yet.")
