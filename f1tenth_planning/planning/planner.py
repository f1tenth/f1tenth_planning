"""The Planner interface (DESIGN.md §4).

A ``Planner`` produces a *reference* -- a path or trajectory for a controller to
track. A ``Controller`` produces an *actuation command*. They are separate entities
with no shared base class, because they do different jobs and a planner cannot drive
the car on its own.

**Controllers do not own planners.** The caller wires them together explicitly::

    reference = planner.plan(observation, track=track)
    controller.update(reference=reference)
    u = controller.compute_control(observation)

That inversion is deliberate: it means the two never have to agree on a type at the
API boundary, so this interface can stay minimal until real planners land, and glue
code adapts whatever a planner emits into what a controller accepts.

This interface is intentionally loose for now. Firming up the return type is deferred
until a real planner is ported -- a full ``Raceline`` is heavy for a short local path
(it spline-fits and rasterises an occupancy grid), and a lane-switcher really emits
"which lane", so picking one representation today would be guesswork.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Planner(ABC):
    """Base class for planners: observation (+ context) -> reference."""

    @abstractmethod
    def plan(self, state: dict, **context):
        """Produce a reference for a controller to track.

        Args:
            state (dict): observation as returned from the environment.
            **context: anything else the planner needs that is not ego state --
                the track/map, opponent poses, a laser scan, a cost map. Planners
                need richer input than controllers, and what they need varies, so
                this is deliberately open.

        Returns:
            A reference suitable for passing to ``Controller.update(reference=...)``.
            Typically a gym ``Raceline`` or an ``(N, m)`` waypoint array.
        """
        raise NotImplementedError("plan not implemented")

    def update(self, *, config=None, **kwargs) -> None:
        """Reconfigure the planner at runtime. Mirrors ``Controller.update``."""
        if config is not None:
            self.config = config

    def reset(self) -> None:
        """Clear per-episode state. Default is a no-op."""
        return None
