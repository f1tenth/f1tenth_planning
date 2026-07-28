"""Self-describing state and control layouts (DESIGN.md §5.2).

Generic code must not hardcode vector positions. `config.x_min[3]` only means
"minimum velocity" for the two models that happen to put velocity at index 3; a
Frenet-frame or point-mass model would silently mean something else.

A model therefore declares the *names* of its state and control variables, and
everything generic asks by name:

    model.state.idx.v          -> 3        (autocompletes; no string keys)
    model.control.idx.a        -> 1
    model.state.names          -> ("x", "y", "delta", "v", "yaw", ...)

Names are metadata resolved once per tick at wiring time, never inside a hot loop
(measured: ~25 ns per attribute hop, i.e. ~0.004 % of a 10 ms control budget). The
numeric path stays a raw array, because batched maths over arrays is ~12x faster
than per-sample objects and `jax.vmap`/`jit` require array-like values.
"""
from __future__ import annotations

from collections import namedtuple

# How a state variable is read out of an f1tenth_gym observation dict. Names absent
# from this map cannot be filled from an observation and must be supplied another way.
OBSERVATION_KEY_FOR_STATE = {
    "x": "pose_x",
    "y": "pose_y",
    "delta": "delta",
    "v": "linear_vel_x",
    "yaw": "pose_theta",
    "yaw_rate": "ang_vel_z",
    "beta": "beta",
}

# How a state variable is read off a gym Raceline when building a reference. Names
# absent from this map are not carried by a raceline and default to zero.
RACELINE_ATTR_FOR_STATE = {
    "x": "xs",
    "y": "ys",
    "v": "vxs",
    "yaw": "yaws",
    "curvature": "ks",
    "s": "ss",
}


class VariableSpec:
    """The named layout of a state or control vector.

    Args:
        names: variable names in vector order, e.g. ``("x", "y", "delta", "v", "yaw")``.
            Each must be a valid Python identifier so it can be reached as an attribute.
    """

    __slots__ = ("names", "size", "idx")

    def __init__(self, names):
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate variable names: {self.names}")
        for name in self.names:
            if not name.isidentifier():
                raise ValueError(f"{name!r} is not a valid identifier")
        self.size = len(self.names)
        # `idx.v` etc. -- a NamedTuple of ints, so it autocompletes and needs no strings
        self.idx = namedtuple("Index", self.names)(*range(self.size))

    def index(self, name: str) -> int:
        """Position of ``name`` in the vector. Raises KeyError if absent."""
        try:
            return getattr(self.idx, name)
        except AttributeError:
            raise KeyError(
                f"{name!r} is not in this layout; available: {self.names}"
            ) from None

    def has(self, name: str) -> bool:
        return name in self.names

    def view(self, vector):
        """Wrap a vector as a NamedTuple for readable inspection.

        Debugging/boundary convenience only -- never use this inside a hot loop or a
        jitted kernel; the numeric path must stay a raw array.
        """
        if len(vector) != self.size:
            raise ValueError(f"expected {self.size} elements, got {len(vector)}")
        return type(self.idx)(*vector)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"VariableSpec{self.names}"
