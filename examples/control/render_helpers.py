"""Render callbacks for the examples.

Rendering lives with the *caller*, not in the library (DESIGN.md §9). Controllers
just expose what they computed as plain attributes, and the caller draws whatever it
cares about. That keeps the library free of any dependency on a particular renderer,
and means a new controller cannot "forget" to implement rendering.

The attributes these read:

===========================  ===================================================
``controller.waypoints``     (N, nx) global reference; columns 0/1 are x/y
``controller.local_plan``    (N+1, 2) the reference slice being tracked
``controller.control_solution``  (2, N+1) predicted x/y, MPC family only
``solver.samples``           MPPI rollouts, ``(a, s, r)``; ``s`` is
                             (n_samples, N, nx)
===========================  ===================================================

Usage::

    from render_helpers import make_render_callbacks

    for callback in make_render_callbacks(planner):
        env.unwrapped.add_render_callback(callback)
"""
import numpy as np

GLOBAL_PLAN_COLOR = (0, 128, 0)
LOCAL_PLAN_COLOR = (0, 0, 128)
SOLUTION_COLOR = (128, 0, 0)
SAMPLE_COLOR = (64, 64, 64)


def _as_numpy(array):
    """Bring a jax device array back to numpy; pass numpy through untouched."""
    return np.asarray(array) if array is not None else None


def make_waypoint_callback(controller, color=GLOBAL_PLAN_COLOR):
    """Draw the controller's global reference once, then keep it updated."""
    handle = {}

    def render(e):
        if controller.waypoints is None:
            return
        points = np.asarray(controller.waypoints)[:, :2]
        if "h" not in handle:
            handle["h"] = e.render_closed_lines(points, color=color, size=1)
        else:
            handle["h"].setData(points)

    return render


def make_local_plan_callback(controller, color=LOCAL_PLAN_COLOR):
    """Draw the slice of reference the controller is currently tracking."""
    handle = {}

    def render(e):
        plan = _as_numpy(getattr(controller, "local_plan", None))
        if plan is None:
            return
        if "h" not in handle:
            handle["h"] = e.render_closed_lines(plan, color=color, size=4)
        else:
            handle["h"].setData(plan)

    return render


def make_solution_callback(controller, color=SOLUTION_COLOR):
    """Draw the controller's own predicted/target points."""
    handle = {}

    def render(e):
        solution = _as_numpy(getattr(controller, "control_solution", None))
        if solution is None:
            return
        # MPC exposes (2, N+1); the classical controllers expose a single point
        points = solution.T if solution.ndim == 2 and solution.shape[0] == 2 else solution
        points = np.atleast_2d(points)
        if "h" not in handle:
            handle["h"] = e.render_points(points, color=color, size=4)
        else:
            handle["h"].setData(points)

    return render


def make_sampled_trajectories_callback(controller, stride=100, color=SAMPLE_COLOR):
    """Draw every `stride`-th MPPI rollout, if the controller's solver kept samples.

    MPPI draws ~1024 rollouts per step; drawing them all is unreadable and slow, so
    this subsamples.
    """
    handles = {}

    def render(e):
        solver = getattr(controller, "solver", None)
        samples = getattr(solver, "samples", None)
        if not samples:
            return
        states = _as_numpy(samples[1])          # (n_samples, N, nx)
        if states is None or states.ndim != 3:
            return
        for slot, i in enumerate(range(0, states.shape[0], stride)):
            xy = states[i, :, :2]
            if slot not in handles:
                handles[slot] = e.render_lines(xy, color=color, size=1)
            else:
                handles[slot].setData(xy)

    return render


def make_render_callbacks(controller, include_samples=False):
    """The callbacks used by every example: global plan, local plan, solution.

    Args:
        controller: any controller exposing the attributes documented above.
        include_samples: also draw MPPI rollouts (MPPI/AP-MPPI only).
    """
    callbacks = [
        make_waypoint_callback(controller),
        make_local_plan_callback(controller),
        make_solution_callback(controller),
    ]
    if include_samples:
        callbacks.append(make_sampled_trajectories_callback(controller))
    return callbacks
