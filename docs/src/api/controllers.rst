Implemented Controllers
===========================

Controllers produces vehicle controls that tracks a given reference trajectory.
All controllers are based on the following abstract class.

.. currentmodule:: f1tenth_planning.control.controller

.. autosummary::
    :toctree: _generated/

    Controller


Pure Pursuit
----------------

Pure Pursuit controller: [TODO] description

.. currentmodule:: f1tenth_planning.control.pure_pursuit.pure_pursuit

.. autosummary::
    :toctree: _generated/

    PurePursuitPlanner


Stanley
----------------

Stanley controller: [TODO] description

.. currentmodule:: f1tenth_planning.control.stanley.stanley

.. autosummary::
    :toctree: _generated/

    StanleyPlanner


LQR
----------------

LQR controller: [TODO] description

.. currentmodule:: f1tenth_planning.control.lqr.lqr

.. autosummary::
    :toctree: _generated/

    LQRPlanner


Kinematic MPC
----------------

Kinematic MPC controller: [TODO] description

.. currentmodule:: f1tenth_planning.control.kinematic_mpc.kinematic_mpc

.. autosummary::
    :toctree: _generated/

    mpc_config
    State
    KMPCPlanner


Dynamic MPC
----------------

Dynamic MPC controller: [TODO] description

.. currentmodule:: f1tenth_planning.control.dynamic_mpc.dynamic_mpc

.. autosummary::
    :toctree: _generated/

    mpc_config
    State
    STMPCPlanner