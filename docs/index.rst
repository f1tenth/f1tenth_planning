.. F1TENTH Autonomous Racing Software Stack documentation master file, created by
   sphinx-quickstart on Mon May  2 18:18:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

F1TENTH Autonomous Racing Software Stack
====================================================

Overview
----------
This is the documentation for F1TENTH's Autonomous Racing Software Stack. The stack will be split into three main topics: Perception, Planning, and Control.
We'll try to include all software algorithms we've encountered, tested, and used in our research in Autonomous Racing. This repo is constantly being updated. If you have algorithms that you've used in your Autonomous Racing applications and wish to make it open source, please consider :ref:`contribute <doc_contribution_guide>` to this repo.

GitHub repo to the source code: https://github.com/f1tenth/f1tenth_planning

Citing
---------
If you use any open source implementations mentioned in this repo NOT authored by us, please cite the original authors and give proper credits. All original author information will be included in the documentation if not authored by us.

Otherwise, if you find this repo helpful in your work, please consider citing:

.. code::

  @inproceedings{o2020textscf1tenth,
    title={textscF1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
    author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
    booktitle={NeurIPS 2019 Competition and Demonstration Track},
    pages={77--89},
    year={2020},
    organization={PMLR}
  }

Maintainers
--------------
**Hongrui Zheng**: hongruiz AT seas DOT upenn DOT edu

**Johannes Betz**: joebetz AT seas DOT upenn DOT edu

.. toctree::
   :maxdepth: 1
   :caption: Perception
   :name: sec-perception
   :hidden:

   perception/particle_filter


.. toctree::
   :maxdepth: 1
   :caption: Planning
   :name: sec-planning
   :hidden:

   planning/wall_follow
   planning/fgm
   planning/lane_switcher
   planning/lattice_planner
   planning/graph_planner


.. toctree::
   :maxdepth: 1
   :caption: Control
   :name: sec-control
   :hidden:

   control/pure_pursuit
   control/stanley
   control/lqr
   control/kinematic_mpc


.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :name: sec-contribute
   :hidden:

   contribute


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
