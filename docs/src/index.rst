F1TENTH Autonomous Racing Software Stack
====================================================

Overview
----------
This is the documentation for F1TENTH's Autonomous Racing Software Stack. The stack will be split into three main topics: Perception, Planning, and Control.

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
**Hongrui Zheng, Luigi Berducci, Renukanandan Tumu, Ahmad Amine**

.. grid:: 2
   :gutter: 4

   .. grid-item-card::

      Installation
      ^^^^^^^^^^^^

      Installation Guide

      .. image:: assets/pip_logo.svg
         :width: 100
         :align: center
      
      +++

      .. button-ref:: install/installation
         :expand:
         :color: secondary
         :click-parent:

         Installation

   .. grid-item-card::

      Quick Start
      ^^^^^^^^^^^

      Example usage

      .. image:: assets/gym.svg
         :width: 100
         :align: center

      +++

      .. button-ref:: usage/index
         :expand:
         :color: secondary
         :click-parent:

         Quick Start

   .. grid-item-card::

      API
      ^^^

      API

      .. image:: assets/gym.svg
         :width: 100
         :align: center

      +++

      .. button-ref:: api/index
         :expand:
         :color: secondary
         :click-parent:

         API

   .. grid-item-card::

      Contribute
      ^^^^^^^^^^

      Contribute

      .. image:: assets/gym.svg
         :width: 100
         :align: center

      +++

      .. button-ref:: contribute/contribute
         :expand:
         :color: secondary
         :click-parent:

         Contribution Guide

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :hidden:

   install/installation
   usage/index
   api/index
   contribute/contribute