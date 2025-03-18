from dataclasses import dataclass, field
import numpy as np
from f1tenth_gym.envs.f110_env import F110Env
import math

@dataclass
class dynamics_config:
    """
    Vehicle dynamics parameters. Defaults are extracted from the f1tenth gym.
    
    Args:
        MIN_STEER: float - minimum steering angle [rad]
        MAX_STEER: float - maximum steering angle [rad]
        MIN_DSTEER: float - minimum steering rate [rad/s]
        MAX_DSTEER: float - maximum steering rate [rad/s]
        MAX_SPEED: float - maximum speed [m/s]
        MIN_SPEED: float - minimum speed [m/s]
        MAX_ACCEL: float - maximum acceleration [m/s^2]
        MIN_ACCEL: float - minimum acceleration [m/s^2]

        WHEELBASE: float - distance between front and rear axles [m]
        MU: float - friction coefficient
        C_SF: float - cornering stiffness front
        C_SR: float - cornering stiffness rear
        LF: float - distance from center of mass to front axle
        LR: float - distance from center of mass to rear axle
        H: float - height of center of mass
        M: float - mass of the vehicle
        I: float - moment of inertia
    """
    MIN_STEER: float = None
    MAX_STEER: float = None
    MIN_DSTEER: float = None
    MAX_DSTEER: float = None
    MAX_SPEED: float = None
    MIN_SPEED: float = None
    MAX_ACCEL: float = None
    MIN_ACCEL: float = None
    WHEELBASE: float = None

    # Vehicle parameters
    MU: float = None
    C_SF: float = None
    C_SR: float = None
    BF: float = None
    BR: float = None
    DF: float = None
    DR: float = None
    CF: float = None
    CR: float = None

    LF: float = None
    LR: float = None
    H: float = None
    M: float = None
    I: float = None

    def __post_init__(self):
        # Extract default values from the f1tenth gym. Adjust the import
        # path and function as needed for your project.
        defaults = F110Env.f1tenth_vehicle_params()

        self.MIN_STEER = defaults["s_min"]
        self.MAX_STEER = defaults["s_max"]
        self.MIN_DSTEER = defaults["sv_min"]
        self.MAX_DSTEER = defaults["sv_max"]
        self.MAX_SPEED = defaults["v_max"]
        self.MIN_SPEED = defaults["v_min"]
        self.MAX_ACCEL = defaults["a_max"]
        self.MIN_ACCEL = defaults["a_min"] if "a_min" in defaults else -defaults["a_max"]
        self.WHEELBASE = defaults["lf"] + defaults["lr"]

        self.MU = defaults["mu"]
        self.C_SF = defaults["C_Sf"]
        self.C_SR = defaults["C_Sr"]
        self.LF = defaults["lf"]
        self.LR = defaults["lr"]
        self.H = defaults["h"]
        self.M = defaults["m"]
        self.I = defaults["I"]

        if "bf" in defaults:
            self.BF = defaults["Bf"]
        if "br" in defaults:
            self.BR = defaults["Br"]
        if "df" in defaults:
            self.DF = defaults["Df"]
        if "dr" in defaults:
            self.DR = defaults["Dr"]
        if "cf" in defaults:
            self.CF = defaults["Cf"]
        if "cr" in defaults:
            self.CR = defaults["Cr"]
            
        g = 9.81
        Fz_total = self.M * g
        # Distribute weight based on distances from the center of mass
        Fzf = Fz_total * self.LR / (self.LF + self.LR)
        Fzr = Fz_total * self.LF / (self.LF + self.LR)

        if "df" not in defaults:
            # Compute peak vertical force on the front tires scaled by friction
            self.DF = self.MU * Fzf

        if "dr" not in defaults:
            # Compute peak vertical force on the rear tires scaled by friction
            self.DR = self.MU * Fzr

        if "cf" not in defaults:
            # Typical Pacejka shape factor for the front tires (commonly around 1.3)
            self.CF = 1.3

        if "cr" not in defaults:
            # Typical Pacejka shape factor for the rear tires (commonly around 1.3)
            self.CR = 1.3

        if "bf" not in defaults:
            # Compute tire stiffness factor for the front tires using the linear cornering stiffness
            self.BF = self.C_SF / (self.CF * self.DF) if self.DF != 0 else 0.0

        if "br" not in defaults:
            # Compute tire stiffness factor for the rear tires using the linear cornering stiffness
            self.BR = self.C_SR / (self.CR * self.DR) if self.DR != 0 else 0.0