from dataclasses import dataclass, field
import numpy as np
from f1tenth_gym.envs.f110_env import F110Env
import math

@dataclass
class dynamics_config:
    """
    Vehicle dynamics configuration dataclass.
    This dataclass contains the vehicle dynamics parameters used for control and planning.
    
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
    MIN_STEER: float
    MAX_STEER: float
    MIN_DSTEER: float
    MAX_DSTEER: float
    MAX_SPEED: float
    MIN_SPEED: float
    MAX_ACCEL: float
    MIN_ACCEL: float
    WHEELBASE: float

    # Vehicle parameters
    MU: float
    C_SF: float
    C_SR: float
    BF: float
    BR: float
    DF: float
    DR: float
    CF: float
    CR: float

    LF: float
    LR: float
    H: float
    M: float
    I: float

def _dynamics_config_from_gym_params(gym_params):
    """
    Generate a dynamics configuration object from gym environment parameters.
    This function extracts and computes various vehicle dynamics parameters 
    from a dictionary of gym environment parameters. It also calculates 
    additional parameters if they are not explicitly provided in the input.
    Args:
        gym_params (dict): A dictionary containing the following keys:
            - "s_min" (float): Minimum steering angle.
            - "s_max" (float): Maximum steering angle.
            - "sv_min" (float): Minimum steering velocity.
            - "sv_max" (float): Maximum steering velocity.
            - "v_max" (float): Maximum speed.
            - "v_min" (float): Minimum speed.
            - "a_max" (float): Maximum acceleration.
            - "a_min" (float, optional): Minimum acceleration. Defaults to -a_max.
            - "lf" (float): Distance from the center of mass to the front axle.
            - "lr" (float): Distance from the center of mass to the rear axle.
            - "mu" (float): Friction coefficient.
            - "C_Sf" (float): Linear cornering stiffness for the front tires.
            - "C_Sr" (float): Linear cornering stiffness for the rear tires.
            - "h" (float): Height of the center of mass.
            - "m" (float): Mass of the vehicle.
            - "I" (float): Moment of inertia of the vehicle.
            - Optional keys:
                - "Bf" (float): Tire stiffness factor for the front tires.
                - "Br" (float): Tire stiffness factor for the rear tires.
                - "Df" (float): Peak vertical force on the front tires scaled by friction.
                - "Dr" (float): Peak vertical force on the rear tires scaled by friction.
                - "Cf" (float): Pacejka shape factor for the front tires.
                - "Cr" (float): Pacejka shape factor for the rear tires.
    Returns:
        dynamics_config: An object containing the vehicle dynamics parameters. Check the `dynamics_config` class for more details.
    Notes:
        - If certain optional parameters are not provided in `gym_params`, 
          they are computed using typical values or derived from other parameters.
        - The function assumes a gravitational acceleration of 9.81 m/s^2.
    """
    MIN_STEER = gym_params["s_min"]
    MAX_STEER = gym_params["s_max"]
    MIN_DSTEER = gym_params["sv_min"]
    MAX_DSTEER = gym_params["sv_max"]
    MAX_SPEED = gym_params["v_max"]
    MIN_SPEED = gym_params["v_min"]
    MAX_ACCEL = gym_params["a_max"]
    MIN_ACCEL = gym_params["a_min"] if "a_min" in gym_params else -gym_params["a_max"]
    WHEELBASE = gym_params["lf"] + gym_params["lr"]

    MU = gym_params["mu"]
    C_SF = gym_params["C_Sf"]
    C_SR = gym_params["C_Sr"]
    LF = gym_params["lf"]
    LR = gym_params["lr"]
    H = gym_params["h"]
    M = gym_params["m"]
    I = gym_params["I"]

    if "bf" in gym_params:
        BF = gym_params["Bf"]
    if "br" in gym_params:
        BR = gym_params["Br"]
    if "df" in gym_params:
        DF = gym_params["Df"]
    if "dr" in gym_params:
        DR = gym_params["Dr"]
    if "cf" in gym_params:
        CF = gym_params["Cf"]
    if "cr" in gym_params:
        CR = gym_params["Cr"]
        
    g = 9.81
    Fz_total = M * g
    # Distribute weight based on distances from the center of mass
    Fzf = Fz_total * LR / (LF + LR)
    Fzr = Fz_total * LF / (LF + LR)

    if "df" not in gym_params:
        # Compute peak vertical force on the front tires scaled by friction
        DF = MU * Fzf

    if "dr" not in gym_params:
        # Compute peak vertical force on the rear tires scaled by friction
        DR = MU * Fzr

    if "cf" not in gym_params:
        # Typical Pacejka shape factor for the front tires (commonly around 1.3)
        CF = 1.3

    if "cr" not in gym_params:
        # Typical Pacejka shape factor for the rear tires (commonly around 1.3)
        CR = 1.3

    if "bf" not in gym_params:
        # Compute tire stiffness factor for the front tires using the linear cornering stiffness
        BF = C_SF / (CF * DF) if DF != 0 else 0.0

    if "br" not in gym_params:
        # Compute tire stiffness factor for the rear tires using the linear cornering stiffness
        BR = C_SR / (CR * DR) if DR != 0 else 0.0

    return dynamics_config(
        MIN_STEER=MIN_STEER,
        MAX_STEER=MAX_STEER,
        MIN_DSTEER=MIN_DSTEER,
        MAX_DSTEER=MAX_DSTEER,
        MAX_SPEED=MAX_SPEED,
        MIN_SPEED=MIN_SPEED,
        MAX_ACCEL=MAX_ACCEL,
        MIN_ACCEL=MIN_ACCEL,
        WHEELBASE=WHEELBASE,
        MU=MU,
        C_SF=C_SF,
        C_SR=C_SR,
        BF=BF,
        BR=BR,
        DF=DF,
        DR=DR,
        CF=CF,
        CR=CR,
        LF=LF,
        LR=LR,
        H=H,
        M=M,
        I=I
    )

def f1tenth_params():
    """
    Generate a `dynamics_config` object for the f1tenth vehicle.
    This function creates a `dynamics_config` object using the default vehicle parameters
    for the f1tenth vehicle used in the F1TENTH gym.
    
    Returns:
        dynamics_config: An object containing the vehicle dynamics parameters for the f1tenth vehicle.
    """
    return _dynamics_config_from_gym_params(F110Env.f1tenth_vehicle_params())

def f1fifth_params():
    """
    Generate a `dynamics_config` object for the f1fifth vehicle. 
    This function creates a `dynamics_config` object using the default vehicle parameters
    for the f1fifth vehicle used in the F1TENTH gym.
    
    Returns:
        dynamics_config: An object containing the vehicle dynamics parameters for the f1fifth vehicle.
    """
    return _dynamics_config_from_gym_params(F110Env.f1fifth_vehicle_params())

def fullscale_params():
    """"
    Generate a `dynamics_config` object for the fullscale vehicle.
    This function creates a `dynamics_config` object using the default vehicle parameters
    for the fullscale vehicle used in the F1TENTH gym.

    Returns:
        dynamics_config: An object containing the vehicle dynamics parameters for the fullscale vehicle.
    """
    return _dynamics_config_from_gym_params(F110Env.fullscale_vehicle_params()) 
