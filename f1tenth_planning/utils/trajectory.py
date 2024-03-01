from typing import Optional, Sequence, Union
import numpy as np
import csv
from dataclasses import dataclass


@dataclass
class Trajectory:
    positions: Sequence[Sequence[float]]
    poses: Sequence[float] = None
    theta: Sequence[float] = None
    v: Sequence[float] = None
    a: Sequence[float] = None
    steer: Sequence[float] = None
    steer_v: Sequence[float] = None

    def from_file():
        pass

    def to_file():
        pass

    def subsample():
        pass

    def render():
        pass
