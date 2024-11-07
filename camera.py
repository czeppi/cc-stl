from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Camera:

    def __init__(self, distance: float, azimuth: float, elevation: float):
        self.distance = distance
        self._azimuth = azimuth
        self._elevation = elevation

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        self._distance = max(1.0, min(100.0, value))  # bound zoom interval

    @property
    def azimuth(self) -> float:
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value: float) -> None:
        self._azimuth = value

    @property
    def elevation(self) -> float:
        return self._elevation

    @elevation.setter
    def elevation(self, value: float) -> None:
        self._elevation = max(-89.0, min(89.0, value))

    @property
    def xyz(self) -> Tuple[float, float, float]:
        elevation_rad = np.radians(self.elevation)
        azimuth_rad = np.radians(self.azimuth)
        x = self.distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = self.distance * np.sin(elevation_rad)
        z = self.distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        return x, y, z
