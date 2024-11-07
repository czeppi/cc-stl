from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Camera:

    def __init__(self, distance: float, azimuth: float, elevation: float):
        self._distance = distance
        self._azimuth = azimuth  # horizontal angle in degree
        self._elevation = elevation  # vertical angle in degree

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

    def rotate_vertical(self, delta_angle: float) -> None:
        """ delta_angle in degree
            y is up-axis
        """
        r = self.distance
        x1, y1, z1 = self.xyz
        xz1 = math.hypot(x1, z1)
        angle_old = math.atan(y1 / xz1)
        angle_new = angle_old + delta_angle * math.pi / 180

        xz2 = r * math.cos(angle_new)
        y2 = r * math.sin(angle_new)

        x2 = xz2 / xz1 * x1
        z2 = xz2 / xz1 * z1

        self._distance, self._azimuth, self._elevation = self._calc_distance_azimuth_elevation(x2, y2, z2)

    @staticmethod
    def _calc_distance_azimuth_elevation(x: float, y: float, z: float) -> Tuple[float, float, float]:
        distance = math.hypot(x, y, z)
        xz = math.hypot(x, z)
        elevation = 180 / math.pi * math.atan(y / xz)
        azimuth = 180 / math.pi * math.atan2(x, z)
        return distance, azimuth, elevation
