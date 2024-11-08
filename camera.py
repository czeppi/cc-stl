from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PySide6.QtGui import QMatrix4x4


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

    @staticmethod
    def _calc_distance_azimuth_elevation(x: float, y: float, z: float) -> Tuple[float, float, float]:
        distance = math.hypot(x, y, z)
        xz = math.hypot(x, z)
        elevation = 180 / math.pi * math.atan(y / xz)
        azimuth = 180 / math.pi * math.atan2(x, z)
        return distance, azimuth, elevation

    def create_view_matrix(self) -> QMatrix4x4:
        """ cals view matrix in dependency of the camera
        """
        eye = np.array(self.xyz, dtype=np.float32)
        up_vector = np.array([0, 1, 0], dtype=np.float32)

        z_axis = 1 * eye
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(up_vector, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        m = QMatrix4x4(
            x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, eye),
            y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, eye),
            z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, eye),
            0, 0, 0, 1
        )
        return m
