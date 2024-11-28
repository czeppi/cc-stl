from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other: Vector3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def length(self) -> float:
        return math.hypot(self.x, self.y, self.z)

    def norm(self) -> Vector3D:
        s = self.length
        return Vector3D(self.x / s, self.y / s, self.z / s)


@dataclass
class Plane:
    normal: Vector3D
    distance: float  # from origin (>= 0)


@dataclass
class Line3D:
    pass  # todo


@dataclass
class Sphere:
    center: Vector3D
    radius: float


@dataclass
class EndlessCylinder:
    axis: Line3D
    radius: float


def calc_sphere_from_4_points(p1: Vector3D, p2: Vector3D, p3: Vector3D, p4: Vector3D) -> Optional[Sphere]:
    A = np.array([
        [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z],
        [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z],
        [p4.x - p1.x, p4.y - p1.y, p4.z - p1.z],
    ])

    B = 0.5 * np.array([
        p2.dot(p2) - p1.dot(p1),
        p3.dot(p3) - p1.dot(p1),
        p4.dot(p4) - p1.dot(p1),
    ])

    try:
        center_array = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return

    center = Vector3D(*center_array)
    radius = (p1 - center).length
    return Sphere(center=center, radius=radius)
