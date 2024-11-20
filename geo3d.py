from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

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
class Spherical:
    center: Vector3D
    radius: float


@dataclass
class EndlessCylinder:
    axis: Line3D
    radius: float


