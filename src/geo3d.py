from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import zpk2ss
from sympy.benchmarks.bench_meijerint import normal


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
    normal: Vector3D  # normed normal vector
    distance: float  # from origin (>= 0)

    def calc_distance_to_point(self, p: Vector3D) -> float:
        x, y, z = p
        a, b, c = self.normal
        return abs(a * x + b * y + c * z - self.distance)


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


def calc_plane_from_3_points(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> Plane:
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3

    x21 = x2 - x1
    x31 = x3 - x1
    y21 = y2 - y1
    y31 = y3 - y1
    z21 = z2 - z1
    z31 = z3 - z1

    a = y21 * z31 - z21 * y31
    b = z21 * x31 - x21 * z31
    c = x21 * y31 - y21 * x31
    l = math.hypot(a, b, c)
    na = a / l
    nb = b / l
    nc = c / l

    d = na * x1 + nb * y1 + nc * z1
    return Plane(normal=Vector3D(na, nb, nc), distance=d)


def calc_angle_from_3_points(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> float:
    """ to edges in a path must not connect in a sharp angle """
    p12 = p1 - p2
    p32 = p3 - p2

    cos_phi = p12.dot(p32) / (p12.length * p32.length)
    phi_degree = math.acos(cos_phi) * (180 / math.pi)
    return phi_degree
