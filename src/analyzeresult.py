from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Set

from trimesh.primitives import Sphere

from geo3d import Plane, EndlessCylinder, Line3D


@dataclass
class AnalyzeResult:
    surface_parts: List[SurfacePart]
    edge_parts: List[EdgePart]


class SurfaceType(Enum):
    PLANE=1
    SPHERE=2
    CYLINDER=3


@dataclass
class SurfacePart:  # or PartialSurface
    type: SurfaceType
    triangle_indices: Set[int]   # face indices in mesh
    form: Plane | Sphere | EndlessCylinder


class EdgeType(Enum):
    LINE=1
    CIRCLE=2
    BEZIER2=3  # bezier curve 2nd degree
    BEZIER3=4  # bezier curve 3nd degree


@dataclass
class EdgePart:
    type: EdgeType
    edge_indices: List[int]
    plane: Plane
    form: Line3D # | Circle3D | Bezier2ndDegree3D  # perhaps better to use planar variants?






