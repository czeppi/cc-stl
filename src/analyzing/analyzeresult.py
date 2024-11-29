from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Dict, Optional

from trimesh.primitives import Sphere

from analyzing.stlmesh import StlMesh
from geo3d import Plane, EndlessCylinder, Line3D


@dataclass
class AnalyzeResultData:
    surface_patches: List[SurfacePatch]
    edge_segments: List[EdgeSegment]
    stl_mesh: StlMesh
    edge_sphere_map: Dict[int, Sphere]


class AnalyzeResult:

    def __init__(self, result_data: AnalyzeResultData):
        self._result_data = result_data
        self._triangle_surface_patch_map = self._create_triangle_surface_patch_map(result_data)

    @staticmethod
    def _create_triangle_surface_patch_map(analyze_result: AnalyzeResultData) -> Dict[int, SurfacePatch]:
        return {tri_index: patch
                for patch in analyze_result.surface_patches
                for tri_index in patch.triangle_indices}

    @property
    def data(self) -> AnalyzeResultData:
        return self._result_data

    def find_surface_patch(self, face_index: int) -> Optional[SurfacePatch]:
        return self._triangle_surface_patch_map.get(face_index, None)

    def count_planes(self) -> int:
        return sum(1 for patch in self._result_data.surface_patches if patch.type == SurfaceKind.PLANE)

    def count_spheres(self) -> int:
        return sum(1 for patch in self._result_data.surface_patches if patch.type == SurfaceKind.SPHERE)

    def count_cylinders(self) -> int:
        return sum(1 for patch in self._result_data.surface_patches if patch.type == SurfaceKind.CYLINDER)


class SurfaceKind(Enum):
    PLANE = 1
    SPHERE = 2
    CYLINDER = 3


@dataclass
class SurfacePatch:  # or PartialSurface
    type: SurfaceKind
    triangle_indices: Set[int]  # face indices in mesh
    form: Plane | Sphere | EndlessCylinder


class EdgeType(Enum):
    LINE = 1
    CIRCLE = 2
    BEZIER2 = 3  # bezier curve 2nd degree
    BEZIER3 = 4  # bezier curve 3nd degree


@dataclass
class EdgeSegment:
    type: EdgeType
    edge_indices: List[int]
    plane: Plane
    form: Line3D  # | Circle3D | Bezier2ndDegree3D  # perhaps better to use planar variants?
