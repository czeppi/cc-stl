from typing import Iterator, Tuple, List, Optional, Dict

import numpy as np

from analyzing.analyzeresult import SurfacePatch, SurfaceKind
from analyzing.quasiequalrowfinder import QuasiEqualRowFinder
from analyzing.stlmesh import StlMesh, StlEdge, StlVertex
from geo3d import Sphere, Vector3D, calc_sphere_from_4_points


class SphereFinder:

    def __init__(self, mesh: StlMesh):
        self._mesh = mesh

    def find_spheres(self) -> Iterator[SurfacePatch]:
        print('SphereFinder: create edge->sphere map...')
        edge_sphere_map = dict(self._iter_edge_spheres())

        print('SphereFinder: group spheres...')
        yield from self._iter_sphere_groups(edge_sphere_map)

        print('SphereFinder: ready.')

    def _iter_edge_spheres(self) -> Iterator[Tuple[int, Sphere]]:
        for edge in self._mesh.iter_edges():
            if len(edge.faces) == 2:
                vertices = {v.index: v for face in edge.faces for v in face.vertices}
                assert len(vertices) == 4
                p1, p2, p3, p4 = [vertex.pos for vertex in vertices.values()]
                sphere = calc_sphere_from_4_points(p1, p2, p3, p4)
                if sphere:
                    yield edge.index, sphere

    def _iter_sphere_groups(self, edge_sphere_map: Dict[int, Sphere]) -> Iterator[SurfacePatch]:
        edge_sphere_array = np.array(
            list([*sphere.center, sphere.radius, edge_index]
                 for edge_index, sphere in edge_sphere_map.items())
        )

        eps = 0.001
        max_radius = 100.0
        group_finder = QuasiEqualRowFinder(edge_sphere_array, eps_list=[eps, eps, eps, eps])
        for edge_indices in group_finder.iter_groups():
            if len(edge_indices) >= 2:
                face_indices = {face.index
                                for e in edge_indices
                                for face in self._mesh.get_edge(e).faces}
                sphere = edge_sphere_map[edge_indices[0]]  # todo: use average values instead of any ?!
                if sphere.radius <= max_radius:
                    yield SurfacePatch(type=SurfaceKind.SPHERE,
                                       triangle_indices=face_indices,
                                       form=sphere)
