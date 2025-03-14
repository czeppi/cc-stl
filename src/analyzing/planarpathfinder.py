import math
from dataclasses import dataclass
from typing import Tuple, Iterator, Set, Optional

from sympy import Point3D

from analyzing.localanalyzeresult import PlanarPath
from analyzing.stlmesh import StlMesh, StlEdge, StlVertex, StlFace
from geo3d import Plane, calc_plane_from_3_points, Vector3D, calc_angle_from_3_points

_PLANE_DIST_EPS = 0.001
_WITH_PRINT = False

@dataclass
class EdgeVertexPair:
    edge: StlEdge
    vertex: StlVertex


class PlanarPathFinder:

    def __init__(self, mesh: StlMesh):
        self._mesh = mesh

    def find_path(self, start_edge: StlEdge) -> Iterator[PlanarPath]:
        if not self._is_edge_valid(start_edge):
            return

        found_edge_indices: Set[int] = set()  # to avoid double paths, if start_edge is in midst of path
        for left_edge, vertex0, right_edge in self._iter_all_edge_pairs(start_edge):
            if left_edge.index in found_edge_indices or right_edge.index in found_edge_indices:
                continue

            planar_path = self._calc_path_from_two_edges(left_edge, vertex0, right_edge)
            if planar_path and len(planar_path.edges) >= 3:
                yield planar_path

                found_edge_indices |= set(edge.index for edge in planar_path.edges
                                          if edge.index != start_edge.index)

    @staticmethod
    def _is_edge_valid(edge: StlEdge) -> bool:
        return not _EdgeChecker(edge).is_inside_facet()

    @staticmethod
    def _is_angle_ok(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> bool:
        """ to edges in a path must not connect in a sharp angle """
        phi_degree = calc_angle_from_3_points(p1, p2, p3)
        return phi_degree >= 120.0

    def _iter_all_edge_pairs(self, start_edge: StlEdge) -> Iterator[Tuple[StlEdge, StlVertex, StlEdge]]:
        vertex1 = start_edge.vertex1
        vertex2 = start_edge.vertex2

        for edge1 in vertex1.iter_edges():
            if edge1.index != start_edge.index and self._is_edge_valid(edge1):
                yield edge1, vertex1, start_edge

        for edge2 in vertex2.iter_edges():
            if edge2.index != start_edge.index and self._is_edge_valid(edge2):
                yield start_edge, vertex2, edge2

    def _calc_path_from_two_edges(self, left_edge: StlEdge, vertex0: StlVertex, right_edge: StlEdge
                                  ) -> Optional[PlanarPath]:
        left_vertex = left_edge.other_vertex(vertex0)
        right_vertex = right_edge.other_vertex(vertex0)

        if not self._is_angle_ok(left_vertex.pos, vertex0.pos, right_vertex.pos):
            return None

        plane = calc_plane_from_3_points(left_vertex.pos, vertex0.pos, right_vertex.pos)

        left_edge_vertex = EdgeVertexPair(edge=left_edge, vertex=left_vertex)
        left_edge_vertex_list = list(self._iter_extend_path(plane, left_edge_vertex))
        left_edge_list = [edge_vertex.edge for edge_vertex in left_edge_vertex_list]
        left_vertex_list = [edge_vertex.vertex for edge_vertex in left_edge_vertex_list]

        right_edge_vertex = EdgeVertexPair(edge=right_edge, vertex=right_vertex)
        right_edge_vertex_list = list(self._iter_extend_path(plane, right_edge_vertex))
        right_edge_list = [edge_vertex.edge for edge_vertex in right_edge_vertex_list]
        right_vertex_list = [edge_vertex.vertex for edge_vertex in right_edge_vertex_list]

        return PlanarPath(plane=plane,
                          vertices=list(reversed(left_vertex_list)) + [vertex0] + right_vertex_list,
                          edges=list(reversed(left_edge_list)) + right_edge_list)

    def _calc_plane(self, p1: Point3D, p2: Point3D, p3: Point3D) -> Plane:
        raise NotImplementedError()

    def _iter_extend_path(self, plane: Plane, edge_vertex: EdgeVertexPair) -> Iterator[EdgeVertexPair]:
        found_edge_indices: Set[int] = set()
        if _WITH_PRINT or True:
            print(f'iter_extend_path(edge={edge_vertex.edge.index}, vertex={edge_vertex.vertex.index})')
        while True:
            # print(f'E{edge_vertex.edge.index}, V{edge_vertex.vertex.index}')
            if edge_vertex.edge.index in found_edge_indices:
                break  # stopp at cycles

            yield edge_vertex
            found_edge_indices.add(edge_vertex.edge.index)

            next_edge_vertex_list = list(self._iter_next_edges_in_plane(plane, edge_vertex))
            n = len(next_edge_vertex_list)
            if _WITH_PRINT or n >= 2:
                print(f'  n={n}, edge={edge_vertex.edge.index}, vertex={edge_vertex.vertex.index}')
                for ev in next_edge_vertex_list:
                    print(f'    edge={ev.edge.index}, vertex={ev.vertex.index}')
            assert n <= 1
            if n == 1:
                edge_vertex = next_edge_vertex_list[0]
            else:
                break

    def _iter_next_edges_in_plane(self, plane: Plane, edge_vertex: EdgeVertexPair) -> Iterator[EdgeVertexPair]:
        cur_edge = edge_vertex.edge
        cur_vertex = edge_vertex.vertex
        prev_vertex = cur_edge.other_vertex(cur_vertex)
        for next_edge in cur_vertex.iter_edges():
            if next_edge.index != cur_edge.index and self._is_edge_valid(next_edge):
                next_vertex = next_edge.other_vertex(cur_vertex)
                if self._is_angle_ok(prev_vertex.pos, cur_vertex.pos, next_vertex.pos):
                    dist = plane.calc_distance_to_point(next_vertex.pos)
                    if dist < _PLANE_DIST_EPS:
                        yield EdgeVertexPair(edge=next_edge, vertex=next_vertex)


class _EdgeChecker:
    """ check, if edge is a path candidate

        edge should not inside a facet <-> the two adjoining faces should not be planar
    """

    def __init__(self, edge: StlEdge):
        self._edge = edge

    def is_inside_facet(self) -> bool:
        num_faces = len(self._edge.faces)
        assert num_faces <= 2

        if num_faces < 2:
            return False

        face1, face2 = self._edge.faces
        vertex3 = self._find_third_vertex_in_face(face1)
        vertex4 = self._find_third_vertex_in_face(face2)

        plane = calc_plane_from_3_points(self._edge.vertex1.pos, self._edge.vertex2.pos, vertex3.pos)
        dist = plane.calc_distance_to_point(vertex4.pos)
        return dist < _PLANE_DIST_EPS

    def _find_third_vertex_in_face(self, face: StlFace) -> StlVertex:
        edge_vertex_indices = {vertex.index for vertex in self._edge.vertices}
        poss_vertices = [vertex for vertex in face.vertices
                         if vertex.index not in edge_vertex_indices]
        assert len(poss_vertices) == 1
        return poss_vertices[0]
