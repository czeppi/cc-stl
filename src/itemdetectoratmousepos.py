from dataclasses import dataclass
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Iterator

import numpy as np
import trimesh
from PySide6.QtCore import QPoint
from PySide6.QtGui import QMatrix4x4

from projgeo import ProjTriangle, ProjVertex, ProjEdge

MOUSE_VERTEX_DIST = 5.0  # max distance from mouse, to select vertex
MOUSE_EDGE_DIST = 5.0  # max distance from mouse, to select edge


class MeshItemType(Enum):
    VERTEX = 1
    EDGE = 2
    FACE = 3


@dataclass
class MeshItemKey:
    type: MeshItemType
    index: int


@dataclass
class RectArray:  # array of rectangles
    x_min: np.array
    y_min: np.array
    x_max: np.array
    y_max: np.array


class ItemDetectorAtMousePos:

    def __init__(self, mesh: trimesh.Trimesh, mvp_matrix: QMatrix4x4, view_size: Tuple[int, int]):
        self._mesh = mesh
        self._mvp_matrix = mvp_matrix
        self._view_size = view_size

        self._projected_vertices_array = self._project_all_vertices()
        self._triangle_boxes = self._create_triangle_boxes()
        self._edge_boxes = self._create_edge_boxes()

    def _project_all_vertices(self) -> np.array:
        vertices = self._mesh.vertices
        mvp = self._mvp_matrix

        vertices_matrix = np.vstack([np.transpose(vertices), np.full(len(vertices), 1.0)])

        m = np.array([[mvp[0, 0], mvp[0, 1], mvp[0, 2], mvp[0, 3]],
                      [mvp[1, 0], mvp[1, 1], mvp[1, 2], mvp[1, 3]],
                      [mvp[2, 0], mvp[2, 1], mvp[2, 2], mvp[2, 3]],
                      [mvp[3, 0], mvp[3, 1], mvp[3, 2], mvp[3, 3]]])

        x1_vec, y1_vec, z1_vec, w1_vec = m @ vertices_matrix
        width2 = self._view_size[0] / 2.0
        height2 = self._view_size[1] / 2.0
        x2_vec = width2 + x1_vec / w1_vec * width2
        y2_vec = height2 - y1_vec / w1_vec * height2
        z2_vec = z1_vec / w1_vec

        return np.transpose(np.array([x2_vec, y2_vec, z2_vec]))

    def _create_triangle_boxes(self) -> RectArray:
        proj_vertices = self._projected_vertices_array
        tri_vertices = proj_vertices[self._mesh.faces]

        return RectArray(
            x_min=tri_vertices[:, :, 0].min(axis=1),
            y_min=tri_vertices[:, :, 1].min(axis=1),
            x_max=tri_vertices[:, :, 0].max(axis=1),
            y_max=tri_vertices[:, :, 1].max(axis=1),
        )

    def _create_edge_boxes(self) -> RectArray:
        proj_vertices = self._projected_vertices_array
        edge_vertices = proj_vertices[self._mesh.edges_unique]

        return RectArray(
            x_min=edge_vertices[:, :, 0].min(axis=1),
            y_min=edge_vertices[:, :, 1].min(axis=1),
            x_max=edge_vertices[:, :, 0].max(axis=1),
            y_max=edge_vertices[:, :, 1].max(axis=1),
        )

    def find_cur_item(self, mouse_pos: QPoint) -> Optional[MeshItemKey]:
        nearest_triangle = self._find_nearest_triangle(mouse_pos)

        best_vertex = self._find_best_vertex(mouse_pos=mouse_pos, nearest_triangle=nearest_triangle)
        if best_vertex:
            return MeshItemKey(type=MeshItemType.VERTEX, index=best_vertex.index)

        best_edge = self._find_best_edge(mouse_pos=mouse_pos, nearest_triangle=nearest_triangle)
        if best_edge:
            return MeshItemKey(type=MeshItemType.EDGE, index=best_edge.index)

        if nearest_triangle:
            return MeshItemKey(type=MeshItemType.FACE, index=nearest_triangle.index)

    def _find_nearest_triangle(self, mouse_pos: QPoint) -> Optional[ProjTriangle]:
        poss_triangles = list(self._iter_poss_triangles(mouse_pos))
        if len(poss_triangles):
            sorted_triangles = sorted(poss_triangles,
                                      key=lambda triangle: triangle.calc_z_at_xy(x=mouse_pos.x(), y=mouse_pos.y()))
            return sorted_triangles[0]

    def _iter_poss_triangles(self, mouse_pos: QPoint) -> Iterator[ProjTriangle]:
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        for triangle_index in self._find_box_indices(mouse_pos, boxes=self._triangle_boxes, distance=0.0):
            face = self._mesh.faces[triangle_index]
            proj_triangle = ProjTriangle(index=triangle_index,
                                         p1=self._create_proj_vertex_from_numpy_vertex(face[0]),
                                         p2=self._create_proj_vertex_from_numpy_vertex(face[1]),
                                         p3=self._create_proj_vertex_from_numpy_vertex(face[2]))
            if proj_triangle.contains_point(mouse_x, mouse_y):
                yield proj_triangle

    @staticmethod
    def _find_box_indices(mouse_pos: QPoint, boxes: RectArray, distance: float) -> np.array:
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        d = distance

        x_min = mouse_x - d
        x_max = mouse_x + d
        y_min = mouse_y - d
        y_max = mouse_y + d

        overlap_cond = ((boxes.x_max >= x_min) & (boxes.x_min <= x_max) &
                        (boxes.y_max >= y_min) & (boxes.y_min <= y_max))

        touching_triangles = np.where(overlap_cond)[0]
        return touching_triangles

    def _find_best_vertex(self, mouse_pos: QPoint, nearest_triangle: Optional[ProjTriangle]) -> Optional[ProjVertex]:
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()

        poss_vertices = list(self._iter_poss_vertices(mouse_pos=mouse_pos, nearest_triangle=nearest_triangle))
        if len(poss_vertices):
            return min(poss_vertices, key=lambda vertex: vertex.calc_dist_to_point(mouse_x, mouse_y))

    def _iter_poss_vertices(self, mouse_pos: QPoint, nearest_triangle: Optional[ProjTriangle]) -> Iterator[ProjVertex]:
        for vertex_index in self._find_vertex_indices_at_mouse(mouse_pos):
            proj_vertex = self._create_proj_vertex_from_numpy_vertex(vertex_index)
            if nearest_triangle is None or not nearest_triangle.cover_vertex_at_xy(proj_vertex, x=mouse_pos.x(), y=mouse_pos.y()):
                yield proj_vertex

    def _find_vertex_indices_at_mouse(self, mouse_pos: QPoint) -> List[int]:
        proj_vertices = self._projected_vertices_array
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        d = MOUSE_VERTEX_DIST  # max. distance between mouse and vertex on view
        x_min = mouse_x - d
        x_max = mouse_x + d
        y_min = mouse_y - d
        y_max = mouse_y + d

        vertex_indices = np.where((proj_vertices[:, 0] >= x_min) & (proj_vertices[:, 0] <= x_max) &
                                  (proj_vertices[:, 1] >= y_min) & (proj_vertices[:, 1] <= y_max))
        return vertex_indices[0].tolist()

    def _find_best_edge(self, mouse_pos: QPoint, nearest_triangle: Optional[ProjTriangle]) -> Optional[ProjEdge]:
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()

        poss_edges = list(self._iter_poss_edges(mouse_pos=mouse_pos, nearest_triangle=nearest_triangle))
        if len(poss_edges):
            return min(poss_edges, key=lambda edge: edge.calc_dist_to_point(mouse_x, mouse_y))

    def _iter_poss_edges(self, mouse_pos: QPoint, nearest_triangle: Optional[ProjTriangle]) -> Iterator[ProjEdge]:
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()

        for edge_index in self._find_box_indices(mouse_pos, boxes=self._edge_boxes, distance=MOUSE_EDGE_DIST):
            vertex1_index, vertex2_index = self._mesh.edges_unique[edge_index]

            proj_vertex1 = self._create_proj_vertex_from_numpy_vertex(vertex1_index)
            proj_vertex2 = self._create_proj_vertex_from_numpy_vertex(vertex2_index)
            proj_edge = ProjEdge(index=edge_index, p1=proj_vertex1, p2=proj_vertex2)

            mouse_dist = proj_edge.calc_dist_to_point(mouse_x, mouse_y)
            if mouse_dist <= MOUSE_EDGE_DIST:
                if nearest_triangle is None or not nearest_triangle.cover_edge_at_xy(proj_edge, x=mouse_x, y=mouse_y):
                    yield proj_edge

    def _create_proj_vertex_from_numpy_vertex(self, index: int) -> ProjVertex:
        x, y, z = self._projected_vertices_array[index]
        return ProjVertex(index=index, x=x, y=y, z=z)

