from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import trimesh
from PySide6.QtCore import QPoint
from PySide6.QtGui import QMatrix4x4


SELECTED_VERTEX_INDICES = [20000]


class MeshItemType(Enum):
    VERTEX = 1
    EDGE = 2
    FACE = 3


@dataclass
class MeshItemKey:
    type: MeshItemType
    index: int



class ItemDetectorAtMousePos:

    def __init__(self, mesh: trimesh.Trimesh, mvp_matrix: QMatrix4x4, view_size: Tuple[int, int]):
        self._mesh = mesh
        self._mvp_matrix = mvp_matrix
        self._view_size = view_size

        self._projected_vertices = self._project_all_vertices()

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
        x2_vec = width2 + x1_vec / z1_vec * width2
        y2_vec = height2 - y1_vec / z1_vec * height2
        #z2_vec = z1_vec    # todo

        if len(SELECTED_VERTEX_INDICES) > 0:
            i = SELECTED_VERTEX_INDICES[0]  # take only one
            print(f'sel_transformed: {x2_vec[i]}, {y2_vec[i]}, {z1_vec[i]}')

        return np.array([x2_vec, y2_vec, z1_vec])

    def find_cur_item(self, mouse_pos: QPoint) -> Optional[MeshItemKey]:
        pass

    def _find_vertex_indices_at_mouse(self, mouse_pos: QPoint) -> np.array:
        proj_vertices = np.transpose(self._projected_vertices)
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        d = 10.0  # max. distance between mouse and vertex on view
        x_min = mouse_x - d
        x_max = mouse_x + d
        y_min = mouse_y - d
        y_max = mouse_y + d

        vertex_indices = np.where((proj_vertices[:, 0] >= x_min) & (proj_vertices[:, 0] <= x_max) &
                                  (proj_vertices[:, 1] >= y_min) & (proj_vertices[:, 1] <= y_max))
        #print(f'vertex_indices: {list(vertex_indices)}')
        return vertex_indices