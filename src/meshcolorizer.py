from typing import List, Optional, Iterator, Tuple

import numpy as np
import trimesh

from analyzing.analyzeresult import AnalyzeResult
from itemdetectoratmousepos import MeshItemKey, MeshItemType
from shaders import ColorTuple

DEFAULT_FACET_COLOR = 0.4, 0.4, 0.8
DEFAULT_EDGE_COLOR = 0.0, 0.0, 0.0
CUR_ITEM_COLOR = 0.0, 1.0, 0.0


class MeshColorizer:

    def __init__(self, mesh: trimesh.Trimesh, analyze_result: Optional[AnalyzeResult] = None):
        self._mesh = mesh
        self._analyze_result = analyze_result
        self._cur_item: Optional[MeshItemKey] = None
        self._sel_items: List[MeshItemKey] = []

    def set_cur_item(self, cur_item: Optional[MeshItemKey]) -> None:
        self._cur_item = cur_item

    def set_sel_items(self, sel_items: List[MeshItemKey]) -> None:
        self._sel_items = sel_items

    def iter_face_colors(self) -> Iterator[Tuple[ColorTuple, np.array]]:
        cur_item = self._cur_item
        num_faces = len(self._mesh.faces)
        face_index_array = np.arange(num_faces)

        if cur_item and cur_item.type == MeshItemType.FACE:
            all_not_cur_faces = np.delete(face_index_array, np.where(face_index_array == cur_item.index))
            yield DEFAULT_FACET_COLOR, all_not_cur_faces
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
        else:
            yield DEFAULT_FACET_COLOR, face_index_array

    def iter_edge_colors(self) -> Iterator[Tuple[ColorTuple, np.array]]:
        cur_item = self._cur_item
        num_edges = len(self._mesh.edges_unique)
        edge_index_array = np.arange(num_edges)

        if cur_item and cur_item.type == MeshItemType.EDGE:
            all_not_cur_faces = np.delete(edge_index_array, np.where(edge_index_array == cur_item.index))
            yield DEFAULT_EDGE_COLOR, all_not_cur_faces
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
        else:
            yield DEFAULT_EDGE_COLOR, edge_index_array

    def iter_vertex_colors(self) -> Iterator[Tuple[ColorTuple, np.array]]:
        cur_item = self._cur_item

        if cur_item and cur_item.type == MeshItemType.VERTEX:
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
