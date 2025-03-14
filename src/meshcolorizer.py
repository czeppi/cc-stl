from typing import List, Optional, Iterator, Tuple, Set

import numpy as np
import trimesh

from analyzing.globalanalyzeresult import GlobalAnalyzeResult
from analyzing.localanalyzeresult import LocalAnalyzeResultData
from itemdetectoratmousepos import MeshItemKey, MeshItemType
from shaders import ColorTuple

DEFAULT_FACET_COLOR = 0.4, 0.4, 0.8
DEFAULT_EDGE_COLOR = 0.0, 0.0, 0.0
CUR_ITEM_COLOR = 0.0, 1.0, 0.0
MARK_ITEM_COLOR = 0.5, 1.0, 0.0


class MeshColorizer:

    def __init__(self, mesh: trimesh.Trimesh,
                 global_analyze_result: Optional[GlobalAnalyzeResult] = None,
                 local_analyze_result: Optional[LocalAnalyzeResultData] = None):
        self._mesh = mesh
        self._global_analyze_result = global_analyze_result
        self._local_analyze_result = local_analyze_result
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
            marked_face_indices = self._get_marked_face_indices(cur_item.index)
            not_default_indices = marked_face_indices | {cur_item.index}
            not_default_index_array = np.array(list(not_default_indices))
            default_index_array = face_index_array[~np.isin(face_index_array, not_default_index_array)]
            yield DEFAULT_FACET_COLOR, default_index_array
            if len(marked_face_indices) > 0:
                yield MARK_ITEM_COLOR, np.array(list(marked_face_indices))
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
        else:
            yield DEFAULT_FACET_COLOR, face_index_array

    def _get_marked_face_indices(self, face_index: int) -> Set[int]:
        if self._global_analyze_result:
            surface_patch = self._global_analyze_result.find_surface_patch(face_index)
            if surface_patch:
                return surface_patch.triangle_indices - {face_index}

        return set()

    def iter_edge_colors(self) -> Iterator[Tuple[ColorTuple, np.array]]:
        cur_item = self._cur_item
        num_edges = len(self._mesh.edges_unique)
        edge_index_array = np.arange(num_edges)

        if cur_item and cur_item.type == MeshItemType.EDGE:
            marked_edge_indices = self._get_marked_edge_indices(cur_item.index)
            not_default_indices = marked_edge_indices | {cur_item.index}
            not_default_index_array = np.array(list(not_default_indices))
            default_index_array = edge_index_array[~np.isin(edge_index_array, not_default_index_array)]
            yield DEFAULT_EDGE_COLOR, default_index_array
            if len(marked_edge_indices) > 0:
                yield MARK_ITEM_COLOR, np.array(list(marked_edge_indices))
            all_not_cur_faces = np.delete(edge_index_array, np.where(edge_index_array == cur_item.index))
            yield DEFAULT_EDGE_COLOR, all_not_cur_faces
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
        else:
            yield DEFAULT_EDGE_COLOR, edge_index_array

    def _get_marked_edge_indices(self, edge_index: int) -> Set[int]:
        if self._local_analyze_result:
            planar_path = self._local_analyze_result.planar_path
            if planar_path:
                return {edge.index for edge in planar_path.edges} - {edge_index}

        return set()

    def iter_vertex_colors(self) -> Iterator[Tuple[ColorTuple, np.array]]:
        cur_item = self._cur_item

        if cur_item and cur_item.type == MeshItemType.VERTEX:
            yield CUR_ITEM_COLOR, np.array([cur_item.index])
