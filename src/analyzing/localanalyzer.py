from typing import Union, Optional

from analyzing.localanalyzeresult import LocalAnalyzeResultData, PlanarPath
from analyzing.planarpathfinder import PlanarPathFinder
from analyzing.stlmesh import StlMesh, StlVertex, StlEdge, StlFace
from itemdetectoratmousepos import MeshItemKey, MeshItemType


class LocalAnalyzer:

    def __init__(self, stl_mesh: StlMesh):
        self._stl_mesh = stl_mesh

    def analyze_item(self, start_item_key: Optional[MeshItemKey]) -> LocalAnalyzeResultData:
        planar_path: Optional[PlanarPath] = None
        if start_item_key and start_item_key.type == MeshItemType.EDGE:
            stl_edge = self._stl_mesh.get_edge(start_item_key.index)
            planar_paths = list(PlanarPathFinder(self._stl_mesh).find_path(stl_edge))
            print(f'local analyze edge {start_item_key.index}, #paths={len(planar_paths)}, '
                  f'path_lengths={[len(p.edges) for p in planar_paths]}')
            if len(planar_paths) == 1:
                planar_path = planar_paths[0]

        return LocalAnalyzeResultData(planar_path=planar_path)
