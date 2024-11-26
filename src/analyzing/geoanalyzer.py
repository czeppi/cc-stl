import trimesh

from analyzing.analyzeresult import AnalyzeResultData
from analyzing.planefinder import PlaneFinder


class GeoAnalyzer:

    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh

    def analyze(self) -> AnalyzeResultData:
        print('find planes ...')
        plane_finder = PlaneFinder(self._mesh)
        planes = list(plane_finder.find_planes())
        print('analyze ready.')
        return AnalyzeResultData(surface_patches=planes, edge_segments=[])
