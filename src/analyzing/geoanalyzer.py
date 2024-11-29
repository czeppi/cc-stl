import trimesh

from analyzing.analyzeresult import AnalyzeResultData
from analyzing.planefinder import PlaneFinder
from analyzing.spherefinder import SphereFinder
from analyzing.stlmesh import StlMeshCreator


class GeoAnalyzer:

    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh

    def analyze(self) -> AnalyzeResultData:
        print('find planes ...')
        plane_finder = PlaneFinder(self._mesh)
        planes = list(plane_finder.find_planes())
        print('analyze ready.')

        stl_mesh_creator = StlMeshCreator(self._mesh)
        stl_mesh = stl_mesh_creator.create()

        sphere_finder = SphereFinder(stl_mesh)
        edge_sphere_map = dict(sphere_finder.iter_edge_spheres())
        spheres = list(sphere_finder.iter_surface_patches(edge_sphere_map))

        return AnalyzeResultData(surface_patches=planes + spheres, edge_segments=[],
                                 stl_mesh=stl_mesh, edge_sphere_map=edge_sphere_map)
