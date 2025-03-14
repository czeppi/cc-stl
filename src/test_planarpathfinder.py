import math

import trimesh

from analyzing.localanalyzer import LocalAnalyzer
from analyzing.planarpathfinder import PlanarPathFinder, _EdgeChecker
from analyzing.stlmesh import StlMeshCreator
from geo3d import calc_angle_from_3_points
from itemdetectoratmousepos import MeshItemKey, MeshItemType

mesh = trimesh.load("../stl-files/adapter_v2_bottom_pmw_3389.stl")

rotation_matrix = trimesh.transformations.rotation_matrix(math.pi / 2, direction=[1, 0, 0])
mesh.apply_transform(rotation_matrix)

stl_mesh = StlMeshCreator(mesh).create()

edge_6496 = stl_mesh.get_edge(6496)
edge_6492 = stl_mesh.get_edge(6492)

normal_6496a = mesh.face_normals[edge_6496.faces[0].index]
normal_6496b = mesh.face_normals[edge_6496.faces[1].index]
normal_6492a = mesh.face_normals[edge_6492.faces[0].index]
normal_6492b = mesh.face_normals[edge_6492.faces[1].index]

phi = calc_angle_from_3_points(stl_mesh.get_vertex(2507).pos, stl_mesh.get_vertex(2505).pos, stl_mesh.get_vertex(2503).pos)

edge_6517 = stl_mesh.get_edge(6517)
is_edge_6517_inside = _EdgeChecker(edge_6517).is_inside_facet()

planar_paths = list(PlanarPathFinder(stl_mesh).find_path(edge_6496))
pass

