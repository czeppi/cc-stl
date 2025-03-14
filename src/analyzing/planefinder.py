from typing import Iterator

import numpy as np
import trimesh

from analyzing.globalanalyzeresult import SurfacePatch, SurfaceKind
from geo3d import Plane, Vector3D


class PlaneFinder:
    """ find Planes in a Trimesh

    Round values, to find triangles, which are not exact equal.
    !! Works not exact at boundaries:
       p.e.  eps = 0.01
             value1 = 0.005000  -> 0.01
             value2 = 0.004999  -> 0.00
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh

    def find_planes(self) -> Iterator[SurfacePatch]:
        normals = self._mesh.face_normals
        distances = self._create_plane_distances_from_source()
        planes_data = np.hstack((normals, distances.reshape(-1, 1)))

        eps = 1E-3
        round_data = np.round(planes_data / eps) * eps
        unique_values, reverse_indices, unique_counts = np.unique(round_data, axis=0,
                                                                  return_inverse=True, return_counts=True)
        for unique_index in np.where(unique_counts >= 2)[0]:
            unique_value = unique_values[unique_index]
            x, y, z, plane_dist = [float(v) for v in unique_value]
            plane = Plane(normal=Vector3D(x, y, z), distance=plane_dist)
            triangle_indices = {int(tri_index) for tri_index in (np.where(reverse_indices == unique_index)[0])}
            yield SurfacePatch(type=SurfaceKind.PLANE,
                               triangle_indices=triangle_indices,
                               form=plane)

        # for i in np.unique(reverse_indices):
        #     group = np.where(reverse_indices == i)[0]
        #     if len(group) >= 2:
        #         groups.append(group)
        #
        #         yield SurfacePatch(type=SurfaceKind.PLANE,
        #                            triangle_indices=set(group),
        #                            form=Plane())

    # def _create_place_distances_from_source(self) -> np.array:
    #     triangles = self._mesh.vertices[self._mesh.faces]
    #     edges1 = triangles[:, 1] - triangles[:, 0]
    #     edges2 = triangles[:, 2] - triangles[:, 0]
    #     normals = np.cross(edges1, edges2)
    #     norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    #     unit_normals = normals / norm_lengths
    #
    #     d = -np.einsum('ij,ij->i', unit_normals, triangles[:, 0])
    #     distances = np.abs(d) / norm_lengths.flatten()
    #     return distances

    def _create_plane_distances_from_source(self) -> np.array:
        normals = self._mesh.face_normals
        triangle_points = self._mesh.triangles[:, 0, :]  # take one point of every triangle
        ein_sum = np.einsum('ij,ij->i', normals, triangle_points)
        distances = np.abs(ein_sum) / np.linalg.norm(normals, axis=1)
        return distances
