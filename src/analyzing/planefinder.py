import numpy as np
import trimesh


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

    def find_planes(self):
        distances = self._create_place_distances_from_source()
        np.hstack()

        eps = 1E-3
        round_normals = np.round(self._mesh.face_normals / eps) * eps
        _, unique_indices = np.unique(round_normals, axis=0, return_inverse=True)

        groups = []
        for i in np.unique(unique_indices):
            group = np.where(unique_indices == i)[0]
            if len(group) >= 2:
                groups.append(group)
        pass

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

    def _create_place_distances_from_source(self) -> np.array:
        normals = self._mesh.face_normals
        triangle_points = self._mesh.triangles[:, 0, :]  # take one point of every triangle
        ein_sum = np.einsum('ij,ij->i', normals, triangle_points)
        distances = np.abs(ein_sum) / np.linalg.norm(normals, axis=1)

        return distances


