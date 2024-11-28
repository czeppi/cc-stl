from __future__ import annotations
from typing import Dict, List, Tuple, Iterator

import trimesh

from geo3d import Vector3D


class StlMesh:

    def __init__(self):
        self._vertices: Dict[int, StlVertex] = {}
        self._edges: Dict[int, StlEdge] = {}
        self._faces: Dict[int, StlFace] = {}

        self._vertex2edges_map: Dict[Tuple[int, int], StlEdge] = {}

    def add_vertex(self, vertex_index: int, pos: Vector3D):
        assert len(self._edges) == 0  # edges must add after vertices
        assert len(self._faces) == 0  # faces must add after vertices
        assert vertex_index not in self._vertices
        self._vertices[vertex_index] = StlVertex(vertex_index, pos=pos)

    def add_edge(self, edge_index: int, vertex1_index: int, vertex2_index: int):
        assert len(self._faces) == 0  # faces must add after edges
        assert edge_index not in self._edges
        assert vertex1_index != vertex2_index
        vertex1_index, vertex2_index = self._sort_vertex_indices(vertex1_index, vertex2_index)
        vertex1 = self._vertices[vertex1_index]
        vertex2 = self._vertices[vertex2_index]

        new_edge = StlEdge(edge_index, vertex1=vertex1, vertex2=vertex2)
        self._edges[edge_index] = new_edge
        self._vertex2edges_map[(vertex1_index, vertex2_index)] = new_edge
        vertex1.add_edge(new_edge)
        vertex2.add_edge(new_edge)

    def add_face(self, face_index: int, vertex1_index: int, vertex2_index: int, vertex3_index: int):
        assert face_index not in self._faces
        assert vertex1_index != vertex2_index and vertex2_index != vertex3_index and vertex3_index != vertex1_index
        vertex1 = self._vertices[vertex1_index]
        vertex2 = self._vertices[vertex2_index]
        vertex3 = self._vertices[vertex3_index]

        edge12_key = self._sort_vertex_indices(vertex1_index, vertex2_index)
        edge23_key = self._sort_vertex_indices(vertex2_index, vertex3_index)
        edge31_key = self._sort_vertex_indices(vertex3_index, vertex1_index)

        edge12 = self._vertex2edges_map[edge12_key]
        edge23 = self._vertex2edges_map[edge23_key]
        edge31 = self._vertex2edges_map[edge31_key]

        new_face = StlFace(face_index, vertex1=vertex1, vertex2=vertex2, vertex3=vertex3,
                           edge12=edge12, edge23=edge23, edge31=edge31)
        self._faces[face_index] = new_face
        vertex1.add_face(new_face)
        vertex2.add_face(new_face)
        vertex3.add_face(new_face)
        edge12.add_face(new_face)
        edge23.add_face(new_face)
        edge31.add_face(new_face)

    @staticmethod
    def _sort_vertex_indices(vertex1_index: int, vertex2_index) -> Tuple[int, int]:
        if vertex1_index < vertex2_index:
            return vertex1_index, vertex2_index
        else:
            return vertex2_index, vertex1_index

    def iter_vertices(self) -> Iterator[StlVertex]:
        yield from self._vertices.values()

    def iter_edges(self) -> Iterator[StlEdge]:
        yield from self._edges.values()

    def iter_faces(self) -> Iterator[StlFace]:
        yield from self._faces.values()

    def get_vertex(self, vertex_index: int) -> StlVertex:
        return self._vertices[vertex_index]

    def get_edge(self, edge_index: int) -> StlEdge:
        return self._edges[edge_index]

    def get_face(self, face_index: int) -> StlFace:
        return self._faces[face_index]


class StlVertex:

    def __init__(self, index: int, pos: Vector3D):
        assert index >= 0
        self._index = index
        self._pos = pos
        self._edges: List[StlEdge] = []
        self._faces: List[StlFace] = []

    @property
    def index(self) -> int:
        return self._index

    @property
    def pos(self) -> Vector3D:
        return self._pos

    def add_edge(self, edge: StlEdge) -> None:
        self._edges.append(edge)

    def add_face(self, face: StlFace) -> None:
        self._faces.append(face)


class StlEdge:

    def __init__(self, index: int, vertex1: StlVertex, vertex2: StlVertex):
        assert index >= 0
        self._index = index
        assert vertex1.index < vertex2.index
        self._vertex1 = vertex1
        self._vertex2 = vertex2
        self._faces: List[StlFace] = []

    @property
    def index(self) -> int:
        return self._index

    @property
    def vertex1(self) -> StlVertex:
        return self._vertex1

    @property
    def vertex2(self) -> StlVertex:
        return self._vertex2

    @property
    def faces(self) -> List[StlFace]:
        return self._faces

    def add_face(self, face: StlFace) -> None:
        self._faces.append(face)


class StlFace:

    def __init__(self, index: int, vertex1: StlVertex, vertex2: StlVertex, vertex3: StlVertex,
                 edge12: StlEdge, edge23: StlEdge, edge31: StlEdge):
        assert index >= 0
        self._index = index
        self._vertices: List[StlVertex] = [vertex1, vertex2, vertex3]
        self._edges: List[StlEdge] = [edge12, edge23, edge31]

    @property
    def index(self) -> int:
        return self._index

    @property
    def vertices(self) -> List[StlVertex]:
        return self._vertices

    @property
    def edges(self) -> List[StlEdge]:
        return self._edges


class StlMeshCreator:

    def __init__(self, mesh: trimesh.Trimesh):
        self._tri_mesh = mesh
        self._stl_mesh = StlMesh()

    def create(self) -> StlMesh:
        self._stl_mesh = StlMesh()

        print(f'StlMeshCreator: create vertices...')
        for i, xyz in enumerate(self._tri_mesh.vertices):
            self._stl_mesh.add_vertex(i, Vector3D(*xyz))

        print(f'StlMeshCreator: create edges...')
        for i, (v1, v2) in enumerate(self._tri_mesh.edges_unique):
            self._stl_mesh.add_edge(i, v1, v2)

        print(f'StlMeshCreator: create faces...')
        for i, (v1, v2, v3) in enumerate(self._tri_mesh.faces):
            self._stl_mesh.add_face(i, v1, v2, v3)

        print(f'StlMeshCreator: ready')
        return self._stl_mesh
