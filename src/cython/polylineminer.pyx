from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport int32_t, int64_t
from cython.operator cimport dereference, postincrement
import numpy as np
cimport numpy as cnp
from trimesh import Trimesh
import trimesh as trimeshlib
import time
from dataclasses import dataclass


cdef struct VertexStruct:
    double[3] coords


cdef struct EdgeStruct:
    int64_t[2] vertices
    int64_t[2] faces


cdef struct FaceStruct:
    int64_t[3] vertices
    int64_t[3] edges


cdef class StlMesh:
    cdef:
        vector[VertexStruct] _vertices
        vector[EdgeStruct] _edges
        vector[FaceStruct] _faces
        vector[vector[int64_t]] _vertex_to_faces
        vector[vector[int64_t]] _vertex_to_edges
        unordered_map[int64_t, int64_t] _vertex_pair_to_edge

    def __cinit__(self, object trimesh):  # object = Trimesh
        self._init_mesh(trimesh)

    cdef void _init_mesh(self, object trimesh):  # object = Trimesh
        """Load the mesh from a Trimesh instance and build references."""
        
        cdef:
            size_t num_vertices = trimesh.vertices.shape[0]
            size_t num_edges = trimesh.edges_unique.shape[0]
            size_t num_faces = trimesh.faces.shape[0]
        
        self._vertices.resize(num_vertices)
        self._edges.resize(num_edges)
        self._faces.resize(num_faces)

        self._vertex_to_faces.resize(num_vertices)
        self._vertex_to_edges.resize(num_vertices)
        
        #print('init vertices...')
        self._init_vertices(trimesh)
        #print('init edges...')
        self._init_edges_and_vertex_to_edges(trimesh)
        #print('init faces...')
        self._init_faces_etc(trimesh)
        #print('ready')

        #self._check_edge_to_faces()

    cdef void _init_vertices(self, object trimesh):
        cdef:
            double[:, ::1] vertex_view = trimesh.vertices
            VertexStruct vertex
            size_t vertex_idx 

        for vertex_idx in range(self._vertices.size()):
            vertex.coords[0] = vertex_view[vertex_idx, 0]
            vertex.coords[1] = vertex_view[vertex_idx, 1]
            vertex.coords[2] = vertex_view[vertex_idx, 2]
            self._vertices[vertex_idx] = vertex            

    cdef void _init_edges_and_vertex_to_edges(self, object trimesh):
        cdef:
            const int64_t[:, ::1] edge_view = np.ascontiguousarray(trimesh.edges_unique, dtype=np.int64)
            EdgeStruct edge
            size_t edge_idx
            int64_t v1, v2

        for edge_idx in range(self._edges.size()):
            v1, v2 = edge_view[edge_idx]
            edge.vertices[0] = v1
            edge.vertices[1] = v2
            edge.faces[0] = -1
            edge.faces[1] = -1
            self._edges[edge_idx] = edge
            self._vertex_to_edges[v1].push_back(edge_idx)
            self._vertex_to_edges[v2].push_back(edge_idx)
            self._vertex_pair_to_edge[self._create_vertex_pair_index(v1, v2)] = edge_idx

    cdef void _init_faces_etc(self, object trimesh):
        cdef:
            const int64_t[:, ::1] face_view = np.ascontiguousarray(trimesh.faces, dtype=np.int64)
            FaceStruct face
            int64_t v1, v2, v3
            int64_t e12, e23, e31
            size_t face_idx

        #print(f'#faces: {self._faces.size()}, #vertices: {self._vertices.size()}, #edges: {self._edges.size()}')

        for face_idx in range(self._faces.size()):
            #print(f'face {face_idx}...')
            v1 = face_view[face_idx][0]
            v2 = face_view[face_idx][1]
            v3 = face_view[face_idx][2]
            #print(f'v1={v1}, v2={v2}, v3={v3}')
            e12 = self._get_edge_id_from_vertex_pair(v1, v2)
            e23 = self._get_edge_id_from_vertex_pair(v2, v3)
            e31 = self._get_edge_id_from_vertex_pair(v3, v1)
            #print(f'v1={e12}, v2={e23}, v3={e31}')

            face.vertices[0] = v1
            face.vertices[1] = v2
            face.vertices[2] = v3
            face.edges[0] = e12
            face.edges[1] = e23
            face.edges[2] = e31
            self._faces[face_idx] = face

            self._vertex_to_faces[v1].push_back(face_idx)
            self._vertex_to_faces[v2].push_back(face_idx)
            self._vertex_to_faces[v3].push_back(face_idx)

            self._add_edge_to_face(e12, face_idx)
            self._add_edge_to_face(e23, face_idx)
            self._add_edge_to_face(e31, face_idx)

    cdef void _add_edge_to_face(self, int64_t edge_idx, int64_t face_idx):
        if self._edges[edge_idx].faces[0] == -1:
            self._edges[edge_idx].faces[0] = face_idx
        else:
            assert self._edges[edge_idx].faces[1] == -1
            self._edges[edge_idx].faces[1] = face_idx

    cdef int64_t _get_edge_id_from_vertex_pair(self, int64_t v1, int64_t v2):
        return self._vertex_pair_to_edge[self._create_vertex_pair_index(v1, v2)]

    cdef int64_t _create_vertex_pair_index(self, int64_t v1, int64_t v2):
        cdef int64_t num_vertices = self._vertices.size()
        if v1 < v2:
            return v1 * num_vertices + v2
        else:
            return v2 * num_vertices + v1

    cdef void _check_edge_to_faces(self):
        cdef:
            EdgeStruct edge
            size_t edge_idx
            int64_t v1, v2

        cdef size_t num_invalid_faces = 0
        for edge_idx in range(self._edges.size()):
            if self._edges[edge_idx].faces[0] == -1:
                num_invalid_faces += 1

        print(f'#edges: {self._edges.size()}, #invalid-faces: {num_invalid_faces}')

    cdef const vector[int64_t] get_faces_of_vertex(self, int64_t vertex_idx) noexcept:
        """Return the faces containing a specific vertex."""
        return self._vertex_to_faces[vertex_idx]
    
    cdef const vector[int64_t] get_faces_of_edge(self, int64_t edge_idx) noexcept:
        """Return the faces containing a specific edge."""
        return self._edge_to_faces[edge_idx]
    
    cdef const vector[int64_t] get_edges_of_vertex(self, int64_t vertex_idx) noexcept:
        """Return the edges connected to a specific vertex."""
        return self._vertex_to_edges[vertex_idx]
    
    cdef vector[int64_t] get_edges_of_face(self, int64_t face_idx) noexcept:
        """Return the edges of a specific face."""
        return [self._faces[face_idx].edges[0],
                self._faces[face_idx].edges[1], 
                self._faces[face_idx].edges[2]]
    
    cdef vector[int64_t] get_vertices_of_face(self, int64_t face_idx) noexcept:
        """Return the vertices of a specific face."""
        return [self._faces[face_idx].vertices[0],
                self._faces[face_idx].vertices[1], 
                self._faces[face_idx].vertices[2]]
    
    cdef vector[int64_t] get_vertices_of_edge(self, int64_t edge_idx) noexcept:
        """Return the vertices of a specific edge."""
        return [self._edges[edge_idx].vertices[0], 
                self._edges[edge_idx].vertices[1]]


def test():
    t0 = time.process_time()
    print('read...')
    trimesh = trimeshlib.load('../../stl-files/KLP_Lame_Tilted.stl')

    cdef int num_vertices = trimesh.vertices.shape[0]
    cdef int num_faces = trimesh.faces.shape[0]
    cdef int num_edges = trimesh.edges_unique.shape[0]

    t1 = time.process_time()
    print(f'... {t1 - t0} s.')

    print('build StlMesh...')
    stl_mesh = StlMesh(trimesh)
    t2 = time.process_time()
    print(f'... {t2 - t1} s.')


@dataclass
class PolyLine:
    vertex_indices: list[int]  # sorted vertices
    edge_indices: list[int]     # sorted edges (len(edges) == len(vertices) - 1)



cdef class PolyLineMiner:
    # cdef:
    #     # cnp.ndarray[cnp.float64_t, ndim=1] _data
    #     object _data

#    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] array):
#        self._data = np.ascontiguousarray(array, dtype=np.float64)

    def __init__(self, trimesh: Trimesh):
        self._stl_mesh = StlMesh(trimesh)

    def sum(self) -> float:
        cdef:
            cnp.float64_t total
            Py_ssize_t i, n 

        total = 0.0
        n = self._data.shape[0]

        for i in range(n):
            total += self._data[i]

        return total

