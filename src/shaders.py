from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import trimesh
from OpenGL.GL import *

from PySide6.QtGui import QVector3D

from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader, QOpenGLVertexArrayObject, QOpenGLBuffer

from camera import Camera


DEFAULT_FACET_COLOR = 0.4, 0.4, 0.8
DEFAULT_EDGE_COLOR = 0.0, 0.0, 0.0
ColorTuple = Tuple[float, float, float]  # all values in [0, 1]


FACES_VERTEX_SHADER_SRC = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal; 

    uniform mat4 mvp_matrix;

    out vec3 fragNormal;
    out vec3 fragPosition;

    void main() {
        // fragPosition = vec3(model * vec4(position, 1.0));
        // fragNormal = mat3(transpose(inverse(model))) * normal;
        fragPosition = position;
        fragNormal = vec3(normal);
        
        gl_Position = mvp_matrix * vec4(fragPosition, 1.0);
    }
"""

FACES_FRAGMENT_SHADER_SRC = """
    #version 330 core
    in vec3 fragNormal;
    in vec3 fragPosition;

    uniform vec3 ambientColor;    // Umgebungslichtfarbe
    uniform vec3 cameraPos;
    uniform vec3 objectColor; 

    out vec4 fragColor;

    void main() {
        vec3 normal = normalize(fragNormal);
        vec3 viewDir = normalize(cameraPos - fragPosition);
        float intensity = max(dot(viewDir, normal), 0.0);
        vec3 result = ambientColor * objectColor * intensity ;  // todo: offset is nor correct

        fragColor = vec4(result, 1.0);
    }
"""

EDGES_VERTEX_SHADER_SRC = """
    #version 330
    layout(location = 0) in vec3 position;
    uniform mat4 mvp_matrix;
    void main() {
        gl_Position = mvp_matrix * vec4(position, 1.0);
    }
"""

EDGES_FRAGMENT_SHADER_SRC = """
    #version 330
    uniform vec3 edgeColor; 
        
    out vec4 fragColor;
    void main() {
        fragColor = vec4(edgeColor, 1.0);  // blck
    }
"""

VERTICES_VERTEX_SHADER_SRC = """
    #version 330
    layout(location = 0) in vec3 position;
    uniform mat4 mvp_matrix;
    void main() {
        gl_Position = mvp_matrix * vec4(position, 1.0);
    }
"""

VERTICES_FRAGMENT_SHADER_SRC = """
    #version 330
    out vec4 fragColor;
    void main() {
        fragColor = vec4(0.0, 1.0, 0.0, 1.0);  // selected color for vertices
    }
"""


class ShaderProgram:

    def __init__(self, vertex_shader_src: str, fragment_shader_src: str):
        self._vertex_shader_src = vertex_shader_src
        self._fragment_shader_src = fragment_shader_src
        self._gl_program: Optional[QOpenGLShaderProgram] = None

    def link(self) -> None:
        self._gl_program = QOpenGLShaderProgram()
        self._gl_program.addShaderFromSourceCode(QOpenGLShader.Vertex, self._vertex_shader_src)
        self._gl_program.addShaderFromSourceCode(QOpenGLShader.Fragment, self._fragment_shader_src)
        self._gl_program.link()


class FacesShaderProgram(ShaderProgram):

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__(vertex_shader_src=FACES_VERTEX_SHADER_SRC, fragment_shader_src=FACES_FRAGMENT_SHADER_SRC)
        self._mesh = mesh

        self._positions_vbo = None
        self._normals_vbo = None

        self._vao = QOpenGLVertexArrayObject()
        self._ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)

    def init(self, positions_vbo: QOpenGLBuffer, normals_vbo: QOpenGLBuffer) -> None:
        self._positions_vbo = positions_vbo
        self._normals_vbo = normals_vbo

        prg = self._gl_program
        vao = self._vao = QOpenGLVertexArrayObject()
        ebo = self._ebo = self._create_ebo()

        vao.create()
        vao.bind()

        positions_vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        normals_vbo.bind()
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        ebo.bind()

        prg.bind()
        prg.enableAttributeArray(0)
        prg.enableAttributeArray(1)

        # set light parameters
        prg.setUniformValue("ambientColor", QVector3D(1.0, 1.0, 1.0))
        #prg.setUniformValue("ambientColor", QVector3D(0.5, 0.5, 0.5))  # Gedimmtes Umgebungslicht
        prg.setUniformValue("objectColor", QVector3D(*DEFAULT_FACET_COLOR))  # object color

        vao.release()
        normals_vbo.release()
        ebo.release()
        positions_vbo.release()
        prg.release()

    def _create_ebo(self) -> QOpenGLBuffer:
        faces = np.array(self._mesh.faces, dtype=np.uint32)  # important!

        ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ebo.create()
        ebo.bind()
        ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        ebo.allocate(faces.tobytes(), faces.nbytes)
        ebo.release()
        return ebo

    def paint(self, camera: Camera, mvp_matrix: np.array,
              color: Optional[ColorTuple]=None,
              face_index_array: Optional[np.array]=None) -> None:
        prg = self._gl_program
        vao = self._vao

        ebo_array = np.array(self._mesh.faces, dtype=np.uint32)
        if face_index_array is not None and len(face_index_array) > 0:
            ebo_array = ebo_array[face_index_array]

        ebo = self._ebo
        ebo.bind()
        ebo.allocate(ebo_array.tobytes(), ebo_array.nbytes)
        ebo.release()

        prg.bind()
        prg.setUniformValue("mvp_matrix", mvp_matrix)
        prg.setUniformValue("cameraPos", QVector3D(*camera.xyz))

        if color is None:
            color = DEFAULT_FACET_COLOR
        prg.setUniformValue("objectColor", QVector3D(*color))

        vao.bind()

        glDrawElements(GL_TRIANGLES, ebo_array.size, GL_UNSIGNED_INT, None)

        vao.release()
        prg.release()


class EdgesShaderProgram(ShaderProgram):

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__(vertex_shader_src=EDGES_VERTEX_SHADER_SRC, fragment_shader_src=EDGES_FRAGMENT_SHADER_SRC)
        self._mesh = mesh
        self._edges_array = np.array(self._mesh.edges_unique, dtype=np.uint32)

        self._vao = QOpenGLVertexArrayObject()
        self._ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)

    def init(self, positions_vbo: QOpenGLBuffer) -> None:
        prg = self._gl_program
        vao = self._vao = QOpenGLVertexArrayObject()
        ebo = self._ebo = self._create_ebo()

        vao.create()
        vao.bind()

        positions_vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        ebo.bind()

        prg.bind()
        prg.enableAttributeArray(0)

        vao.release()
        ebo.release()
        positions_vbo.release()
        prg.release()

    def _create_ebo(self) -> QOpenGLBuffer:
        edges = self._edges_array

        ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ebo.create()
        ebo.bind()
        ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        ebo.allocate(edges.tobytes(), edges.nbytes)
        ebo.release()
        return ebo

    def paint(self, mvp_matrix: np.array,
              color: Optional[ColorTuple]=None,
              edge_index_array: Optional[np.array]=None) -> None:
        prg = self._gl_program
        vao = self._vao

        ebo_array = self._edges_array
        if edge_index_array is not None and len(edge_index_array) > 0:
            ebo_array = ebo_array[edge_index_array]

        ebo = self._ebo
        ebo.bind()
        ebo.allocate(ebo_array.tobytes(), ebo_array.nbytes)
        ebo.release()

        #glDepthMask(GL_FALSE)  # depth buffer writing deactivate
        #glEnable(GL_POLYGON_OFFSET_LINE)  # no effect
        #glPolygonOffset(-1.0, -1.0)  # no effect
        glLineWidth(2.0)

        prg.bind()
        prg.setUniformValue("mvp_matrix", mvp_matrix)

        if color is None:
            color = DEFAULT_EDGE_COLOR
        prg.setUniformValue("edgeColor", QVector3D(*color))

        vao.bind()

        glDrawElements(GL_LINES, ebo_array.size, GL_UNSIGNED_INT, None)

        vao.release()
        prg.release()

        #glDepthMask(GL_TRUE)  # depth buffer writing reactivate
        #glDisable(GL_POLYGON_OFFSET_LINE)


class VerticesShaderProgram(ShaderProgram):

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__(vertex_shader_src=VERTICES_VERTEX_SHADER_SRC, fragment_shader_src=VERTICES_FRAGMENT_SHADER_SRC)
        self._mesh = mesh
        self._selected_indices = []

        self._vao = QOpenGLVertexArrayObject()
        self._ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)

    def init(self, positions_vbo: QOpenGLBuffer) -> None:
        prg = self._gl_program
        vao = self._vao = QOpenGLVertexArrayObject()
        ebo = self._ebo = self._create_ebo()

        vao.create()
        vao.bind()

        positions_vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        ebo.bind()

        prg.bind()
        prg.enableAttributeArray(0)

        vao.release()
        ebo.release()
        positions_vbo.release()
        prg.release()

    @staticmethod
    def _create_ebo() -> QOpenGLBuffer:
        ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ebo.create()
        ebo.bind()
        ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        ebo.release()
        return ebo

    def set_selected_vertices(self, selected_indices: List[int]) -> None:
        if len(selected_indices) > 0:
            ebo = self._ebo
            ebo.bind()

            vertex_indices = np.array(selected_indices)
            ebo.allocate(vertex_indices.tobytes(), vertex_indices.nbytes)
            ebo.release()

        self._selected_indices = selected_indices

    def paint(self, mvp_matrix: np.array) -> None:
        if len(self._selected_indices) == 0:
            return

        prg = self._gl_program
        vao = self._vao

        glPointSize(5.0)

        prg.bind()
        prg.setUniformValue("mvp_matrix", mvp_matrix)
        vao.bind()

        n = len(self._selected_indices)
        glDrawElements(GL_POINTS, n, GL_UNSIGNED_INT, None)

        vao.release()
        prg.release()
