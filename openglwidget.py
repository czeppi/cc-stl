from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import trimesh
from OpenGL.GL import *
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow
from PySide6.QtOpenGL import QOpenGLBuffer
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from camera import Camera
from itemdetectoratmousepos import ItemDetectorAtMousePos, MeshItemType
from shaders import FacesShaderProgram, EdgesShaderProgram, VerticesShaderProgram


GL_BACKGROUND_COLOR = [1.0, 1.0, 1.0]
GL_VIEW_SIZE = 1200, 900


@dataclass
class MouseData:
    last_position: QPoint = field(default_factory=QPoint)
    is_right_button_pressed: bool = False


class OpenGlWin(QOpenGLWidget):

    def __init__(self, mesh: trimesh.Trimesh, parent: QWindow):
        super().__init__()
        self._mesh = mesh
        self._edges_array = np.array(self._mesh.edges_unique, dtype=np.uint32)  # self._mesh.edges_unique

        self._faces_shader_program = FacesShaderProgram(self._mesh)
        self._edges_shader_program = EdgesShaderProgram(self._mesh)
        self._vertices_shader_program = VerticesShaderProgram(self._mesh)
        self._positions_vbo: Optional[QOpenGLBuffer] = None
        self._normals_vbo: Optional[QOpenGLBuffer] = None

        self._projection_matrix = QMatrix4x4()
        self._view_size: Optional[Tuple[int, int]] = None

        # camera
        self._camera = Camera(distance=30.0, azimuth=0.0, elevation=45.0)
        self._mouse_data = MouseData()

        self.setMouseTracking(True)

    @staticmethod
    def _rotate_mesh_90degree_around_x_axis(mesh: trimesh.Trimesh) -> None:
        """ rotate mesh, so that z axis looks up instead of y axis
        """
        rot_mat = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot_mat)

    def initializeGL(self):
        self._faces_shader_program.link()
        self._edges_shader_program.link()
        self._vertices_shader_program.link()

        self._positions_vbo = self._create_positions_vbo_buffer(np.array(self._mesh.vertices, dtype=np.float32))
        self._normals_vbo = self._create_normals_vbo_buffer(np.array(self._mesh.vertex_normals, dtype=np.float32))

        self._faces_shader_program.init(positions_vbo=self._positions_vbo, normals_vbo=self._normals_vbo)
        self._edges_shader_program.init(positions_vbo=self._positions_vbo)
        self._vertices_shader_program.init(positions_vbo=self._positions_vbo)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*GL_BACKGROUND_COLOR,1.0)

    @staticmethod
    def _create_positions_vbo_buffer(vertices: np.array) -> QOpenGLBuffer:
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(vertices.tobytes(), vertices.nbytes)
        vbo.release()
        return vbo

    @staticmethod
    def _create_normals_vbo_buffer(vertices: np.array) -> QOpenGLBuffer:
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(vertices.tobytes(), vertices.nbytes)
        vbo.release()
        return vbo

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        aspect_ratio = width / height
        self._projection_matrix = self._camera.create_perspective_matrix(aspect_ratio)
        self._view_size = width, height

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # mvp_matrix
        model_matrix = self._create_eye_matrix()
        view_matrix = self._camera.create_view_matrix()
        mvp_matrix = self._projection_matrix * view_matrix * model_matrix

        item_detector = ItemDetectorAtMousePos(mesh=self._mesh, mvp_matrix=mvp_matrix, view_size=self._view_size)
        cur_item = item_detector.find_cur_item(self._mouse_data.last_position)

        # paint
        num_faces = len(self._mesh.faces)
        face_index_array = np.arange(num_faces)
        if cur_item and cur_item.type == MeshItemType.FACE:
            face_index_array = np.delete(face_index_array, np.where(face_index_array == cur_item.index))
            self._faces_shader_program.paint(camera=self._camera, mvp_matrix=mvp_matrix,
                                             face_index_array=face_index_array)

            face_index_array = np.array([cur_item.index])
            self._faces_shader_program.paint(camera=self._camera, mvp_matrix=mvp_matrix,
                                             face_index_array=face_index_array,
                                             color=(0.0, 1.0, 0.0))
        else:
            self._faces_shader_program.paint(camera=self._camera, mvp_matrix=mvp_matrix,
                                             face_index_array=face_index_array)

        num_edges = len(self._mesh.edges_unique)
        edge_index_array = np.arange(num_edges)
        if cur_item and cur_item.type == MeshItemType.EDGE:
            edge_index_array = np.delete(edge_index_array, np.where(edge_index_array == cur_item.index))
            self._edges_shader_program.paint(mvp_matrix=mvp_matrix,
                                             edge_index_array=edge_index_array)

            edge_index_array = np.array([cur_item.index])
            self._edges_shader_program.paint(mvp_matrix=mvp_matrix,
                                             edge_index_array=edge_index_array,
                                             color=(0.0, 1.0, 0.0))
        else:
            self._edges_shader_program.paint(mvp_matrix=mvp_matrix,
                                             edge_index_array=edge_index_array)

        self._edges_shader_program.paint(mvp_matrix=mvp_matrix)

        if cur_item and cur_item.type == MeshItemType.VERTEX:
            self._vertices_shader_program.set_selected_vertices([cur_item.index])
            self._vertices_shader_program.paint(mvp_matrix=mvp_matrix)

    @staticmethod
    def _create_eye_matrix() -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        )

    def mousePressEvent(self, event):
        mouse_pos = event.position().toPoint()
        if event.button() == Qt.RightButton:
            self._mouse_data.last_position = mouse_pos
            self._mouse_data.is_right_button_pressed = True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._mouse_data.is_right_button_pressed = False

    def mouseMoveEvent(self, event):
        mouse_pos = event.position().toPoint()
        if self._mouse_data.is_right_button_pressed:
            sensitivity = self._camera.distance / 200

            delta = mouse_pos - self._mouse_data.last_position
            d_azimuth = delta.x() * sensitivity
            d_elevation = delta.y() * sensitivity

            self._camera.azimuth += d_azimuth
            self._camera.elevation += d_elevation

            self._mouse_data.last_position = mouse_pos
            self.update()  # update screen
        else:
            self._mouse_data.last_position = mouse_pos
            #print(f'mouse: {mouse_pos}')
            self.update()  # update screen, for selected elements

    def wheelEvent(self, event):
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._camera.distance *= zoom_factor

        # fov_step = 2
        # fov_delta = -fov_step if event.angleDelta().y() > 0 else fov_step
        # self._camera.fov += fov_delta
        # print(f'fov={self._camera.fov}')

        width, height = self._view_size
        aspect_ratio = width / height
        self._projection_matrix = self._camera.create_perspective_matrix(aspect_ratio)

        self.update()  # update screen