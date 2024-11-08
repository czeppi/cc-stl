from __future__ import annotations

import sys
from typing import Optional, Tuple

import numpy as np
import trimesh
from OpenGL.GL import *
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow
from PySide6.QtOpenGL import QOpenGLBuffer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow

from camera import Camera
from shaders import FacesShaderProgram, EdgesShaderProgram, VerticesShaderProgram


class OpenGlWin(QOpenGLWidget):

    def __init__(self, stl_path: str, parent: QWindow):
        super().__init__()
        self._mesh = self._read_mesh(stl_path)
        self._selected_vertex_indices = [20000]
        self._edges_array = np.array(self._mesh.edges_unique, dtype=np.uint32)  # self._mesh.edges_unique

        self._faces_shader_program = FacesShaderProgram(self._mesh)
        self._edges_shader_program = EdgesShaderProgram(self._mesh)
        self._vertices_shader_program = VerticesShaderProgram(self._mesh,
                                                              selected_indices=self._selected_vertex_indices)
        self._positions_vbo: Optional[QOpenGLBuffer] = None
        self._normals_vbo: Optional[QOpenGLBuffer] = None

        self._projection_matrix = QMatrix4x4()
        self._view_size: Optional[Tuple[int, int]] = None

        # camera
        self._camera = Camera(distance=50.0, azimuth=180.0, elevation=45.0)
        self._last_mouse_position = QPoint()
        self._is_right_button_pressed = False

    def _read_mesh(self, stl_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path)
        self._rotate_mesh_90degree_around_x_axis(mesh)
        return mesh

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
        glClearColor(1.0, 1.0, 1.0,1.0)

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
        self._projection_matrix = self._create_perspective_matrix(45.0, aspect_ratio, 0.1, 100.0)
        self._view_size = width, height

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # mvp_matrix
        model_matrix = self._create_eye_matrix()
        view_matrix = self._camera.create_view_matrix()
        mvp_matrix = self._projection_matrix * view_matrix * model_matrix
        self._print_sel_vertices_info(mvp_matrix)

        # paint
        self._faces_shader_program.paint(camera=self._camera, mvp_matrix=mvp_matrix)
        #self._edges_shader_program.paint(mvp_matrix=mvp_matrix)
        self._vertices_shader_program.paint(mvp_matrix=mvp_matrix)

    def _print_sel_vertices_info(self, mvp_matrix: QMatrix4x4) -> None:
        vertex_index = self._selected_vertex_indices[0]  # take only one
        vertex_pos = self._mesh.vertices[vertex_index]
        vertex_pos4d = np.array([vertex_pos[0], vertex_pos[1], vertex_pos[2], 1])

        m = np.array([[mvp_matrix[0,0], mvp_matrix[0,1], mvp_matrix[0,2], mvp_matrix[0,3]],
                      [mvp_matrix[1,0], mvp_matrix[1,1], mvp_matrix[1,2], mvp_matrix[1,3]],
                      [mvp_matrix[2,0], mvp_matrix[2,1], mvp_matrix[2,2], mvp_matrix[2,3]],
                      [mvp_matrix[3,0], mvp_matrix[3,1], mvp_matrix[3,2], mvp_matrix[3,3]]])

        vertex_pos_transformed = m @ vertex_pos4d
        x1, y1, z1, w1 = vertex_pos_transformed
        w, h = self._view_size
        x = (w / 2) + x1 / z1 * (w / 2)
        y = (h / 2) - y1 / z1 * (h / 2)
        print(f'{vertex_pos4d}, {x, y}')

        self._transform_all_vertices(mvp_matrix)

    def _transform_all_vertices(self, mvp_matrix: QMatrix4x4) -> None:
        vertices = self._mesh.vertices
        n = len(vertices)
        transpose_vertices = np.transpose(vertices)
        vertices_matrix = np.vstack([np.transpose(vertices), np.full(n, 1.0)])

        m = np.array([[mvp_matrix[0,0], mvp_matrix[0,1], mvp_matrix[0,2], mvp_matrix[0,3]],
                      [mvp_matrix[1,0], mvp_matrix[1,1], mvp_matrix[1,2], mvp_matrix[1,3]],
                      [mvp_matrix[2,0], mvp_matrix[2,1], mvp_matrix[2,2], mvp_matrix[2,3]],
                      [mvp_matrix[3,0], mvp_matrix[3,1], mvp_matrix[3,2], mvp_matrix[3,3]]])

        x1_vec, y1_vec, z1_vec, w1_vec = m @ vertices_matrix
        width2 = self._view_size[0] / 2.0
        height2 = self._view_size[1] / 2.0
        x2_vec = width2 + x1_vec / z1_vec * width2
        y2_vec = height2 - y1_vec / z1_vec * height2

        i = self._selected_vertex_indices[0]  # take only one
        #sel_transformed =voidertex_pos_transformed[vertex_index]
        print(f'sel_transformed: {x2_vec[i]}, {y2_vec[i]}, {z1_vec[i]}')
        print()

    @staticmethod
    def _create_eye_matrix() -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        )

    @staticmethod
    def _create_perspective_matrix(fov, aspect, near, far) -> QMatrix4x4:
        f = 1.0 / np.tan(np.radians(fov) / 2)
        return QMatrix4x4(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
            0, 0, -1, 0
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_right_button_pressed = True
            self._last_mouse_position = event.position().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_right_button_pressed = False

    def mouseMoveEvent(self, event):
        if self._is_right_button_pressed:
            sensitivity = self._camera.distance / 200

            delta = event.position().toPoint() - self._last_mouse_position
            d_azimuth = -delta.x() * sensitivity
            d_elevation = delta.y() * sensitivity

            self._camera.azimuth += d_azimuth
            self._camera.elevation += d_elevation

            self._last_mouse_position = event.position().toPoint()
            self.update()  # update screen

    def wheelEvent(self, event):
        # adapt zoom-factor (p.e., 1.1 for fast zoom or 1.05 for slow zoom)
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._camera.distance *= zoom_factor
        self.update()  # update screen


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        self.setFixedSize(1200, 900)
        self._open_gl_widget = OpenGlWin(stl_path="KLP_Lame_Tilted.stl", parent=self)
        self.setCentralWidget(self._open_gl_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())